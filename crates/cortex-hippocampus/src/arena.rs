//! Lock-free double-buffered weight arena.
//!
//! Two `UnsafeCell` slots, `active_idx`, and `active_readers[2]`.
//! Fast-path readers never take a mutex: they bump a reader count, re-check
//! the published index, and drop the count when they leave. The writer mutates
//! only the inactive slot and publishes with a release store. It waits until
//! that slot's readers have drained before reusing it, so an old snapshot
//! stays alive for in-flight readers. This is not an `AtomicPtr` swap (no ABA,
//! no freed slot while a reader is inside it).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::mlp::HippocampusWeights;

pub struct SharedWeightArena {
    slots: [UnsafeCell<HippocampusWeights>; 2],
    active_idx: AtomicUsize,
    active_readers: [AtomicUsize; 2],
}

unsafe impl Send for SharedWeightArena {}
unsafe impl Sync for SharedWeightArena {}

impl SharedWeightArena {
    pub fn new(weights: HippocampusWeights) -> Self {
        let shadow = clone_weights(&weights);
        Self {
            slots: [UnsafeCell::new(weights), UnsafeCell::new(shadow)],
            active_idx: AtomicUsize::new(0),
            active_readers: [AtomicUsize::new(0), AtomicUsize::new(0)],
        }
    }

    /// Fast path. No mutex. Retries only if the publisher flips mid-acquire.
    pub fn read<R>(&self, f: impl FnOnce(&HippocampusWeights) -> R) -> R {
        loop {
            let idx = self.active_idx.load(Ordering::Acquire);
            self.active_readers[idx].fetch_add(1, Ordering::AcqRel);
            let idx2 = self.active_idx.load(Ordering::Acquire);
            if idx == idx2 {
                let out = unsafe { f(&*self.slots[idx].get()) };
                self.active_readers[idx].fetch_sub(1, Ordering::Release);
                return out;
            }
            self.active_readers[idx].fetch_sub(1, Ordering::Release);
        }
    }

    /// Mutate the inactive snapshot. Spins only until readers of *that* slot
    /// drain; the fast path is never the waiter.
    pub fn write_shadow<R>(&self, f: impl FnOnce(&mut HippocampusWeights) -> R) -> R {
        let active = self.active_idx.load(Ordering::Acquire);
        let shadow = 1 - active;
        self.wait_readers(shadow);
        unsafe { f(&mut *self.slots[shadow].get()) }
    }

    /// Publish the shadow slot, then copy it back onto the retired slot so the
    /// next write starts from the weights readers are now using.
    pub fn flush(&self) {
        let active = self.active_idx.load(Ordering::Acquire);
        let shadow = 1 - active;
        self.wait_readers(shadow);
        self.active_idx.store(shadow, Ordering::Release);
        self.wait_readers(active);
        unsafe {
            let cloned = (&*self.slots[shadow].get()).clone();
            let dst = &mut *self.slots[active].get();
            dst.copy_params_from(&cloned);
        }
    }

    pub fn active_index(&self) -> usize {
        self.active_idx.load(Ordering::Acquire)
    }

    /// Replace both slots. Used by checkpoint load and `/reset-mem`.
    pub fn replace(&self, weights: HippocampusWeights) {
        self.wait_readers(0);
        self.wait_readers(1);
        let shadow = weights.clone();
        unsafe {
            *self.slots[0].get() = weights;
            *self.slots[1].get() = shadow;
        }
        self.active_idx.store(0, Ordering::Release);
    }

    fn wait_readers(&self, idx: usize) {
        let mut spins = 0u32;
        while self.active_readers[idx].load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

fn clone_weights(w: &HippocampusWeights) -> HippocampusWeights {
    w.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn publish_is_visible_and_readers_do_not_tear() {
        let arena = Arc::new(SharedWeightArena::new(HippocampusWeights::new(8, 8, 8)));
        let start = arena.read(|w| w.version);
        assert_eq!(start, 0);
        let reader = {
            let arena = Arc::clone(&arena);
            std::thread::spawn(move || {
                for _ in 0..200 {
                    let v = arena.read(|w| w.version);
                    assert!(v == 0 || v == 1);
                }
            })
        };
        arena.write_shadow(|w| {
            w.version = 1;
            w.w2.as_mut_slice()[0] = 0.25;
        });
        arena.flush();
        reader.join().unwrap();
        let seen = arena.read(|w| (w.version, w.w2.as_slice()[0]));
        assert_eq!(seen.0, 1);
        assert!((seen.1 - 0.25).abs() < 1e-6);
    }
}
