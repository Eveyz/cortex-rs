//! Lock-free double-buffered weight arena with Dekker `SeqCst` reader drain guard.
//!
//! Two `UnsafeCell` slots, `active_idx`, and `active_readers[2]`.
//! Fast-path readers never take a mutex: they bump `active_readers[idx]` with
//! `SeqCst`, re-verify `active_idx` (`SeqCst`) to prevent TOCTOU races across
//! rapid flips, read `&*slots[idx]`, and release the count on exit.
//! The single background writer waits (`SeqCst`) until `active_readers[inactive] == 0`
//! before mutating the inactive slot in place (zero heap allocation), then
//! publishes `active_idx` with `SeqCst`.

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
        let shadow = weights.clone();
        Self {
            slots: [UnsafeCell::new(weights), UnsafeCell::new(shadow)],
            active_idx: AtomicUsize::new(0),
            active_readers: [AtomicUsize::new(0), AtomicUsize::new(0)],
        }
    }

    /// Lock-free fast-path read. Retries if the writer flipped `active_idx`
    /// between the initial load and reader registration.
    #[inline]
    pub fn read<R>(&self, f: impl FnOnce(&HippocampusWeights) -> R) -> R {
        loop {
            let idx = self.active_idx.load(Ordering::Acquire);
            self.active_readers[idx].fetch_add(1, Ordering::SeqCst);
            let idx2 = self.active_idx.load(Ordering::SeqCst);
            if idx == idx2 {
                let out = unsafe { f(&*self.slots[idx].get()) };
                self.active_readers[idx].fetch_sub(1, Ordering::Release);
                return out;
            }
            self.active_readers[idx].fetch_sub(1, Ordering::Release);
            std::hint::spin_loop();
        }
    }

    /// Snapshot clone of the currently active weights.
    pub fn snapshot_active(&self) -> HippocampusWeights {
        self.read(|w| w.clone())
    }

    /// Publish updated working weights from the single background plasticity
    /// thread into the inactive slot (zero heap allocation) and atomically flip.
    pub fn publish_from_writer(&self, updated: &HippocampusWeights) {
        let active = self.active_idx.load(Ordering::Acquire);
        let shadow = 1 - active;
        self.wait_readers(shadow);
        unsafe {
            let dst = &mut *self.slots[shadow].get();
            dst.copy_params_from(updated);
        }
        self.active_idx.store(shadow, Ordering::SeqCst);
    }

    /// Mutate the inactive snapshot directly (single-writer synchronous API).
    pub fn write_shadow<R>(&self, f: impl FnOnce(&mut HippocampusWeights) -> R) -> R {
        let active = self.active_idx.load(Ordering::Acquire);
        let shadow = 1 - active;
        self.wait_readers(shadow);
        unsafe { f(&mut *self.slots[shadow].get()) }
    }

    /// Publish the shadow slot, then sync it onto the retired slot in place
    /// (zero heap allocation) after in-flight readers drain.
    pub fn flush(&self) {
        let active = self.active_idx.load(Ordering::Acquire);
        let shadow = 1 - active;
        self.wait_readers(shadow);
        self.active_idx.store(shadow, Ordering::SeqCst);
        self.wait_readers(active);
        unsafe {
            let src = &*self.slots[shadow].get();
            let dst = &mut *self.slots[active].get();
            dst.copy_params_from(src);
        }
    }

    pub fn active_index(&self) -> usize {
        self.active_idx.load(Ordering::Acquire)
    }

    /// Replace both slots atomically once readers drain (used by snapshot rollback).
    pub fn replace(&self, weights: HippocampusWeights) {
        self.wait_readers(0);
        self.wait_readers(1);
        let shadow = weights.clone();
        unsafe {
            *self.slots[0].get() = weights;
            *self.slots[1].get() = shadow;
        }
        self.active_idx.store(0, Ordering::SeqCst);
    }

    fn wait_readers(&self, idx: usize) {
        let mut spins = 0u32;
        while self.active_readers[idx].load(Ordering::SeqCst) != 0 {
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn publish_is_visible_and_concurrent_readers_never_tear() {
        let arena = Arc::new(SharedWeightArena::new(HippocampusWeights::new(16, 32, 16)));
        assert_eq!(arena.read(|w| w.version), 0);

        let mut handles = Vec::new();
        for _ in 0..4 {
            let arena_r = Arc::clone(&arena);
            handles.push(std::thread::spawn(move || {
                let q = [0.25f32; 16];
                let mut out = [0.0f32; 16];
                for _ in 0..500 {
                    let (ver, first, last) = arena_r.read(|w| {
                        w.forward_into(&q, &mut out);
                        (
                            w.version,
                            w.w1.as_slice()[0],
                            *w.w1.as_slice().last().unwrap(),
                        )
                    });
                    assert!(out.iter().all(|v| v.is_finite()));
                    if ver > 0 {
                        // Writer sets all of w1 to `ver as f32 * 0.01`; torn reads would mismatch!
                        assert!(
                            (first - last).abs() < 1e-6,
                            "torn read detected: ver={ver} first={first} last={last}"
                        );
                    }
                }
            }));
        }

        let mut working = HippocampusWeights::new(16, 32, 16);
        for step in 1..=50u64 {
            working.w1.fill(step as f32 * 0.01);
            working.version = step;
            arena.publish_from_writer(&working);
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(arena.read(|w| w.version), 50);
    }
}
