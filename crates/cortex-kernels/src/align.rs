//! 64-byte aligned owned buffers.
//!
//! CPU cache lines on the target laptops are 64 bytes. Every packed weight
//! block (`CacheLine16x16`) and every hot parameter / state slab that this
//! crate hands out is allocated at that alignment so a tile never straddles
//! a line. `Vec<T>` is used only when `align_of::<T>() >= 64`.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Owned buffer whose allocation is aligned to at least 64 bytes.
pub struct Aligned64<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    layout: Layout,
    _pd: PhantomData<T>,
}

unsafe impl<T: Send> Send for Aligned64<T> {}
unsafe impl<T: Sync> Sync for Aligned64<T> {}

impl<T: Copy> Aligned64<T> {
    pub fn zeros(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        let align = 64.max(std::mem::align_of::<T>());
        let layout =
            Layout::from_size_align(len * std::mem::size_of::<T>(), align).expect("aligned layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        debug_assert_eq!(ptr.as_ptr() as usize % 64, 0);
        Self {
            ptr: ptr.cast(),
            len,
            cap: len,
            layout,
            _pd: PhantomData,
        }
    }

    pub fn from_slice(src: &[T]) -> Self {
        let mut buf = Self::zeros(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            layout: Layout::from_size_align(0, 64).unwrap_or_else(|_| Layout::new::<u8>()),
            _pd: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const T {
        if self.len == 0 {
            NonNull::dangling().as_ptr()
        } else {
            self.ptr.as_ptr()
        }
    }

    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    pub fn is_cache_aligned(&self) -> bool {
        self.len == 0 || (self.ptr.as_ptr() as usize) % 64 == 0
    }

    pub fn fill(&mut self, value: T) {
        for slot in self.as_mut_slice() {
            *slot = value;
        }
    }

    pub fn copy_from(&mut self, src: &Aligned64<T>) {
        assert_eq!(self.len, src.len);
        self.as_mut_slice().copy_from_slice(src.as_slice());
    }
}

impl<T: Copy> Clone for Aligned64<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<T> Drop for Aligned64<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) }
        }
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for Aligned64<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Aligned64")
            .field("len", &self.len)
            .field("align_ok", &self.is_cache_aligned())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_64_byte_aligned() {
        let buf = Aligned64::<f32>::zeros(17);
        assert!(buf.is_cache_aligned());
        assert_eq!(buf.len(), 17);
        assert!(buf.as_slice().iter().all(|v| *v == 0.0));
        let ptr = buf.as_ptr() as usize;
        assert_eq!(ptr % 64, 0);
    }
}
