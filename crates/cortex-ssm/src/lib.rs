//! Linear-time sequence mixers for the macro-cycle trunk.
//!
//! * Layers 1–3 of each macro: [`BitSsmBlock`] (1.58-bit projections, Gated
//!   DeltaNet, 1.58-bit SwiGLU).
//! * Layer 4 branch A: [`RingSlidingAttention`] (fixed `W = 512` INT8 KV ring).
//!
//! Gated DeltaNet state is `[num_heads, head_dim, head_dim]` and does not
//! grow with sequence length.

mod block;
mod deltanet;
mod ring_attn;
mod swiglu;

pub use block::{BitSsmBlock, RmsNorm};
pub use deltanet::{deltanet_chunk, deltanet_step, deltanet_step_matrix_form, DeltaState};
pub use ring_attn::{RingSlidingAttention, RingState};
pub use swiglu::BitSwiGLU;
