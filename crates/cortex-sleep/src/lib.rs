//! Nightly / idle tuning.
//!
//! MeZO touches only the caller's φ (row scales and RMSNorm weights). The
//! perturbation `z` is a Rademacher vector regenerated from the same seed on
//! every pass, so the implementation does not allocate a gradient buffer and
//! does not build an autograd graph. Peak memory stays at the inference
//! footprint plus the live activations of the forward that computes the loss.
//!
//! Replay draws 80% from the WAL and 20% from a fixed anchor set
//! (`index % 5 == 4` is an anchor).

mod mezo;
mod replay;

pub use mezo::{mezo_3pass, rademacher, InPlacePhi};
pub use replay::{pick_source, take, ReplaySource};
