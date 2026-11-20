//! Local continual-learning runtime.
//!
//! `cortex-cli` is the binary. The library exposes the model, the `.ctxb`
//! loader, the byte-fallback tokenizer, and sampling so tests can drive a
//! session without spawning a process.

pub mod api;
pub mod ctxb;
pub mod generate;
pub mod model;
pub mod sample;
pub mod tokenizer;

pub use ctxb::{from_ctxb, load_ctxb, save_ctxb, to_ctxb};
pub use model::{CortexConfig, CortexModel};
pub use tokenizer::Tokenizer;
