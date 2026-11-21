//! Local continual-learning runtime (`cortex-runtime`).
//!
//! Combines the macro-cycle 1.58-bit `CortexModel`, `.ctxb` loader, dense
//! special-token byte-fallback BPE tokenizer, constrained JSON `ToolCallFsm`,
//! sandboxed `AgentLoop` tool suite, day/night `RuntimeFsm`, L2 `ExpertRouter`,
//! L3 `traces`, and local `RwLock`-concurrent HTTP/Unix server.

pub mod api;
pub mod ctxb;
pub mod expert;
pub mod family;
pub mod fsm;
pub mod generate;
pub mod grammar;
pub mod harness;
pub mod model;
pub mod sample;
pub mod tokenizer;
pub mod tools;
pub mod traces;

pub use ctxb::{from_ctxb, load_ctxb, save_ctxb, to_ctxb};
pub use fsm::{FsmEvent, FsmState, RuntimeFsm};
pub use model::{CortexConfig, CortexModel, MemoryBlock, ModelState};
pub use tokenizer::Tokenizer;
pub use tools::{AgentLoop, AgentOutcome, ToolRegistry};
