//! Nightly / idle consolidation engine (`cortex-sleep`).
//!
//! * [`mezo_3pass`] / [`MezoScaleTuner`] — zero-copy in-place 3-pass MeZO on
//!   continuous scales `φ` (row scales + RMSNorm weights), strictly 2 forwards.
//! * [`ReplayBuffer`] / [`take`] — 80% high-surprise WAL + 20% anchor replay.
//! * [`consolidate_hippocampus`] — Gram–Schmidt row deduplication + soft
//!   singular suppression for hippocampus weights.
//! * [`scaffold_expert`] — L2 domain micro-expert scaffolding from surprise logs.

mod mezo;
mod orthogonalize;
mod replay;
mod scaffold;

pub use mezo::{mezo_3pass, rademacher, InPlacePhi, MezoScaleTuner, MezoStepReport};
pub use orthogonalize::{consolidate_hippocampus, orthogonalize_rows, soft_singular_suppress};
pub use replay::{pick_source, take, ReplayBuffer, ReplaySource};
pub use scaffold::{
    scaffold_expert, ExpertScaffold, SurpriseLog, ADAPTER_BYTES_DEFAULT, ADAPTER_BYTES_MAX,
    ADAPTER_BYTES_MIN,
};
