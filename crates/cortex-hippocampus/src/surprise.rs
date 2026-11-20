//! Surprise gate.
//!
//! ```text
//! Surprise_t = (1/(2d)) || W2 SiLU(W1 k_t) - v_t ||^2
//! ```
//!
//! * `Surprise < τ_low` — drop. Version does not move. No WAL.
//! * `τ_low ≤ Surprise < τ_high` — local outer-product write, no WAL.
//! * `Surprise ≥ τ_high` — same write, and the caller appends the episode.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurpriseBand {
    Drop,
    Local,
    High,
}

#[derive(Clone, Copy, Debug)]
pub struct SurpriseThresholds {
    pub tau_low: f32,
    pub tau_high: f32,
}

impl Default for SurpriseThresholds {
    fn default() -> Self {
        Self {
            tau_low: 1e-4,
            tau_high: 0.05,
        }
    }
}

impl SurpriseThresholds {
    pub fn band(&self, surprise: f32) -> SurpriseBand {
        if surprise < self.tau_low {
            SurpriseBand::Drop
        } else if surprise < self.tau_high {
            SurpriseBand::Local
        } else {
            SurpriseBand::High
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_bands() {
        let t = SurpriseThresholds {
            tau_low: 0.1,
            tau_high: 0.5,
        };
        assert_eq!(t.band(0.0), SurpriseBand::Drop);
        assert_eq!(t.band(0.099), SurpriseBand::Drop);
        assert_eq!(t.band(0.1), SurpriseBand::Local);
        assert_eq!(t.band(0.49), SurpriseBand::Local);
        assert_eq!(t.band(0.5), SurpriseBand::High);
        assert_eq!(t.band(10.0), SurpriseBand::High);
    }
}
