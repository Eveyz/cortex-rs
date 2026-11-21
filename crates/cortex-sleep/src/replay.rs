//! Sleep-time replay mixer: 80% high-surprise WAL episodes + 20% anti-forgetting anchors.

use std::path::Path;

use cortex_hippocampus::{Episode, WalArchive};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplaySource {
    Wal,
    Anchor,
}

/// Deterministic 80/20 schedule: `index % 5 == 4` selects `Anchor` (20%).
pub fn pick_source(index: usize) -> ReplaySource {
    if index % 5 == 4 {
        ReplaySource::Anchor
    } else {
        ReplaySource::Wal
    }
}

/// Mix a WAL pool with a fixed anchor pool. Empty pools fall back to the other.
pub fn take<'a, T>(wal: &'a [T], anchors: &'a [T], index: usize) -> Option<&'a T> {
    match pick_source(index) {
        ReplaySource::Anchor if !anchors.is_empty() => Some(&anchors[index % anchors.len()]),
        ReplaySource::Wal if !wal.is_empty() => Some(&wal[index % wal.len()]),
        ReplaySource::Anchor => wal
            .get(index % wal.len().max(1))
            .filter(|_| !wal.is_empty()),
        ReplaySource::Wal => anchors
            .get(index % anchors.len().max(1))
            .filter(|_| !anchors.is_empty()),
    }
}

/// Stateful replay buffer loading daytime high-surprise WAL episodes and mixing
/// them 80/20 with anti-forgetting anchor sequences for nightly MeZO & consolidation.
#[derive(Clone, Debug, Default)]
pub struct ReplayBuffer {
    pub wal_episodes: Vec<Episode>,
    pub anchors: Vec<Episode>,
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load high-surprise episodes from a `WAL1` file, sorted by surprise descending.
    pub fn load_wal(&mut self, wal_file: &Path) -> std::io::Result<usize> {
        let (mut eps, _trunc) = WalArchive::read_file(wal_file)?;
        eps.sort_by(|a, b| {
            b.surprise
                .partial_cmp(&a.surprise)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = eps.len();
        self.wal_episodes.extend(eps);
        Ok(n)
    }

    pub fn add_anchor(&mut self, ep: Episode) {
        self.anchors.push(ep);
    }

    pub fn add_anchor_tokens(&mut self, tokens: Vec<u32>) {
        self.anchors.push(Episode {
            surprise: 0.0,
            key: Vec::new(),
            value: Vec::new(),
            context: tokens,
        });
    }

    /// Draw `batch_size` episodes at an 80% WAL / 20% anchor ratio (when both
    /// pools are populated), returning `(batch, actual_wal_fraction)`.
    pub fn mixed_batch(&self, batch_size: usize) -> (Vec<Episode>, f32) {
        if batch_size == 0 || (self.wal_episodes.is_empty() && self.anchors.is_empty()) {
            return (Vec::new(), 0.0);
        }
        let mut out = Vec::with_capacity(batch_size);
        let mut wal_count = 0usize;
        for i in 0..batch_size {
            if let Some(ep) = take(&self.wal_episodes, &self.anchors, i) {
                if !self.wal_episodes.is_empty()
                    && (pick_source(i) == ReplaySource::Wal || self.anchors.is_empty())
                {
                    wal_count += 1;
                }
                out.push(ep.clone());
            }
        }
        let frac = if out.is_empty() {
            0.0
        } else {
            wal_count as f32 / out.len() as f32
        };
        (out, frac)
    }

    /// Extract token sequences (`len >= 2`) from an 80/20 mixed batch for MeZO CE tuning.
    pub fn mixed_token_sequences(&self, batch_size: usize) -> Vec<Vec<u32>> {
        let (batch, _) = self.mixed_batch(batch_size);
        batch
            .into_iter()
            .map(|ep| ep.context)
            .filter(|ctx| ctx.len() >= 2)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_ratio_is_exact_20_percent() {
        let mut anchors = 0usize;
        for i in 0..100 {
            if pick_source(i) == ReplaySource::Anchor {
                anchors += 1;
            }
        }
        assert_eq!(anchors, 20);

        let mut buf = ReplayBuffer::new();
        for i in 0..40 {
            buf.wal_episodes.push(Episode {
                surprise: 0.5 + i as f32 * 0.01,
                key: vec![0.1],
                value: vec![0.2],
                context: vec![1, 2, 3],
            });
        }
        for _ in 0..10 {
            buf.add_anchor_tokens(vec![4, 5, 6]);
        }
        let (batch, frac) = buf.mixed_batch(50);
        assert_eq!(batch.len(), 50);
        assert!(
            (frac - 0.8).abs() < 1e-6,
            "expected 0.8 WAL fraction, got {frac}"
        );
    }
}
