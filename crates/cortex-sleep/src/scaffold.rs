//! Scaffold a micro-expert from high-surprise logs. This does not train.

/// Design size of a shareable 1.58-bit adapter. Not a measured footprint.
pub const ADAPTER_BYTES_MIN: usize = 20 * 1024 * 1024;
pub const ADAPTER_BYTES_MAX: usize = 50 * 1024 * 1024;
pub const ADAPTER_BYTES_DEFAULT: usize = 24 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct SurpriseLog {
    pub domain: String,
    pub surprise: f32,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertScaffold {
    pub domain: String,
    pub target_bytes: usize,
    pub init_seed: u64,
    /// Always false. Filling the adapter is a later training job.
    pub trained: bool,
}

/// If one domain has at least `min_high` logs at or above `tau`, scaffold it.
/// Ties break on the lexicographic domain name so the result is deterministic.
pub fn scaffold_expert(logs: &[SurpriseLog], min_high: usize, tau: f32) -> Option<ExpertScaffold> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for log in logs {
        if log.surprise < tau || log.domain.is_empty() {
            continue;
        }
        if let Some(slot) = counts.iter_mut().find(|(d, _)| d == &log.domain) {
            slot.1 += 1;
        } else {
            counts.push((log.domain.clone(), 1));
        }
    }
    counts.retain(|(_, n)| *n >= min_high);
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let (domain, _) = counts.into_iter().next()?;
    Some(ExpertScaffold {
        init_seed: hash_domain(&domain),
        domain,
        target_bytes: ADAPTER_BYTES_DEFAULT,
        trained: false,
    })
}

fn hash_domain(domain: &str) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    for b in domain.as_bytes() {
        x ^= u64::from(*b);
        x = x.wrapping_mul(0x100_0000_01b3);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_is_untrained_and_sized() {
        let logs = vec![
            SurpriseLog {
                domain: "math".into(),
                surprise: 0.9,
                text: "a".into(),
            },
            SurpriseLog {
                domain: "math".into(),
                surprise: 0.2,
                text: "low".into(),
            },
            SurpriseLog {
                domain: "math".into(),
                surprise: 0.8,
                text: "b".into(),
            },
            SurpriseLog {
                domain: "code".into(),
                surprise: 0.99,
                text: "once".into(),
            },
        ];
        let got = scaffold_expert(&logs, 2, 0.5).unwrap();
        assert_eq!(got.domain, "math");
        assert!(!got.trained);
        assert!(got.target_bytes >= ADAPTER_BYTES_MIN && got.target_bytes <= ADAPTER_BYTES_MAX);
        assert!(scaffold_expert(&logs, 5, 0.5).is_none());
    }
}
