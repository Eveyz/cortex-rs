//! L3 verified-trace extraction. Failed detours are dropped. Nothing is trained.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceStep {
    pub text: String,
    pub ok: bool,
    pub detour: bool,
}

#[derive(Clone, Debug)]
pub struct VerifiedEpisode {
    pub problem: String,
    pub verified: bool,
    pub steps: Vec<TraceStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenPair {
    pub problem: String,
    pub chain: Vec<String>,
}

/// Keep only the successful, non-detour steps. Unverified episodes yield nothing.
pub fn extract_golden(episode: &VerifiedEpisode) -> Option<GoldenPair> {
    if !episode.verified || episode.problem.trim().is_empty() {
        return None;
    }
    let chain: Vec<String> = episode
        .steps
        .iter()
        .filter(|step| step.ok && !step.detour && !step.text.trim().is_empty())
        .map(|step| step.text.trim().to_string())
        .collect();
    if chain.is_empty() {
        None
    } else {
        Some(GoldenPair {
            problem: episode.problem.clone(),
            chain,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_failed_detours() {
        let episode = VerifiedEpisode {
            problem: "add two numbers".into(),
            verified: true,
            steps: vec![
                TraceStep {
                    text: "try the wrong tool".into(),
                    ok: false,
                    detour: true,
                },
                TraceStep {
                    text: "call add".into(),
                    ok: true,
                    detour: false,
                },
                TraceStep {
                    text: "report 5".into(),
                    ok: true,
                    detour: false,
                },
            ],
        };
        let gold = extract_golden(&episode).unwrap();
        assert_eq!(gold.chain, vec!["call add".to_string(), "report 5".into()]);
        let mut failed = episode;
        failed.verified = false;
        assert!(extract_golden(&failed).is_none());
    }
}
