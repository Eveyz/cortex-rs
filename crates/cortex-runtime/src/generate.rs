//! Streaming token generation with online hippocampus plasticity, `<|tool_call|>`
//! JSON Grammar FSM masking, stop-token truncation, and cross-platform RSS telemetry.

use std::time::Instant;

use crate::grammar::ToolCallFsm;
use crate::model::{CortexModel, ModelState};
use crate::sample::sample_token;
use crate::tokenizer::{Tokenizer, TOOL_CALL_OPEN};

pub struct Session<'a> {
    pub model: &'a CortexModel,
    pub state: ModelState,
    pub history: Vec<u32>,
    pub prompt_len: usize,
    pub tokenizer: Tokenizer,
    pub logits: Vec<f32>,
    pub remaining: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub rng: u64,
    pub online_plasticity: bool,
    pub constrain_tool_json: bool,
    pub stop_on_control: bool,
    active_tool_fsm: Option<ToolCallFsm>,
    halted: bool,
}

impl<'a> Session<'a> {
    pub fn new(
        model: &'a CortexModel,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
    ) -> Self {
        Self::with_options(
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            seed,
            Tokenizer::byte_level(),
            true,
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_tokenizer(
        model: &'a CortexModel,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        tokenizer: Tokenizer,
    ) -> Self {
        Self::with_options(
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            seed,
            tokenizer,
            true,
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        model: &'a CortexModel,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        tokenizer: Tokenizer,
        online_plasticity: bool,
        constrain_tool_json: bool,
        stop_on_control: bool,
    ) -> Self {
        let ids = tokenizer.encode(prompt);
        let prompt_len = ids.len();
        let mut state = model.new_state();
        let mut logits = vec![0f32; model.cfg.vocab];
        if ids.is_empty() {
            logits = if online_plasticity {
                model.step_plastic(0, false, &[], &mut state)
            } else {
                model.step(0, &mut state)
            };
        } else {
            for (i, &id) in ids.iter().enumerate() {
                let is_last = i + 1 == ids.len();
                logits = if online_plasticity {
                    model.step_plastic(id, is_last, &ids[..=i], &mut state)
                } else {
                    model.step(id, &mut state)
                };
            }
        }
        let active_tool_fsm = if constrain_tool_json && prompt.ends_with(TOOL_CALL_OPEN) {
            Some(ToolCallFsm::new())
        } else {
            None
        };
        Self {
            model,
            state,
            history: ids,
            prompt_len,
            tokenizer,
            logits,
            remaining: max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            rng: seed.max(1),
            online_plasticity,
            constrain_tool_json,
            stop_on_control,
            active_tool_fsm,
            halted: false,
        }
    }

    pub fn continuation_text(&self) -> String {
        let start = self.prompt_len.min(self.history.len());
        self.tokenizer.decode_string(&self.history[start..])
    }
}

pub struct TokenTick {
    pub index: usize,
    pub id: u32,
    pub dt_us: u128,
    pub rss_kb: u64,
}

/// Generate one streaming token. Applies `ToolCallFsm` JSON masking when inside
/// `<|tool_call|>` and halts immediately on `<|im_end|>` or `</|tool_call|>`.
pub fn next_token(session: &mut Session) -> Option<TokenTick> {
    if session.remaining == 0 || session.halted {
        return None;
    }
    let index = session.history.len();
    let t0 = Instant::now();

    let mut masked_logits = session.logits.clone();
    if let Some(fsm) = &session.active_tool_fsm {
        fsm.mask_logits(&mut masked_logits, &session.tokenizer);
    }

    let id = sample_token(
        &masked_logits,
        session.temperature,
        session.top_p,
        session.repetition_penalty,
        &session.history,
        &mut session.rng,
    ) as u32;
    let id = id.min(session.model.cfg.vocab.saturating_sub(1) as u32);

    // Advance constrained JSON FSM if active, or enter FSM on `<|tool_call|>`.
    let mut completed_tool_call = false;
    if let Some(mut fsm) = session.active_tool_fsm.take() {
        let mut bytes = Vec::with_capacity(16);
        session.tokenizer.token_bytes(id, &mut bytes);
        let _ = fsm.push_bytes(&bytes);
        if fsm.done() {
            completed_tool_call = true;
        } else {
            session.active_tool_fsm = Some(fsm);
        }
    } else if session.constrain_tool_json && id == session.tokenizer.tool_call_open_id() {
        session.active_tool_fsm = Some(ToolCallFsm::new());
    }

    session.history.push(id);
    session.remaining -= 1;

    let hit_stop =
        session.stop_on_control && (id == session.tokenizer.im_end_id() || completed_tool_call);
    if hit_stop {
        session.halted = true;
    }

    let is_last = session.remaining == 0 || session.halted;
    session.logits = if session.online_plasticity {
        session
            .model
            .step_plastic(id, is_last, &session.history, &mut session.state)
    } else {
        session.model.step(id, &mut session.state)
    };

    let dt_us = t0.elapsed().as_micros();
    Some(TokenTick {
        index,
        id,
        dt_us,
        rss_kb: rss_kb(),
    })
}

pub fn stream_tokens(session: &mut Session) -> Vec<u32> {
    let mut out = Vec::new();
    while let Some(tick) = next_token(session) {
        out.push(tick.id);
    }
    out
}

/// Process resident / peak resident memory in KiB on Linux, macOS, and BSD.
pub fn rss_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/self/status") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(n) = rest.split_whitespace().next() {
                        if let Ok(kb) = n.parse::<u64>() {
                            if kb > 0 {
                                return kb;
                            }
                        }
                    }
                }
            }
        }
    }
    #[cfg(unix)]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            #[cfg(target_os = "macos")]
            {
                return (usage.ru_maxrss as u64) / 1024;
            }
            #[cfg(not(target_os = "macos"))]
            {
                return usage.ru_maxrss as u64;
            }
        }
    }
    0
}

pub fn format_tick(tick: &TokenTick) -> String {
    format!(
        "token[{}] id={} dt_us={} rss_kb={}",
        tick.index, tick.id, tick.dt_us, tick.rss_kb
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CortexConfig;

    #[test]
    fn session_grammar_fsm_enforces_valid_json_after_tool_call_open() {
        let model = CortexModel::from_seed(CortexConfig::micro(), 3);
        let tok = Tokenizer::byte_level();
        // Start prompt ending with `<|tool_call|>` so FSM activates immediately!
        let mut session = Session::with_tokenizer(
            &model,
            "Call tool: <|tool_call|>",
            64,
            0.0,
            1.0,
            1.0,
            1,
            tok,
        );
        let first = next_token(&mut session).unwrap();
        // RootOpen only allows JSON whitespace or '{' (123).
        let ch = first.id as u8;
        assert!(
            matches!(ch, b'{' | b' ' | b'\t' | b'\n' | b'\r'),
            "Grammar FSM must force '{{' or whitespace at tool_call root, got byte {ch}"
        );
        assert!(
            first.rss_kb > 0,
            "rss_kb must report real memory on macOS/Linux"
        );
    }
}
