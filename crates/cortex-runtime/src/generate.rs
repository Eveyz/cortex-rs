//! Token streaming used by the CLI and the HTTP endpoint.

use std::time::Instant;

use crate::model::CortexModel;
use crate::sample::sample_token;
use crate::tokenizer::Tokenizer;

pub struct Session<'a> {
    pub model: &'a mut CortexModel,
    pub state: crate::model::ModelState,
    pub history: Vec<u32>,
    pub logits: Vec<f32>,
    pub remaining: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub rng: u64,
}

impl<'a> Session<'a> {
    pub fn new(
        model: &'a mut CortexModel,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
    ) -> Self {
        let tok = Tokenizer::byte_level();
        let ids = tok.encode(prompt);
        let mut state = model.new_state();
        let mut logits = vec![0f32; model.cfg.vocab];
        if ids.is_empty() {
            logits = model.step(0, &mut state);
        } else {
            for id in &ids {
                logits = model.step(*id, &mut state);
            }
        }
        Self {
            model,
            state,
            history: ids,
            logits,
            remaining: max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            rng: seed.max(1),
        }
    }
}

pub struct TokenTick {
    pub index: usize,
    pub id: u32,
    pub dt_us: u128,
    pub rss_kb: u64,
}

/// Generate up to `remaining` tokens. Each call of the iterator is one step.
pub fn next_token(session: &mut Session) -> Option<TokenTick> {
    if session.remaining == 0 {
        return None;
    }
    let index = session.history.len();
    let t0 = Instant::now();
    let id = sample_token(
        &session.logits,
        session.temperature,
        session.top_p,
        session.repetition_penalty,
        &session.history,
        &mut session.rng,
    ) as u32;
    // Keep the id inside the embedding table. Byte BPE ids above vocab are
    // wrapped by `embed`, but we also clamp here so the printed id is legal.
    let id = id % session.model.cfg.vocab.max(1) as u32;
    session.logits = session.model.step(id, &mut session.state);
    session.history.push(id);
    session.remaining -= 1;
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

pub fn rss_kb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(n) = rest.split_whitespace().next() {
                return n.parse().unwrap_or(0);
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
