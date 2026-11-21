//! Constrained JSON + `</|tool_call|>` Grammar FSM for structured tool decoding.
//!
//! Activated immediately after the model emits `<|tool_call|>`. Masks logits at
//! every sampling step (`mask_logits`) so only syntactically valid JSON object
//! continuations (with optional standard JSON whitespace) followed by the
//! closing `</|tool_call|>` tag can be sampled.

use crate::tokenizer::Tokenizer;

const CLOSE: &[u8] = b"</|tool_call|>";

#[inline]
fn is_json_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[derive(Clone, Debug)]
pub struct ToolCallFsm {
    stack: Vec<Frame>,
    phase: Phase,
    close_at: usize,
    done: bool,
}

#[derive(Clone, Copy, Debug)]
enum Frame {
    Object { empty: bool },
    Array { empty: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Root must begin with `{` (a tool call is always a JSON object).
    RootOpen,
    Value,
    ObjectKey {
        allow_close: bool,
    },
    Colon,
    AfterValue,
    String {
        escape: bool,
        is_key: bool,
    },
    Number,
    Literal {
        rest: &'static [u8],
    },
    Close,
}

impl Default for ToolCallFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallFsm {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            phase: Phase::RootOpen,
            close_at: 0,
            done: false,
        }
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn allows(&self, byte: u8) -> bool {
        if self.done {
            return false;
        }
        match self.phase {
            Phase::RootOpen => is_json_ws(byte) || byte == b'{',
            Phase::Close => {
                (self.close_at == 0 && is_json_ws(byte))
                    || (self.close_at < CLOSE.len() && byte == CLOSE[self.close_at])
            }
            Phase::String { escape, .. } => {
                if escape {
                    matches!(
                        byte,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                    )
                } else {
                    byte >= 0x20 && byte != 0x7f
                }
            }
            Phase::Number => {
                byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-')
                    || is_json_ws(byte)
                    || self.after_value_ok(byte)
            }
            Phase::Literal { rest } => !rest.is_empty() && byte == rest[0],
            Phase::Colon => is_json_ws(byte) || byte == b':',
            Phase::ObjectKey { allow_close } => {
                is_json_ws(byte) || byte == b'"' || (allow_close && byte == b'}')
            }
            Phase::AfterValue => is_json_ws(byte) || self.after_value_ok(byte),
            Phase::Value => is_json_ws(byte) || self.value_ok(byte),
        }
    }

    #[allow(clippy::result_unit_err)]
    pub fn push(&mut self, byte: u8) -> Result<(), ()> {
        if !self.allows(byte) {
            return Err(());
        }
        match self.phase {
            Phase::RootOpen => {
                if is_json_ws(byte) {
                    return Ok(());
                }
                self.stack.push(Frame::Object { empty: true });
                self.phase = Phase::ObjectKey { allow_close: true };
            }
            Phase::Close => {
                if self.close_at == 0 && is_json_ws(byte) {
                    return Ok(());
                }
                self.close_at += 1;
                if self.close_at == CLOSE.len() {
                    self.done = true;
                }
            }
            Phase::String { escape, is_key } => {
                if escape {
                    self.phase = Phase::String {
                        escape: false,
                        is_key,
                    };
                } else if byte == b'\\' {
                    self.phase = Phase::String {
                        escape: true,
                        is_key,
                    };
                } else if byte == b'"' {
                    if is_key {
                        self.phase = Phase::Colon;
                    } else {
                        self.finish_value();
                    }
                }
            }
            Phase::Number => {
                if byte.is_ascii_digit() || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-') {
                    // continue scanning number
                } else {
                    self.finish_value();
                    return self.push(byte);
                }
            }
            Phase::Literal { rest } => {
                let next = &rest[1..];
                if next.is_empty() {
                    self.finish_value();
                } else {
                    self.phase = Phase::Literal { rest: next };
                }
            }
            Phase::Colon => {
                if !is_json_ws(byte) {
                    self.phase = Phase::Value;
                }
            }
            Phase::ObjectKey { .. } => {
                if is_json_ws(byte) {
                    return Ok(());
                }
                if byte == b'}' {
                    self.stack.pop();
                    self.finish_value();
                } else {
                    if let Some(Frame::Object { empty }) = self.stack.last_mut() {
                        *empty = false;
                    }
                    self.phase = Phase::String {
                        escape: false,
                        is_key: true,
                    };
                }
            }
            Phase::AfterValue => {
                if is_json_ws(byte) {
                    return Ok(());
                }
                if byte == b',' {
                    match self.stack.last() {
                        Some(Frame::Object { .. }) => {
                            self.phase = Phase::ObjectKey { allow_close: false };
                        }
                        Some(Frame::Array { .. }) => self.phase = Phase::Value,
                        None => return Err(()),
                    }
                } else if byte == b'}' || byte == b']' {
                    self.stack.pop();
                    self.finish_value();
                }
            }
            Phase::Value => {
                if is_json_ws(byte) {
                    return Ok(());
                }
                self.start_value(byte);
            }
        }
        Ok(())
    }

    /// Push a slice of bytes (e.g. a decoded BPE or special token).
    #[allow(clippy::result_unit_err)]
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), ()> {
        if bytes.is_empty() {
            return Err(());
        }
        for &b in bytes {
            self.push(b)?;
        }
        Ok(())
    }

    /// Mask `logits` in place so only tokens permitted by the current FSM state
    /// remain finite; illegal tokens are set to `f32::NEG_INFINITY`.
    pub fn mask_logits(&self, logits: &mut [f32], tok: &Tokenizer) {
        let mut scratch = Vec::with_capacity(16);
        let mut any_allowed = false;
        for (id, score) in logits.iter_mut().enumerate() {
            let tid = id as u32;
            // Disallow non-closing special tokens inside a tool call JSON body.
            if tok.is_special_id(tid) && tid != tok.tool_call_close_id() {
                *score = f32::NEG_INFINITY;
                continue;
            }
            scratch.clear();
            tok.token_bytes(tid, &mut scratch);
            let mut probe = self.clone();
            if probe.push_bytes(&scratch).is_ok() {
                any_allowed = true;
            } else {
                *score = f32::NEG_INFINITY;
            }
        }
        if !any_allowed {
            // Safety fallback: never leave all logits -inf.
            let close_id = tok.tool_call_close_id() as usize;
            if close_id < logits.len() {
                logits[close_id] = 0.0;
            }
        }
    }

    fn after_value_ok(&self, byte: u8) -> bool {
        match self.stack.last() {
            Some(Frame::Object { .. }) => byte == b',' || byte == b'}',
            Some(Frame::Array { .. }) => byte == b',' || byte == b']',
            None => byte == CLOSE[0],
        }
    }

    fn value_ok(&self, byte: u8) -> bool {
        if byte == b']' && matches!(self.stack.last(), Some(Frame::Array { empty: true })) {
            return true;
        }
        matches!(byte, b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-') || byte.is_ascii_digit()
    }

    fn start_value(&mut self, byte: u8) {
        if let Some(Frame::Array { empty }) = self.stack.last_mut() {
            if byte != b']' {
                *empty = false;
            }
        }
        match byte {
            b'{' => {
                self.stack.push(Frame::Object { empty: true });
                self.phase = Phase::ObjectKey { allow_close: true };
            }
            b'[' => {
                self.stack.push(Frame::Array { empty: true });
                self.phase = Phase::Value;
            }
            b'"' => {
                self.phase = Phase::String {
                    escape: false,
                    is_key: false,
                }
            }
            b't' => self.phase = Phase::Literal { rest: b"rue" },
            b'f' => self.phase = Phase::Literal { rest: b"alse" },
            b'n' => self.phase = Phase::Literal { rest: b"ull" },
            b'-' => self.phase = Phase::Number,
            b if b.is_ascii_digit() => self.phase = Phase::Number,
            b']' => {
                self.stack.pop();
                self.finish_value();
            }
            _ => {}
        }
    }

    fn finish_value(&mut self) {
        if self.stack.is_empty() {
            self.phase = Phase::Close;
            self.close_at = 0;
        } else {
            self.phase = Phase::AfterValue;
        }
    }
}

/// Greedy decode under the byte mask.
pub fn constrained_tool_json(score: impl Fn(usize, u8) -> f32, max_len: usize) -> Vec<u8> {
    let mut fsm = ToolCallFsm::new();
    let mut out = Vec::new();
    for step in 0..max_len {
        if fsm.done() {
            break;
        }
        let mut best: Option<u8> = None;
        let mut best_s = f32::NEG_INFINITY;
        for byte in 0u8..=255 {
            if fsm.allows(byte) {
                let s = score(step, byte);
                if s > best_s {
                    best_s = s;
                    best = Some(byte);
                }
            }
        }
        let Some(byte) = best else { break };
        if fsm.push(byte).is_err() {
            break;
        }
        out.push(byte);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_decode_allows_whitespace_and_rejects_invalid_bytes() {
        let target = br#"{ "name": "add", "arguments": { "a": 1, "b": 2 } }</|tool_call|>"#;
        let score = |step: usize, byte: u8| -> f32 {
            if byte == 0x01 {
                return 1000.0;
            }
            if step < target.len() && byte == target[step] {
                return 10.0;
            }
            0.0
        };
        let out = constrained_tool_json(score, target.len() + 8);
        let text = String::from_utf8(out.clone()).unwrap();
        assert_eq!(text.as_bytes(), target, "{text}");
        let json = text.trim_end_matches("</|tool_call|>");
        let parsed = crate::harness::parse_tool_call(&format!("<|tool_call|>{json}</|tool_call|>"))
            .expect(&text);
        assert_eq!(parsed.name, "add");
        assert_eq!(parsed.arguments.get("a").map(String::as_str), Some("1"));
        assert_eq!(parsed.arguments.get("b").map(String::as_str), Some("2"));
        assert!(!out.contains(&0x01));
    }
}
