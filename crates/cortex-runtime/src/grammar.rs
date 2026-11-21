//! Grammar mask after `<|tool_call|>`.
//!
//! The FSM only allows bytes that keep the tool-call body inside a JSON object
//! and then the closing tag. 100% valid calls is a property of this mask, not
//! a claim about an untrained language model.

const CLOSE: &[u8] = b"</|tool_call|>";

#[derive(Clone, Debug)]
pub struct ToolCallFsm {
    stack: Vec<Frame>,
    phase: Phase,
    close_at: usize,
    done: bool,
}

#[derive(Clone, Copy, Debug)]
enum Frame {
    Object { expect_key: bool },
    Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Value,
    ObjectKey,
    Colon,
    AfterValue,
    String { escape: bool, is_key: bool },
    Number,
    Literal { rest: &'static [u8] },
    Close,
}

impl ToolCallFsm {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            phase: Phase::Value,
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
            Phase::Close => self.close_at < CLOSE.len() && byte == CLOSE[self.close_at],
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
                    || self.after_value_ok(byte)
            }
            Phase::Literal { rest } => !rest.is_empty() && byte == rest[0],
            Phase::Colon => byte == b':',
            Phase::ObjectKey => byte == b'"' || (byte == b'}' && self.can_close_object()),
            Phase::AfterValue => self.after_value_ok(byte),
            Phase::Value => self.value_ok(byte),
        }
    }

    pub fn push(&mut self, byte: u8) -> Result<(), ()> {
        if !self.allows(byte) {
            return Err(());
        }
        match self.phase {
            Phase::Close => {
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
                    // stay in the number
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
            Phase::Colon => self.phase = Phase::Value,
            Phase::ObjectKey => {
                if byte == b'}' {
                    self.stack.pop();
                    self.finish_value();
                } else {
                    self.phase = Phase::String {
                        escape: false,
                        is_key: true,
                    };
                }
            }
            Phase::AfterValue => {
                if byte == b',' {
                    match self.stack.last() {
                        Some(Frame::Object { .. }) => self.phase = Phase::ObjectKey,
                        Some(Frame::Array) => self.phase = Phase::Value,
                        None => return Err(()),
                    }
                } else if byte == b'}' || byte == b']' {
                    self.stack.pop();
                    self.finish_value();
                }
            }
            Phase::Value => self.start_value(byte),
        }
        Ok(())
    }

    fn can_close_object(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Object { expect_key: true }))
    }

    fn after_value_ok(&self, byte: u8) -> bool {
        match self.stack.last() {
            Some(Frame::Object { .. }) => byte == b',' || byte == b'}',
            Some(Frame::Array) => byte == b',' || byte == b']',
            None => byte == CLOSE[0],
        }
    }

    fn value_ok(&self, byte: u8) -> bool {
        if byte == b']' && matches!(self.stack.last(), Some(Frame::Array)) {
            return true;
        }
        matches!(byte, b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-') || byte.is_ascii_digit()
    }

    fn start_value(&mut self, byte: u8) {
        match byte {
            b'{' => {
                self.stack.push(Frame::Object { expect_key: true });
                self.phase = Phase::ObjectKey;
            }
            b'[' => {
                self.stack.push(Frame::Array);
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

/// Greedy decode under the mask. `score(step, byte)` is the dummy model.
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
    fn constrained_decode_is_parseable_json() {
        let target = br#"{"name":"add","arguments":{"a":1,"b":2}}</|tool_call|>"#;
        let score = |step: usize, byte: u8| -> f32 {
            // 0x01 is illegal in every state. Without the mask it would win.
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
