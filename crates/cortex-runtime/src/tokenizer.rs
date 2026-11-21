//! Zero-dependency UTF-8 byte-fallback BPE tokenizer with dense atomic control tokens.
//!
//! * Base vocabulary (`0..256`): the 256 raw UTF-8 byte values, guaranteeing
//!   100% lossless round-trip on arbitrary rare CJK, mathematical symbols, and
//!   emoji with **zero `<UNK>` crashes**.
//! * Merge rules (`256 .. 256 + merges.len()`): applied greedily in rank order.
//! * Atomic control tokens (`256 + merges.len() .. 256 + merges.len() + 10`):
//!   dense IDs immediately following the BPE vocabulary (`<|im_start|>`,
//!   `<|im_end|>`, `<|thought|>`, `</|thought|>`, `<|tools|>`, `</|tools|>`,
//!   `<|tool_call|>`, `</|tool_call|>`, `<|tool_response|>`, `</|tool_response|>`),
//!   avoiding modulo ID collisions in the tied INT8 embedding / LM head.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

pub const SPECIALS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|thought|>",
    "</|thought|>",
    "<|tools|>",
    "</|tools|>",
    "<|tool_call|>",
    "</|tool_call|>",
    "<|tool_response|>",
    "</|tool_response|>",
];

pub const IDX_IM_START: usize = 0;
pub const IDX_IM_END: usize = 1;
pub const IDX_THOUGHT_OPEN: usize = 2;
pub const IDX_THOUGHT_CLOSE: usize = 3;
pub const IDX_TOOLS_OPEN: usize = 4;
pub const IDX_TOOLS_CLOSE: usize = 5;
pub const IDX_TOOL_CALL_OPEN: usize = 6;
pub const IDX_TOOL_CALL_CLOSE: usize = 7;
pub const IDX_TOOL_RESP_OPEN: usize = 8;
pub const IDX_TOOL_RESP_CLOSE: usize = 9;

pub const TOOL_CALL_OPEN: &str = "<|tool_call|>";
pub const TOOL_CALL_CLOSE: &str = "</|tool_call|>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
}

pub fn render_chat(turns: &[ChatTurn]) -> String {
    let mut out = String::new();
    for turn in turns {
        out.push_str("<|im_start|>");
        out.push_str(turn.role.as_str());
        out.push('\n');
        out.push_str(&turn.content);
        out.push_str("<|im_end|>\n");
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    /// Merge `i` consumes `(left, right)` and emits `256 + i`.
    merges: Vec<(u32, u32)>,
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        Self { merges: Vec::new() }
    }

    pub fn with_merges(merges: Vec<(u32, u32)>) -> Self {
        Self { merges }
    }

    /// Parse merge rules (`left right` or `left right advisory_merged_id` per line).
    pub fn from_merge_file(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_merge_text(&text)
    }

    pub fn from_merge_text(text: &str) -> Result<Self, String> {
        let mut merges = Vec::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 && parts.len() != 3 {
                return Err(format!(
                    "line {}: expected 2 or 3 integer columns, got {}",
                    lineno + 1,
                    parts.len()
                ));
            }
            let left: u32 = parts[0]
                .parse()
                .map_err(|_| format!("line {}: bad left id", lineno + 1))?;
            let right: u32 = parts[1]
                .parse()
                .map_err(|_| format!("line {}: bad right id", lineno + 1))?;
            if parts.len() == 3 {
                let _: u32 = parts[2]
                    .parse()
                    .map_err(|_| format!("line {}: bad merged id", lineno + 1))?;
            }
            merges.push((left, right));
        }
        Ok(Self { merges })
    }

    /// Base byte + BPE vocabulary size (`256 + merges.len()`).
    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    /// Total dense vocabulary size including all 10 atomic control tokens.
    pub fn vocab_with_specials(&self) -> usize {
        256 + self.merges.len() + SPECIALS.len()
    }

    #[inline]
    pub fn special_base(&self) -> u32 {
        (256 + self.merges.len()) as u32
    }

    #[inline]
    pub fn special_id(&self, index: usize) -> u32 {
        self.special_base() + index as u32
    }

    #[inline]
    pub fn is_special_id(&self, id: u32) -> bool {
        let base = self.special_base();
        id >= base && id < base + SPECIALS.len() as u32
    }

    #[inline]
    pub fn im_end_id(&self) -> u32 {
        self.special_id(IDX_IM_END)
    }

    #[inline]
    pub fn tool_call_open_id(&self) -> u32 {
        self.special_id(IDX_TOOL_CALL_OPEN)
    }

    #[inline]
    pub fn tool_call_close_id(&self) -> u32 {
        self.special_id(IDX_TOOL_CALL_CLOSE)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in self.split_specials(text) {
            match piece {
                Piece::Special(id) => out.push(id),
                Piece::Text(chunk) => out.extend(self.encode_plain(chunk)),
            }
        }
        out
    }

    fn encode_plain(&self, text: &str) -> Vec<u32> {
        if self.merges.is_empty() {
            return text.bytes().map(|b| b as u32).collect();
        }
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut start = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\n' {
                if start < i {
                    out.extend(encode_segment(&bytes[start..i], &self.merges));
                }
                out.push(u32::from(b'\n'));
                start = i + 1;
            }
        }
        if start < bytes.len() {
            out.extend(encode_segment(&bytes[start..], &self.merges));
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            self.expand(id, &mut out);
        }
        out
    }

    pub fn decode_string(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode(ids)).into_owned()
    }

    /// Raw bytes represented by a single token `id` (used by Grammar FSM masking).
    pub fn token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        self.expand(id, out);
    }

    fn expand(&self, id: u32, out: &mut Vec<u8>) {
        let base = self.special_base();
        if id >= base {
            let index = (id - base) as usize;
            if let Some(token) = SPECIALS.get(index) {
                out.extend_from_slice(token.as_bytes());
            }
            return;
        }
        if id < 256 {
            out.push(id as u8);
            return;
        }
        let rank = (id as usize) - 256;
        if let Some(&(left, right)) = self.merges.get(rank) {
            self.expand(left, out);
            self.expand(right, out);
        }
    }

    fn split_specials<'a>(&self, text: &'a str) -> Vec<Piece<'a>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        let mut start = 0usize;
        while i < bytes.len() {
            if let Some((len, id)) = self.match_special(&bytes[i..]) {
                if start < i {
                    out.push(Piece::Text(&text[start..i]));
                }
                out.push(Piece::Special(id));
                i += len;
                start = i;
            } else {
                i += 1;
            }
        }
        if start < bytes.len() {
            out.push(Piece::Text(&text[start..]));
        }
        out
    }

    fn match_special(&self, bytes: &[u8]) -> Option<(usize, u32)> {
        let mut best: Option<(usize, u32)> = None;
        for (index, token) in SPECIALS.iter().enumerate() {
            let raw = token.as_bytes();
            if bytes.starts_with(raw) && best.map(|(len, _)| raw.len() > len).unwrap_or(true) {
                best = Some((raw.len(), self.special_id(index)));
            }
        }
        best
    }
}

enum Piece<'a> {
    Text(&'a str),
    Special(u32),
}

fn encode_segment(bytes: &[u8], merges: &[(u32, u32)]) -> Vec<u32> {
    let mut ids: Vec<u32> = bytes.iter().copied().map(u32::from).collect();
    if ids.len() < 2 || merges.is_empty() {
        return ids;
    }
    let mut present = pair_set(&ids);
    for (rank, &(left, right)) in merges.iter().enumerate() {
        if !present.contains(&(left, right)) {
            continue;
        }
        let new_id = 256 + rank as u32;
        loop {
            let mut changed = false;
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
                    out.push(new_id);
                    i += 2;
                    changed = true;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
            if !changed {
                break;
            }
        }
        present = pair_set(&ids);
    }
    ids
}

fn pair_set(ids: &[u32]) -> HashSet<(u32, u32)> {
    let mut set = HashSet::new();
    for w in ids.windows(2) {
        set.insert((w[0], w[1]));
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_rare_cjk_gothic_and_emoji_never_unk() {
        let tok = Tokenizer::byte_level();
        assert_eq!(tok.vocab_size(), 256);
        assert_eq!(tok.vocab_with_specials(), 266);
        for text in [
            "hello",
            "    fn main",
            "你好世界 — 龘靐齉齾爩",
            "𐍈 Gothic & 🦀 Rust",
            "def\n    return",
        ] {
            let ids = tok.encode(text);
            assert!(ids.iter().all(|id| *id < 256), "byte vocab emitted {ids:?}");
            assert_eq!(tok.decode_string(&ids), text);
        }
    }

    #[test]
    fn merge_roundtrip_two_and_three_col_formats() {
        let tok2 = Tokenizer::from_merge_text("228 189\n256 160\n").unwrap();
        let tok3 = Tokenizer::from_merge_text("# comment\n228 189 256\n256 160 257\n").unwrap();
        assert_eq!(tok2.vocab_size(), 258);
        assert_eq!(tok3.vocab_size(), 258);
        assert_eq!(tok2.encode("你\n你"), vec![257, u32::from(b'\n'), 257]);
        assert_eq!(tok3.encode("你\n你"), vec![257, u32::from(b'\n'), 257]);
        assert_eq!(tok3.decode_string(&[257, u32::from(b'\n'), 257]), "你\n你");
    }

    #[test]
    fn dense_specials_are_atomic_and_collision_free() {
        let tok = Tokenizer::byte_level();
        let text = "a<|tool_call|>{\"name\":\"add\"}</|tool_call|>b";
        let ids = tok.encode(text);
        let specials: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| tok.is_special_id(id))
            .collect();
        assert_eq!(
            specials,
            vec![tok.tool_call_open_id(), tok.tool_call_close_id()]
        );
        assert!(ids
            .iter()
            .all(|&id| (id as usize) < tok.vocab_with_specials()));
        assert_eq!(tok.decode_string(&ids), text);

        let rendered = render_chat(&[
            ChatTurn {
                role: Role::User,
                content: "hi".into(),
            },
            ChatTurn {
                role: Role::Assistant,
                content: "<|thought|>think</|thought|>ok".into(),
            },
        ]);
        let encoded = tok.encode(&rendered);
        assert!(encoded
            .iter()
            .all(|&id| (id as usize) < tok.vocab_with_specials()));
        assert_eq!(tok.decode_string(&encoded), rendered);
    }
}
