//! Zero-dependency byte-fallback BPE.
//!
//! The base vocabulary is the 256 byte values, so every UTF-8 string round-trips
//! and there is no UNK. Optional merge rules append ids `256 + rank`. Applying
//! merges in rank order and expanding them on decode is lossless.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    /// Merge `i` consumes the pair `(left, right)` and emits id `256 + i`.
    merges: Vec<(u32, u32)>,
}

impl Tokenizer {
    pub fn byte_level() -> Self {
        Self { merges: Vec::new() }
    }

    pub fn with_merges(merges: Vec<(u32, u32)>) -> Self {
        Self { merges }
    }

    /// Whitespace pairs, one merge per line: `left right`. `#` comments ignored.
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
            let mut parts = line.split_whitespace();
            let left = parts
                .next()
                .ok_or_else(|| format!("line {}: missing left", lineno + 1))?;
            let right = parts
                .next()
                .ok_or_else(|| format!("line {}: missing right", lineno + 1))?;
            if parts.next().is_some() {
                return Err(format!("line {}: extra token", lineno + 1));
            }
            let left: u32 = left
                .parse()
                .map_err(|_| format!("line {}: bad left id", lineno + 1))?;
            let right: u32 = right
                .parse()
                .map_err(|_| format!("line {}: bad right id", lineno + 1))?;
            merges.push((left, right));
        }
        Ok(Self { merges })
    }

    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.merges.is_empty() {
            return text.bytes().map(|b| b as u32).collect();
        }
        // Newlines are never inside a trained merge (BPE is fit per line), so
        // they stay a single byte id and sequences encode independently.
        // Within a line, merges apply in rank order. A rank is skipped when
        // its pair is absent, which matches a full scan that changes nothing.
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

    fn expand(&self, id: u32, out: &mut Vec<u8>) {
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
    fn roundtrip_cjk_ascii_and_indent_never_unk() {
        let tok = Tokenizer::byte_level();
        assert_eq!(tok.vocab_size(), 256);
        for text in [
            "hello",
            "    fn main",
            "你好世界",
            "def\n    return",
            "a\u{1F600}b",
        ] {
            let ids = tok.encode(text);
            assert!(ids.iter().all(|id| *id < 256), "byte vocab emitted {ids:?}");
            assert_eq!(tok.decode_string(&ids), text);
        }
    }

    #[test]
    fn merge_roundtrip() {
        let tok = Tokenizer::with_merges(vec![(b'a' as u32, b'b' as u32)]);
        assert_eq!(tok.vocab_size(), 257);
        let ids = tok.encode("abXab");
        assert_eq!(ids, vec![256, b'X' as u32, 256]);
        assert_eq!(tok.decode_string(&ids), "abXab");
    }

    #[test]
    fn merge_file_roundtrips_utf8_line() {
        // 你 is UTF-8 e4 bd a0. Two merges assemble that character; newline stays a byte.
        let text = "228 189\n256 160\n";
        let tok = Tokenizer::from_merge_text(text).unwrap();
        assert_eq!(tok.vocab_size(), 258);
        let ids = tok.encode("你\n你");
        assert_eq!(ids, vec![257, u32::from(b'\n'), 257]);
        assert_eq!(tok.decode_string(&ids), "你\n你");
    }
}
