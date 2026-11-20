//! Zero-dependency byte-fallback BPE.
//!
//! The base vocabulary is the 256 byte values, so every UTF-8 string round-trips
//! and there is no UNK. Optional merge rules append ids `256 + rank`. Applying
//! merges in rank order and expanding them on decode is lossless.

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

    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        for (rank, &(left, right)) in self.merges.iter().enumerate() {
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
        }
        ids
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
}
