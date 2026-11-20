//! 80% WAL episodes, 20% factory anchors.
//!
//! Index `i` is an anchor when `i % 5 == 4`, which is exactly one in five.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplaySource {
    Wal,
    Anchor,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_ratio_is_20_percent() {
        let mut anchors = 0usize;
        let n = 100usize;
        for i in 0..n {
            if pick_source(i) == ReplaySource::Anchor {
                anchors += 1;
            }
        }
        assert_eq!(anchors, 20);
        let wal = ["w0", "w1"];
        let anc = ["a0"];
        let mut a = 0;
        for i in 0..n {
            let item = take(&wal, &anc, i).unwrap();
            if *item == "a0" && pick_source(i) == ReplaySource::Anchor {
                a += 1;
            }
        }
        assert_eq!(a, 20);
    }
}
