//! Temperature, nucleus (top-p), and repetition penalty.
//!
//! `temperature <= 0` is greedy argmax after the penalty. Otherwise logits are
//! divided by temperature, restricted to the top-p nucleus, and sampled with
//! an xorshift64* generator so a seed reproduces a stream.

pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-8 {
        return;
    }
    let p = if penalty <= 0.0 { 1.0 } else { penalty };
    for &id in history {
        let i = id as usize;
        if i >= logits.len() {
            continue;
        }
        if logits[i] > 0.0 {
            logits[i] /= p;
        } else {
            logits[i] *= p;
        }
    }
}

pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    history: &[u32],
    rng: &mut u64,
) -> usize {
    let mut scores = logits.to_vec();
    apply_repetition_penalty(&mut scores, history, repetition_penalty);
    if temperature <= 1e-8 {
        return argmax(&scores);
    }
    let inv_t = 1.0 / temperature;
    for s in scores.iter_mut() {
        *s *= inv_t;
    }
    let probs = softmax(&scores);
    let keep = nucleus_mask(&probs, top_p.clamp(0.0, 1.0));
    let mut mass = 0f32;
    for (i, p) in probs.iter().enumerate() {
        if keep[i] {
            mass += *p;
        }
    }
    if mass <= 0.0 {
        return argmax(&scores);
    }
    let draw = xorshift(rng) * mass;
    let mut acc = 0f32;
    for (i, p) in probs.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        acc += *p;
        if acc >= draw {
            return i;
        }
    }
    keep.iter().rposition(|k| *k).unwrap_or(0)
}

pub fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in xs.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    best_i
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = vec![0f32; logits.len()];
    let mut sum = 0f32;
    for (i, l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        exps[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

fn nucleus_mask(probs: &[f32], top_p: f32) -> Vec<bool> {
    if top_p >= 1.0 - 1e-8 {
        return vec![true; probs.len()];
    }
    if top_p <= 0.0 {
        let mut mask = vec![false; probs.len()];
        let i = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        mask[i] = true;
        return mask;
    }
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut mask = vec![false; probs.len()];
    let mut cum = 0f32;
    for idx in order {
        mask[idx] = true;
        cum += probs[idx];
        if cum >= top_p {
            break;
        }
    }
    mask
}

fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 40) as f32 / ((1u64 << 24) as f32)
}

pub fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    if logits.is_empty() || target >= logits.len() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for l in logits {
        sum += (l - max).exp();
    }
    (sum.ln() + max) - logits[target]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_penalty_and_top_p() {
        let logits = [0.1, 2.0, 0.5];
        assert_eq!(argmax(&logits), 1);
        let mut penalized = logits;
        apply_repetition_penalty(&mut penalized, &[1], 10.0);
        assert!(penalized[1] < logits[2]);
        assert_eq!(argmax(&penalized), 2);
        let mut rng = 1u64;
        let a = sample_token(&logits, 0.0, 0.9, 1.0, &[], &mut rng);
        assert_eq!(a, 1);
        let mut rng = 99u64;
        let b = sample_token(&logits, 0.8, 0.9, 1.1, &[1], &mut rng);
        assert!(b < 3);
        let ce = cross_entropy(&[0.0, 5.0, 0.0], 1);
        assert!(ce < 0.1, "{ce}");
    }
}
