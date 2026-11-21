//! Hippocampus orthogonalization & singular-direction slimming during sleep.
//!
//! * [`orthogonalize_rows`]: Gram–Schmidt cosine deduplication in place. Rows
//!   whose cosine similarity with an earlier row exceeds `cosine_threshold`
//!   have their parallel component subtracted and are rescaled to preserve
//!   norm, reclaiming capacity wasted by redundant daytime memories.
//! * [`soft_singular_suppress`]: shrinks over-large row norms toward the mean
//!   row norm, preventing saturated directions from dominating retrieval.

use cortex_hippocampus::HippocampusWeights;

/// Gram–Schmidt approximate deduplication of the rows of a `[rows, cols]`
/// row-major matrix in place. Returns the number of deduplicated rows.
pub fn orthogonalize_rows(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    cosine_threshold: f32,
) -> usize {
    assert_eq!(weights.len(), rows * cols);
    let mut deduped = 0usize;
    for i in 0..rows {
        let (done, rest) = weights.split_at_mut(i * cols);
        let (row_i, _) = rest.split_at_mut(cols);
        let norm_i0: f32 = row_i.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm_i0 == 0.0 {
            continue;
        }
        let mut changed = false;
        for j in 0..i {
            let row_j = &done[j * cols..(j + 1) * cols];
            let norm_j: f32 = row_j.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_j == 0.0 {
                continue;
            }
            let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            let cos = dot / (norm_i0 * norm_j);
            if cos.abs() > cosine_threshold {
                let coeff = dot / (norm_j * norm_j);
                for c in 0..cols {
                    row_i[c] -= coeff * row_j[c];
                }
                changed = true;
            }
        }
        if changed {
            let norm_i1: f32 = row_i.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_i1 > 1e-9 {
                let s = norm_i0 / norm_i1;
                for x in row_i.iter_mut() {
                    *x *= s;
                }
            }
            deduped += 1;
        }
    }
    deduped
}

/// Shrink over-large row norms toward the mean norm by `shrink` (`0..=1`).
pub fn soft_singular_suppress(weights: &mut [f32], rows: usize, cols: usize, shrink: f32) -> usize {
    assert_eq!(weights.len(), rows * cols);
    if rows == 0 {
        return 0;
    }
    let norms: Vec<f32> = (0..rows)
        .map(|i| {
            weights[i * cols..(i + 1) * cols]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    let mean = norms.iter().sum::<f32>() / rows as f32;
    let mut shrunk = 0usize;
    for i in 0..rows {
        if norms[i] > mean && norms[i] > 0.0 {
            let target = mean + (norms[i] - mean) * (1.0 - shrink);
            let s = target / norms[i];
            for x in weights[i * cols..(i + 1) * cols].iter_mut() {
                *x *= s;
            }
            shrunk += 1;
        }
    }
    shrunk
}

/// Consolidate a hippocampus weight snapshot in place: orthogonalize `W1` & `W2`
/// and softly suppress saturated rows. Returns `(deduped_w1, deduped_w2)`.
pub fn consolidate_hippocampus(
    weights: &mut HippocampusWeights,
    cosine_threshold: f32,
) -> (usize, usize) {
    let d1 = orthogonalize_rows(
        weights.w1.as_mut_slice(),
        weights.d_hidden,
        weights.d_in,
        cosine_threshold,
    );
    let d2 = orthogonalize_rows(
        weights.w2.as_mut_slice(),
        weights.d_out,
        weights.d_hidden,
        cosine_threshold,
    );
    soft_singular_suppress(
        weights.w1.as_mut_slice(),
        weights.d_hidden,
        weights.d_in,
        0.2,
    );
    soft_singular_suppress(
        weights.w2.as_mut_slice(),
        weights.d_out,
        weights.d_hidden,
        0.2,
    );
    weights.reset_momentum();
    (d1, d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonalize_removes_duplicate_rows_and_preserves_norm() {
        let mut w = vec![
            1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
        ];
        let n = orthogonalize_rows(&mut w, 3, 3, 0.9);
        assert_eq!(n, 1);
        let dot: f32 = (0..3).map(|c| w[c] * w[3 + c]).sum();
        assert!(dot.abs() < 0.5, "deduped rows stayed parallel: dot={dot}");
        let n0: f32 = w[0..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn consolidate_hippocampus_keeps_weights_finite() {
        let mut w = HippocampusWeights::new(8, 16, 8);
        let _ = consolidate_hippocampus(&mut w, 0.95);
        assert!(w.w1.as_slice().iter().all(|x| x.is_finite()));
        assert!(w.w2.as_slice().iter().all(|x| x.is_finite()));
    }
}
