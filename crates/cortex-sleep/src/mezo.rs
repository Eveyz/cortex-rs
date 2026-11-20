//! In-place 3-pass MeZO.
//!
//! Given seed `s` and Rademacher `z_i = rademacher(s, i)`:
//!
//! 1. `θ ← θ + ε z`, measure `L+`
//! 2. `θ ← θ - 2 ε z`, measure `L-`  (now at `θ_0 - ε z`)
//! 3. `θ ← θ + (ε - lr * (L+ - L-) / (2ε)) z`
//!    which lands on `θ_0 - lr * (L+ - L-) / (2ε) * z`
//!
//! Pass 3 both restores the probe and applies the update, so `z` is never
//! stored.

/// Stateless ±1 from `(seed, index)`. Same seed regenerates the same `z`.
pub fn rademacher(seed: u64, index: usize) -> f32 {
    let mut x = seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    if x & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Something MeZO can perturb in place and score with a scalar loss.
pub trait InPlacePhi {
    fn apply_z(&mut self, seed: u64, scale: f32);
    fn loss(&mut self) -> f32;
}

/// One MeZO step. Returns `(L+, L-)`.
pub fn mezo_3pass(model: &mut impl InPlacePhi, seed: u64, eps: f32, lr: f32) -> (f32, f32) {
    assert!(eps > 0.0, "MeZO epsilon must be positive");
    model.apply_z(seed, eps);
    let lp = model.loss();
    model.apply_z(seed, -2.0 * eps);
    let lm = model.loss();
    let coeff = (lp - lm) / (2.0 * eps);
    // From θ0 - εz, add (ε - lr*coeff) z → θ0 - lr*coeff z.
    model.apply_z(seed, eps - lr * coeff);
    (lp, lm)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bowl {
        phi: Vec<f32>,
        target: Vec<f32>,
    }

    impl InPlacePhi for Bowl {
        fn apply_z(&mut self, seed: u64, scale: f32) {
            for (i, w) in self.phi.iter_mut().enumerate() {
                *w += scale * rademacher(seed, i);
            }
        }
        fn loss(&mut self) -> f32 {
            self.phi
                .iter()
                .zip(self.target.iter())
                .map(|(p, t)| {
                    let d = p - t;
                    d * d
                })
                .sum()
        }
    }

    #[test]
    fn rademacher_is_reproducible_and_balanced() {
        let mut plus = 0i32;
        for i in 0..1000 {
            let z = rademacher(42, i);
            assert!(z == 1.0 || z == -1.0);
            assert_eq!(z, rademacher(42, i));
            plus += z as i32;
        }
        assert!(plus.abs() < 200, "expected a rough balance, got {plus}");
    }

    #[test]
    fn quadratic_bowl_descends_without_a_gradient_buffer() {
        let mut bowl = Bowl {
            phi: vec![0.0; 8],
            target: vec![0.5, -0.25, 0.1, 0.0, 0.3, -0.4, 0.2, -0.1],
        };
        let l0 = bowl.loss();
        let mut seed = 7u64;
        // Small step size: one-sample MeZO on 8 coordinates is noisy, and the
        // quadratic Hessian is 2, so a large lr diverges.
        for _ in 0..80 {
            mezo_3pass(&mut bowl, seed, 0.05, 0.02);
            seed = seed.wrapping_add(1);
        }
        let l1 = bowl.loss();
        assert!(l1 < l0 * 0.5, "quadratic MeZO {l0} -> {l1}");
    }
}
