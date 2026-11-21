//! In-place zero-copy 3-pass MeZO (`MezoScaleTuner` & `mezo_3pass`).
//!
//! Given deterministic seed `s` and stateless Rademacher `z_i = rademacher(s, i)`:
//! 1. `φ ← φ + ε z`, measure `L+` (forward #1)
//! 2. `φ ← φ - 2 ε z`, measure `L-` (forward #2, evaluated at `φ_0 - ε z`)
//! 3. `φ ← φ + (ε - Δ) z` where `Δ = clamp(lr * (L+ - L-) / (2ε), -max_step, max_step)`
//!    restores `φ_0` and applies the projected gradient step in one pass.
//!
//! Zero copies of `φ` or `z` are ever allocated, and each step runs **strictly
//! two pure forward passes** with zero autograd graph.

/// Stateless ±1 from `(seed, index)` via SplitMix64. Same seed regenerates the exact `z`.
#[inline]
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

/// Trait implemented by models whose continuous scales `φ` can be perturbed in place.
pub trait InPlacePhi {
    fn apply_z(&mut self, seed: u64, scale: f32);
    fn loss(&mut self) -> f32;
}

/// One unclipped 3-pass MeZO step. Calls `model.loss()` **exactly twice**. Returns `(L+, L-)`.
pub fn mezo_3pass(model: &mut impl InPlacePhi, seed: u64, eps: f32, lr: f32) -> (f32, f32) {
    assert!(eps > 0.0, "MeZO epsilon must be positive");
    model.apply_z(seed, eps);
    let lp = model.loss();
    model.apply_z(seed, -2.0 * eps);
    let lm = model.loss();
    let coeff = (lp - lm) / (2.0 * eps);
    model.apply_z(seed, eps - lr * coeff);
    (lp, lm)
}

#[derive(Clone, Copy, Debug)]
pub struct MezoStepReport {
    pub loss_plus: f32,
    pub loss_minus: f32,
    pub g_proj: f32,
}

/// Stateful zero-copy MeZO scale tuner with optional scalar step clipping
/// (prevents high-variance projected gradients from collapsing small row scales
/// while maintaining zero extra parameter buffers and 2 forwards per step).
#[derive(Clone, Debug)]
pub struct MezoScaleTuner {
    pub eps: f32,
    pub lr: f32,
    pub base_seed: u64,
    pub steps_done: u64,
    pub max_step: Option<f32>,
}

impl MezoScaleTuner {
    pub fn new(eps: f32, lr: f32, base_seed: u64) -> Self {
        Self {
            eps,
            lr,
            base_seed,
            steps_done: 0,
            max_step: None,
        }
    }

    pub fn with_max_step(mut self, max_step: f32) -> Self {
        self.max_step = Some(max_step);
        self
    }

    /// Execute one in-place MeZO step on `model` (strictly 2 forward passes, 0 clones).
    pub fn step_in_place(&mut self, model: &mut impl InPlacePhi) -> MezoStepReport {
        assert!(self.eps > 0.0, "MeZO epsilon must be positive");
        let seed = self.base_seed.wrapping_add(self.steps_done);
        model.apply_z(seed, self.eps);
        let lp = model.loss();
        model.apply_z(seed, -2.0 * self.eps);
        let lm = model.loss();
        let g_proj = (lp - lm) / (2.0 * self.eps);
        let mut delta = self.lr * g_proj;
        if let Some(cap) = self.max_step {
            delta = delta.clamp(-cap, cap);
        }
        model.apply_z(seed, self.eps - delta);
        self.steps_done += 1;
        MezoStepReport {
            loss_plus: lp,
            loss_minus: lm,
            g_proj,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bowl {
        phi: Vec<f32>,
        target: Vec<f32>,
        forwards: usize,
    }

    impl InPlacePhi for Bowl {
        fn apply_z(&mut self, seed: u64, scale: f32) {
            for (i, w) in self.phi.iter_mut().enumerate() {
                *w += scale * rademacher(seed, i);
            }
        }
        fn loss(&mut self) -> f32 {
            self.forwards += 1;
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
    fn quadratic_bowl_descends_in_strictly_two_forwards_per_step() {
        let mut bowl = Bowl {
            phi: vec![0.0; 8],
            target: vec![0.5, -0.25, 0.1, 0.0, 0.3, -0.4, 0.2, -0.1],
            forwards: 0,
        };
        let l0 = bowl.loss();
        bowl.forwards = 0;
        let mut tuner = MezoScaleTuner::new(0.05, 0.02, 7).with_max_step(0.05);
        for _ in 0..80 {
            tuner.step_in_place(&mut bowl);
        }
        assert_eq!(bowl.forwards, 160, "strictly 2 forward passes per step");
        let l1 = bowl.loss();
        assert!(l1 < l0 * 0.5, "quadratic MeZO {l0} -> {l1}");
    }
}
