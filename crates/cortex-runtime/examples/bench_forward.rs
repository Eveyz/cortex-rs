use cortex_runtime::generate::rss_kb;
use cortex_runtime::{CortexConfig, CortexModel};
use std::time::Instant;

fn main() {
    let model = CortexModel::from_seed(CortexConfig::micro(), 42);
    let mut state = model.new_state();
    let t0 = Instant::now();
    let n = 200usize;
    for i in 0..n {
        let _ = model.step((65 + (i % 26)) as u32, &mut state);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{n} forward steps in {ms:.2} ms ({:.3} ms/token, isa={}, rss_kb={})",
        ms / n as f64,
        cortex_kernels::active_isa(),
        rss_kb()
    );
}
