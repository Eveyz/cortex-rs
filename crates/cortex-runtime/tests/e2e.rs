use std::process::Command;

use cortex_runtime::ctxb::save_ctxb;
use cortex_runtime::model::{CortexConfig, CortexModel};

#[test]
fn test_end_to_end_cli_generation() {
    let dir = std::env::temp_dir().join(format!(
        "cortex-e2e-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("mini.ctxb");
    let model = CortexModel::from_seed(CortexConfig::mini(), 2);
    save_ctxb(&model, &model_path).unwrap();

    let bin = env!("CARGO_BIN_EXE_cortex-cli");
    let output = Command::new(bin)
        .args([
            "generate",
            "--model",
            model_path.to_str().unwrap(),
            "--prompt",
            "ab",
            "--max-tokens",
            "32",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--repetition-penalty",
            "1.0",
            "--seed",
            "1",
        ])
        .output()
        .expect("spawn cortex-cli");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "cli failed: {stderr}\n{stdout}");
    let ticks = stdout.lines().filter(|l| l.starts_with("token[")).count();
    assert_eq!(ticks, 32, "expected 32 streamed tokens\n{stdout}");
    assert!(
        stdout.contains("dt_us="),
        "missing per-token latency\n{stdout}"
    );
    assert!(stdout.contains("rss_kb="), "missing memory stats\n{stdout}");
    assert!(stdout.contains("tokens=32"), "{stdout}");
    // The lines themselves are the latency report. Print them so
    // `cargo test -- --nocapture` shows the measurement.
    print!("{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}
