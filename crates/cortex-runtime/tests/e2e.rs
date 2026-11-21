use std::process::Command;

use cortex_runtime::ctxb::save_ctxb;
use cortex_runtime::model::{CortexConfig, CortexModel};

#[test]
fn test_end_to_end_cli_generation_sleep_rollback_and_tools() {
    let dir = std::env::temp_dir().join(format!(
        "cortex-e2e-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("tiny.ctxb");
    let model = CortexModel::from_seed(CortexConfig::mezo_tiny(), 2);
    save_ctxb(&model, &model_path).unwrap();

    let bin = env!("CARGO_BIN_EXE_cortex-cli");

    // 1. Streaming generation + real RSS telemetry + WAL persistence.
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
        .expect("spawn cortex-cli generate");
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
    let rss_line = stdout
        .lines()
        .find(|l| l.starts_with("rss_kb="))
        .expect("rss_kb summary line");
    let rss_val: u64 = rss_line
        .trim_start_matches("rss_kb=")
        .parse()
        .expect("rss_kb parse");
    #[cfg(unix)]
    assert!(rss_val > 0, "rss_kb must be non-zero on Unix: {stdout}");
    assert!(stdout.contains("tokens=32"), "{stdout}");
    print!("{stdout}");

    // 2. `/sleep` runs WAL + anchor mixed replay, MeZO, consolidation, and HPSP snapshots.
    let sleep_out = Command::new(bin)
        .args([
            "/sleep",
            "--model",
            model_path.to_str().unwrap(),
            "--steps",
            "3",
        ])
        .output()
        .expect("spawn cortex-cli /sleep");
    let sleep_stdout = String::from_utf8_lossy(&sleep_out.stdout);
    assert!(
        sleep_out.status.success(),
        "sleep failed: {}\n{sleep_stdout}",
        String::from_utf8_lossy(&sleep_out.stderr)
    );
    assert!(sleep_stdout.contains("slept ce"), "{sleep_stdout}");
    assert!(sleep_stdout.contains("snap=Some(2)"), "{sleep_stdout}");

    // 3. `/reset-mem` rolls back to the pre-sleep snapshot `Some(1)`.
    let reset_out = Command::new(bin)
        .args(["/reset-mem", "--model", model_path.to_str().unwrap()])
        .output()
        .expect("spawn cortex-cli /reset-mem");
    let reset_stdout = String::from_utf8_lossy(&reset_out.stdout);
    assert!(reset_out.status.success(), "{reset_stdout}");
    assert!(
        reset_stdout.contains("rollback=Some(1)"),
        "expected snapshot rollback to v1, got: {reset_stdout}"
    );

    // 4. `tools` renders the pinned 5-tool system prompt.
    let tools_out = Command::new(bin)
        .args(["tools"])
        .output()
        .expect("spawn cortex-cli tools");
    let tools_stdout = String::from_utf8_lossy(&tools_out.stdout);
    assert!(tools_out.status.success());
    assert!(tools_stdout.contains("- calculator:"));
    assert!(tools_stdout.contains("<tool_call>"));

    let _ = std::fs::remove_dir_all(&dir);
}
