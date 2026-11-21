//! cortex-cli — local 1.58-bit BitSSM + online neural hippocampus runtime.
//!
//! ```text
//! cortex-cli generate --model toy.ctxb --prompt "hi" --max-tokens 32 \
//!     --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 --seed 1
//! cortex-cli agent --model toy.ctxb --prompt "What is 6 * 7?" --workspace ./cortex-work
//! cortex-cli tools
//! cortex-cli repl --model toy.ctxb
//! cortex-cli logits --model toy.ctxb --ids 1,2,3
//! cortex-cli --check-ctxb toy.ctxb --prompt "hello"
//! cortex-cli /status --model toy.ctxb
//! cortex-cli /sleep --model toy.ctxb
//! cortex-cli /reset-mem --model toy.ctxb
//! cortex-cli serve --model toy.ctxb --bind 127.0.0.1:43123 --unix /tmp/cortex.sock
//! ```

use std::env;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, RwLock};

use cortex_hippocampus::SnapshotStore;
use cortex_kernels::{l2_normalize_inplace, quantize_activations_sym_i8, sigmoid, silu};
use cortex_runtime::ctxb::{load_ctxb, save_ctxb};
use cortex_runtime::expert::{
    list_packages, load_package, save_package, tiny_adapter, ExpertMeta, ExpertPackage, ROUTER_ABI,
};
use cortex_runtime::fsm::{FsmEvent, RuntimeFsm};
use cortex_runtime::generate::{format_tick, next_token, rss_kb, Session};
use cortex_runtime::model::{CortexConfig, CortexModel, ModelState};
use cortex_runtime::tokenizer::Tokenizer;
use cortex_runtime::tools::{AgentLoop, ToolRegistry};
use cortex_sleep::ReplayBuffer;
use cortex_ssm::deltanet_step;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::from(2);
    }
    if args[0] == "--check-ctxb" {
        return cmd_check_ctxb(&args[1..]);
    }
    if let Some(stripped) = args[0].strip_prefix('/') {
        args[0] = stripped.to_string();
    }
    let cmd = args[0].as_str();
    match cmd {
        "help" | "-h" | "--help" => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        "generate" => cmd_generate(&args[1..]),
        "agent" => cmd_agent(&args[1..]),
        "tools" => cmd_tools(),
        "repl" => cmd_repl(&args[1..]),
        "logits" => cmd_logits(&args[1..]),
        "dump" => cmd_dump(&args[1..]),
        "status" => cmd_status(&args[1..]),
        "sleep" => cmd_sleep(&args[1..]),
        "reset-mem" => cmd_reset(&args[1..]),
        "serve" => cmd_serve(&args[1..]),
        "init-mini" => cmd_init_mini(&args[1..]),
        "encode" => cmd_encode(&args[1..]),
        "export-expert" => cmd_export_expert(&args[1..]),
        "install-expert" => cmd_install_expert(&args[1..]),
        "list-experts" => cmd_list_experts(&args[1..]),
        other => {
            eprintln!("unknown command {other}\n{}", usage());
            ExitCode::from(2)
        }
    }
}

fn usage() -> &'static str {
    "cortex-cli commands:\n  \
     generate --model FILE --prompt TEXT [--vocab MERGES] [--max-tokens 32] [--temperature 0.8]\n            \
     [--top-p 0.9] [--repetition-penalty 1.1] [--seed 1] [--no-plasticity] [--save-model]\n  \
     agent --model FILE --prompt TEXT [--workspace DIR] [--vocab MERGES] [--max-tokens 128] [--max-steps 5]\n  \
     tools\n  \
     repl --model FILE [--vocab MERGES] [--workspace DIR]\n  \
     encode --vocab MERGES --input TEXT --output TOKENS.u32\n  \
     export-expert --domain NAME --license MIT --d-model N --vocab N --out FILE.cxex\n  \
     install-expert --pack FILE.cxex --dir DIR --d-model N --vocab N\n  \
     list-experts --dir DIR\n  \
     logits --model FILE [--ids 1,2,3 | --prompt TEXT]\n  \
     dump --model FILE [--ids 1,2,3 | --prompt TEXT] [--out FILE]\n  \
     --check-ctxb FILE [--prompt TEXT]\n  \
     /status --model FILE\n  \
     /sleep --model FILE [--steps 2] [--eps 0.02] [--lr 0.01]\n  \
     /reset-mem --model FILE [--version V]\n  \
     serve --model FILE [--bind 127.0.0.1:43123] [--unix PATH]\n  \
     init-mini --out FILE [--shape mini|tiny|256|micro|step1] [--seed 1]\n\n\
Sampling: temperature 0 is greedy; otherwise softmax(logits/T) over the top-p nucleus,\n\
with repetition penalty and constrained JSON Grammar FSM masking inside <|tool_call|>.\n\
/sleep runs 80% WAL + 20% anchor mixed replay with zero-copy in-place MeZO + Gram-Schmidt consolidation.\n\
/reset-mem rolls hippocampus back to the prior HPSP snapshot (or deterministic init)."
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn wal_dir_for(model_path: &Path) -> PathBuf {
    model_path.with_extension("wal")
}

fn snap_dir_for(model_path: &Path) -> PathBuf {
    model_path.with_extension("snaps")
}

fn anchor_seqs_for(vocab: usize) -> Vec<Vec<u32>> {
    let cap = vocab.max(2) as u32;
    vec![
        vec![1 % cap, 2 % cap, 3 % cap, 4 % cap],
        vec![4 % cap, 3 % cap, 2 % cap, 1 % cap],
        vec![1 % cap, 1 % cap, 2 % cap, 2 % cap],
    ]
}

fn require_model(args: &[String]) -> Result<(CortexModel, PathBuf), ExitCode> {
    let path = flag(args, "--model").ok_or_else(|| {
        eprintln!("missing --model");
        ExitCode::from(2)
    })?;
    let pb = PathBuf::from(path);
    let m = load_ctxb(&pb).map_err(|e| {
        eprintln!("load {path}: {e}");
        ExitCode::from(1)
    })?;
    Ok((m, pb))
}

fn load_tokenizer_arg(args: &[String]) -> Result<Tokenizer, ExitCode> {
    match flag(args, "--vocab") {
        Some(path) => Tokenizer::from_merge_file(Path::new(path)).map_err(|e| {
            eprintln!("vocab {path}: {e}");
            ExitCode::from(1)
        }),
        None => Ok(Tokenizer::byte_level()),
    }
}

fn cmd_generate(args: &[String]) -> ExitCode {
    let (model, model_path) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let prompt = flag(args, "--prompt").unwrap_or("");
    let max_tokens: usize = flag(args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let temperature: f32 = flag(args, "--temperature")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8);
    let top_p: f32 = flag(args, "--top-p")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);
    let penalty: f32 = flag(args, "--repetition-penalty")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.1);
    let seed: u64 = flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let online_plasticity = !has_flag(args, "--no-plasticity");
    let save_model = has_flag(args, "--save-model");

    if let Some(tools) = flag(args, "--tools") {
        let text = match std::fs::read_to_string(tools) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("tools {tools}: {e}");
                return ExitCode::from(1);
            }
        };
        let specs = cortex_runtime::harness::load_tool_file(&text);
        println!("tools={}", specs.len());
    }

    let tokenizer = match load_tokenizer_arg(args) {
        Ok(tok) => tok,
        Err(c) => return c,
    };

    if online_plasticity {
        model.attach_wal_dir(&wal_dir_for(&model_path));
    }

    let mut session = Session::with_options(
        &model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        penalty,
        seed,
        tokenizer,
        online_plasticity,
        true,
        true,
    );
    let mut n = 0usize;
    while let Some(tick) = next_token(&mut session) {
        println!("{}", format_tick(&tick));
        n += 1;
    }
    if online_plasticity {
        model.flush_hippocampus();
        if save_model {
            if let Err(e) = save_ctxb(&model, &model_path) {
                eprintln!("save {}: {e}", model_path.display());
                return ExitCode::from(1);
            }
        }
    }
    println!("tokens={n}");
    println!("continuation={}", session.continuation_text());
    println!("rss_kb={}", rss_kb());
    ExitCode::SUCCESS
}

fn cmd_tools() -> ExitCode {
    println!("{}", ToolRegistry::new().render_system_prompt());
    ExitCode::SUCCESS
}

fn cmd_agent(args: &[String]) -> ExitCode {
    let (model, model_path) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let request = flag(args, "--prompt").unwrap_or("What is 6 * 7?");
    let workspace = PathBuf::from(flag(args, "--workspace").unwrap_or("./cortex-work"));
    let max_tokens: usize = flag(args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let max_steps: usize = flag(args, "--max-steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let tokenizer = match load_tokenizer_arg(args) {
        Ok(tok) => tok,
        Err(c) => return c,
    };
    if let Err(e) = std::fs::create_dir_all(&workspace) {
        eprintln!("workspace {}: {e}", workspace.display());
        return ExitCode::from(1);
    }
    model.attach_wal_dir(&wal_dir_for(&model_path));
    let mut agent = AgentLoop::new(workspace);
    agent.max_steps = max_steps;
    let outcome = agent.run(request, &mut |transcript| {
        let mut session = Session::with_tokenizer(
            &model,
            transcript,
            max_tokens,
            0.0,
            1.0,
            1.0,
            1,
            tokenizer.clone(),
        );
        while next_token(&mut session).is_some() {}
        session.continuation_text()
    });
    model.flush_hippocampus();
    if has_flag(args, "--transcript") {
        println!("{}", outcome.transcript);
    }
    println!(
        "steps={} tool_calls={} finished={}",
        outcome.steps, outcome.tool_calls, outcome.finished
    );
    println!("answer={}", outcome.answer);
    ExitCode::SUCCESS
}

struct SleepSummary {
    before: f32,
    after: f32,
    wal_eps: usize,
    dedup_w1: usize,
    dedup_w2: usize,
    snap_ver: Option<u64>,
}

fn run_sleep_cycle(
    model: &mut CortexModel,
    model_path: &Path,
    steps: usize,
    eps: f32,
    lr: f32,
    seed: u64,
    fsm: &mut RuntimeFsm,
) -> Result<SleepSummary, String> {
    fsm.event(FsmEvent::SleepRequested);
    model.flush_hippocampus();

    let mut replay = ReplayBuffer::new();
    let wal_root = wal_dir_for(model_path);
    for m in 0..model.cfg.n_macros {
        let wf = wal_root.join(format!("macro_{m}")).join("episodes.wal");
        if wf.exists() {
            let _ = replay.load_wal(&wf);
        }
    }
    for seq in anchor_seqs_for(model.cfg.vocab) {
        replay.add_anchor_tokens(seq);
    }
    let wal_eps = replay.wal_episodes.len();
    let seqs = replay.mixed_token_sequences(10);
    let (before, after) = model.mezo_steps(&seqs, steps, eps, lr, seed);

    let sdir = snap_dir_for(model_path);
    let snap_store = SnapshotStore::new(&sdir).map_err(|e| e.to_string())?;
    if snap_store.list_versions().is_empty() {
        let pre: Vec<_> = model.mem.iter().map(|m| m.hip.snapshot()).collect();
        snap_store.save_macros(&pre).map_err(|e| e.to_string())?;
    }
    let (dedup_w1, dedup_w2, snap_ver) = model
        .consolidate_all_hippocampus(Some(&snap_store))
        .map_err(|e| e.to_string())?;
    save_ctxb(model, model_path)?;
    fsm.event(FsmEvent::SleepFinished);
    Ok(SleepSummary {
        before,
        after,
        wal_eps,
        dedup_w1,
        dedup_w2,
        snap_ver,
    })
}

fn run_reset_mem(
    model: &CortexModel,
    model_path: &Path,
    target_version: Option<u64>,
) -> Result<Option<u64>, String> {
    model.flush_hippocampus();
    let sdir = snap_dir_for(model_path);
    let mut rolled_back = None;
    if let Ok(store) = SnapshotStore::new(&sdir) {
        if let Some(ver) = target_version {
            model
                .rollback_hippocampus(&store, ver)
                .map_err(|e| e.to_string())?;
            rolled_back = Some(ver);
        } else {
            let vs = store.list_versions();
            if vs.len() >= 2 {
                let prev = vs[vs.len() - 2];
                if model.rollback_hippocampus(&store, prev).is_ok() {
                    rolled_back = Some(prev);
                }
            }
        }
    }
    if rolled_back.is_none() {
        model.reset_memory_to_init();
    }
    save_ctxb(model, model_path)?;
    Ok(rolled_back)
}

fn cmd_repl(args: &[String]) -> ExitCode {
    let (mut model, model_path) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let tokenizer = match load_tokenizer_arg(args) {
        Ok(tok) => tok,
        Err(c) => return c,
    };
    let workspace = PathBuf::from(flag(args, "--workspace").unwrap_or("./cortex-work"));
    let _ = std::fs::create_dir_all(&workspace);
    model.attach_wal_dir(&wal_dir_for(&model_path));

    let mut fsm = RuntimeFsm::default();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    println!(
        "Cortex-Local REPL (layers={} d_model={} isa={}) — type /help for commands",
        model.n_layers(),
        model.cfg.d_model,
        cortex_kernels::active_isa()
    );
    loop {
        print!("cortex> ");
        let _ = stdout.flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin: {e}");
                return ExitCode::from(1);
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        fsm.event(FsmEvent::UserActivity);
        match trimmed {
            "/quit" | "/exit" => break,
            "/help" => {
                println!("commands: /status | /sleep | /reset-mem | /tools | /agent <req> | /quit");
            }
            "/status" => {
                println!(
                    "layers={} d_model={} vocab={} hip_version={} rss_kb={} {}",
                    model.n_layers(),
                    model.cfg.d_model,
                    model.cfg.vocab,
                    model.hip_version_sum(),
                    rss_kb(),
                    fsm.status()
                );
            }
            "/tools" => {
                println!("{}", ToolRegistry::new().render_system_prompt());
            }
            "/sleep" => {
                match run_sleep_cycle(&mut model, &model_path, 4, 1e-4, 5e-6, 7, &mut fsm) {
                    Ok(s) => println!(
                        "slept ce {:.6} -> {:.6} wal_eps={} dedup={}+{} snap={:?} {}",
                        s.before,
                        s.after,
                        s.wal_eps,
                        s.dedup_w1,
                        s.dedup_w2,
                        s.snap_ver,
                        fsm.status()
                    ),
                    Err(e) => eprintln!("sleep error: {e}"),
                }
            }
            "/reset-mem" => match run_reset_mem(&model, &model_path, None) {
                Ok(rb) => println!("hip_version={} rollback={rb:?}", model.hip_version_sum()),
                Err(e) => eprintln!("reset-mem error: {e}"),
            },
            rest if rest.starts_with("/agent ") => {
                let req = rest.trim_start_matches("/agent ").trim();
                let agent = AgentLoop::new(workspace.clone());
                let out = agent.run(req, &mut |transcript| {
                    let mut s = Session::with_tokenizer(
                        &model,
                        transcript,
                        96,
                        0.0,
                        1.0,
                        1.0,
                        1,
                        tokenizer.clone(),
                    );
                    while next_token(&mut s).is_some() {}
                    s.continuation_text()
                });
                model.flush_hippocampus();
                println!(
                    "[steps={} calls={}] {}",
                    out.steps, out.tool_calls, out.answer
                );
            }
            prompt => {
                let mut s = Session::with_tokenizer(
                    &model,
                    prompt,
                    64,
                    0.8,
                    0.9,
                    1.1,
                    1,
                    tokenizer.clone(),
                );
                while next_token(&mut s).is_some() {}
                model.flush_hippocampus();
                fsm.event(FsmEvent::EndOfSentence);
                println!("{}", s.continuation_text());
            }
        }
    }
    ExitCode::SUCCESS
}

fn cmd_encode(args: &[String]) -> ExitCode {
    let vocab = match flag(args, "--vocab") {
        Some(p) => p,
        None => {
            eprintln!("missing --vocab");
            return ExitCode::from(2);
        }
    };
    let input = match flag(args, "--input") {
        Some(p) => p,
        None => {
            eprintln!("missing --input");
            return ExitCode::from(2);
        }
    };
    let output = match flag(args, "--output") {
        Some(p) => p,
        None => {
            eprintln!("missing --output");
            return ExitCode::from(2);
        }
    };
    let tok = match Tokenizer::from_merge_file(Path::new(vocab)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("vocab: {e}");
            return ExitCode::from(1);
        }
    };
    let text = match std::fs::read_to_string(input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read {input}: {e}");
            return ExitCode::from(1);
        }
    };
    let ids = tok.encode(&text);
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for id in &ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    if let Err(e) = std::fs::write(output, &bytes) {
        eprintln!("write {output}: {e}");
        return ExitCode::from(1);
    }
    eprintln!("tokens={}", ids.len());
    ExitCode::SUCCESS
}

fn cmd_export_expert(args: &[String]) -> ExitCode {
    let domain = flag(args, "--domain").unwrap_or("general");
    let license = flag(args, "--license").unwrap_or("MIT");
    let d_model: u32 = flag(args, "--d-model")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let vocab: u32 = flag(args, "--vocab")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let out = match flag(args, "--out") {
        Some(p) => p,
        None => {
            eprintln!("missing --out");
            return ExitCode::from(2);
        }
    };
    let meta = ExpertMeta {
        name: domain.to_string(),
        domain: domain.to_string(),
        license: license.to_string(),
        d_model,
        vocab,
        router_abi: ROUTER_ABI,
    };
    let pkg = match ExpertPackage::export(meta, tiny_adapter(domain, 1)) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("export: {err:?}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = save_package(Path::new(out), &pkg) {
        eprintln!("write: {err}");
        return ExitCode::from(1);
    }
    println!("exported {out}");
    ExitCode::SUCCESS
}

fn cmd_install_expert(args: &[String]) -> ExitCode {
    let pack = match flag(args, "--pack") {
        Some(p) => p,
        None => {
            eprintln!("missing --pack");
            return ExitCode::from(2);
        }
    };
    let dir = flag(args, "--dir").unwrap_or("experts");
    let pkg = match load_package(Path::new(pack)) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("load: {err:?}");
            return ExitCode::from(1);
        }
    };
    let d_model: u32 = flag(args, "--d-model")
        .and_then(|s| s.parse().ok())
        .unwrap_or(pkg.meta.d_model);
    let vocab: u32 = flag(args, "--vocab")
        .and_then(|s| s.parse().ok())
        .unwrap_or(pkg.meta.vocab);
    if pkg.meta.d_model != d_model || pkg.meta.vocab != vocab || pkg.meta.router_abi != ROUTER_ABI {
        eprintln!("incompatible expert");
        return ExitCode::from(1);
    }
    let dest = Path::new(dir).join(format!("{}.cxex", pkg.meta.domain));
    if let Err(err) = save_package(&dest, &pkg) {
        eprintln!("install: {err}");
        return ExitCode::from(1);
    }
    println!("installed {}", dest.display());
    ExitCode::SUCCESS
}

fn cmd_list_experts(args: &[String]) -> ExitCode {
    let dir = flag(args, "--dir").unwrap_or("experts");
    let found = list_packages(Path::new(dir));
    if found.is_empty() {
        println!("experts=0");
        return ExitCode::SUCCESS;
    }
    for path in &found {
        match load_package(path) {
            Ok(pkg) => println!(
                "{} domain={} license={} d_model={} vocab={} abi={}",
                path.display(),
                pkg.meta.domain,
                pkg.meta.license,
                pkg.meta.d_model,
                pkg.meta.vocab,
                pkg.meta.router_abi
            ),
            Err(err) => println!("{} error={err:?}", path.display()),
        }
    }
    println!("experts={}", found.len());
    ExitCode::SUCCESS
}

fn cmd_logits(args: &[String]) -> ExitCode {
    let (model, _) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let tokens: Vec<u32> = if let Some(prompt) = flag(args, "--prompt") {
        let tok = match load_tokenizer_arg(args) {
            Ok(t) => t,
            Err(c) => return c,
        };
        tok.encode(prompt)
    } else {
        let ids = flag(args, "--ids").unwrap_or("1");
        ids.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    };
    let mut state = model.new_state();
    for (t, id) in tokens.iter().enumerate() {
        let logits = model.step(*id, &mut state);
        let line: Vec<String> = logits.iter().map(|v| format!("{v:.8}")).collect();
        println!("pos={t} id={id} {}", line.join(" "));
    }
    ExitCode::SUCCESS
}

/// `dump` — diagnostic mirror of the forward pass.
///
/// Writes one line per (position, block/macro, quantity), FP vectors in
/// `%.9e` and INT8 activations as integers:
///   `S pos=.. block=.. in|out`  SSM block input/output
///   `SQ/SK/SV/SO/SP`            SSM q/k/v projections, DeltaNet output, o_proj output
///   `QO`                        o_proj input INT8 quantization
///   `SX`                        FFN input (norm2 output)
///   `QX`                        FFN input INT8 quantization
///   `FG/FU/FH/SF`               FFN gate/up/hidden/down outputs
///   `X/A/Q/M`                   mem-block input, ring-attention output, hippo q, hippo m
///   `L pos=..`                  logits
///
/// This replicates `BitSsmBlock::forward` and `MemoryBlock::forward_step`
/// by hand so the intermediates can be dumped; keep it in sync with those
/// two functions. `dump` output must reproduce `logits` output (the `L`
/// lines) — the Python self-check verifies that, so drift is caught.
fn cmd_dump(args: &[String]) -> ExitCode {
    let (model, _) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let tokens: Vec<u32> = if let Some(prompt) = flag(args, "--prompt") {
        let tok = match load_tokenizer_arg(args) {
            Ok(t) => t,
            Err(c) => return c,
        };
        tok.encode(prompt)
    } else {
        let ids = flag(args, "--ids").unwrap_or("1");
        ids.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    };
    let mut out: Box<dyn Write> = match flag(args, "--out") {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => Box::new(std::io::BufWriter::new(f)),
            Err(e) => {
                eprintln!("create {path}: {e}");
                return ExitCode::from(1);
            }
        },
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };
    let d = model.cfg.d_model;
    let mut state: ModelState = model.new_state();
    for (t, &tok) in tokens.iter().enumerate() {
        let mut x = model.embed(tok);
        for m in 0..model.cfg.n_macros {
            for s in 0..3 {
                let idx = m * 3 + s;
                let blk = &model.ssm[idx];
                if writeln!(out, "S pos={t} block={idx} in {}", fmt_vec(&x)).is_err() {
                    return ExitCode::from(1);
                }
                // Manual replication of BitSsmBlock::forward, with dumps.
                let xn = blk.norm1.forward(&x);
                let q = blk.q_proj.forward(&xn);
                let k = blk.k_proj.forward(&xn);
                let v = blk.v_proj.forward(&xn);
                let (alpha, beta) = {
                    let d = blk.d_model;
                    let aw = blk.alpha_w.as_slice();
                    let bw = blk.beta_w.as_slice();
                    let ab = blk.alpha_b.as_slice();
                    let bb = blk.beta_b.as_slice();
                    let mut alpha = vec![0f32; blk.n_heads];
                    let mut beta = vec![0f32; blk.n_heads];
                    for h in 0..blk.n_heads {
                        let mut da = ab[h];
                        let mut db = bb[h];
                        let row = h * d;
                        for i in 0..d {
                            da += aw[row + i] * xn[i];
                            db += bw[row + i] * xn[i];
                        }
                        alpha[h] = sigmoid(da);
                        beta[h] = sigmoid(db);
                    }
                    (alpha, beta)
                };
                let o_delta = deltanet_step(&mut state.ssm[idx], &q, &k, &v, &alpha, &beta);
                let o = blk.o_proj.forward(&o_delta);
                let mut x1 = vec![0f32; d];
                for i in 0..d {
                    x1[i] = x[i] + o[i];
                }
                let xn2 = blk.norm2.forward(&x1);
                let gate_out = blk.ffn.w_gate.forward(&xn2);
                let up_out = blk.ffn.w_up.forward(&xn2);
                let mut hidden = vec![0f32; gate_out.len()];
                for i in 0..hidden.len() {
                    hidden[i] = silu(gate_out[i]) * up_out[i];
                }
                let ffn = blk.ffn.w_down.forward(&hidden);
                for i in 0..d {
                    x1[i] += ffn[i];
                }
                x = x1;
                let (qn, _) = quantize_activations_sym_i8(&xn);
                let (qo, _) = quantize_activations_sym_i8(&o_delta);
                let (qx, _) = quantize_activations_sym_i8(&xn2);
                let (qh, _) = quantize_activations_sym_i8(&hidden);
                let lines: [(&str, String); 14] = [
                    ("SN", fmt_vec(&xn)),
                    ("QN", fmt_ints(&qn)),
                    ("SP", fmt_vec(&o)),
                    ("QO", fmt_ints(&qo)),
                    ("SX", fmt_vec(&xn2)),
                    ("QX", fmt_ints(&qx)),
                    ("SQ", fmt_vec(&q)),
                    ("SK", fmt_vec(&k)),
                    ("SV", fmt_vec(&v)),
                    ("SO", fmt_vec(&o_delta)),
                    ("FG", fmt_vec(&gate_out)),
                    ("FU", fmt_vec(&up_out)),
                    ("FH", fmt_vec(&hidden)),
                    ("QH", fmt_ints(&qh)),
                ];
                for (tag, body) in lines {
                    if writeln!(out, "{tag} pos={t} block={idx} {body}").is_err() {
                        return ExitCode::from(1);
                    }
                }
                if writeln!(out, "SF pos={t} block={idx} {}", fmt_vec(&ffn)).is_err() {
                    return ExitCode::from(1);
                }
                if writeln!(out, "S pos={t} block={idx} out {}", fmt_vec(&x)).is_err() {
                    return ExitCode::from(1);
                }
            }
            let mem = &model.mem[m];
            let xn = mem.norm1.forward(&x);
            let (mq, _) = quantize_activations_sym_i8(&xn);
            let a = mem.attn.forward(&xn, &mut state.rings[m]);
            if writeln!(out, "X pos={t} macro={m} {}", fmt_vec(&x)).is_err() {
                return ExitCode::from(1);
            }
            if writeln!(out, "MN pos={t} macro={m} {}", fmt_vec(&xn)).is_err() {
                return ExitCode::from(1);
            }
            if writeln!(out, "MQ pos={t} macro={m} {}", fmt_ints(&mq)).is_err() {
                return ExitCode::from(1);
            }
            if writeln!(out, "A pos={t} macro={m} {}", fmt_vec(&a)).is_err() {
                return ExitCode::from(1);
            }
            let mut q = mem.hip_proj.forward(&xn);
            l2_normalize_inplace(&mut q);
            let mut mvec = vec![0f32; d];
            mem.hip.read_forward(&q, &mut mvec);
            if writeln!(out, "Q pos={t} macro={m} {}", fmt_vec(&q)).is_err() {
                return ExitCode::from(1);
            }
            if writeln!(out, "M pos={t} macro={m} {}", fmt_vec(&mvec)).is_err() {
                return ExitCode::from(1);
            }
            // Gate + FFN (mirror MemoryBlock::forward_step).
            let gw = mem.gate_w.as_slice();
            let gb = mem.gate_b.as_slice();
            let mut y = vec![0f32; d];
            for i in 0..d {
                let g = sigmoid(gw[i] * xn[i] + gb[i]);
                y[i] = x[i] + g * a[i] + (1.0 - g) * mvec[i];
            }
            let xn2 = mem.norm2.forward(&y);
            let (mx, _) = quantize_activations_sym_i8(&xn2);
            if writeln!(out, "MY pos={t} macro={m} {}", fmt_vec(&xn2)).is_err() {
                return ExitCode::from(1);
            }
            if writeln!(out, "MX pos={t} macro={m} {}", fmt_ints(&mx)).is_err() {
                return ExitCode::from(1);
            }
            let ffn = mem.ffn.forward(&xn2);
            for i in 0..d {
                y[i] += ffn[i];
            }
            x = y;
        }
        let logits = model.logits_from_hidden(&x);
        if writeln!(out, "L pos={t} {}", fmt_vec(&logits)).is_err() {
            return ExitCode::from(1);
        }
    }
    if out.flush().is_err() {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn fmt_vec(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{x:.9e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn fmt_ints(v: &[i8]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn cmd_check_ctxb(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: cortex-cli --check-ctxb FILE [--prompt TEXT]");
        return ExitCode::from(2);
    }
    let model = match load_ctxb(Path::new(&args[0])) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load {}: {e}", args[0]);
            return ExitCode::from(1);
        }
    };
    let prompt = flag(&args[1..], "--prompt").unwrap_or("hello");
    let tok = Tokenizer::byte_level();
    let ids = tok.encode(prompt);
    let mut state = model.new_state();
    let mut logits = vec![0f32; model.cfg.vocab];
    if ids.is_empty() {
        logits = model.step(0, &mut state);
    } else {
        for &id in &ids {
            logits = model.step(id, &mut state);
        }
    }
    for v in logits {
        println!("{v:.8}");
    }
    ExitCode::SUCCESS
}

fn cmd_status(args: &[String]) -> ExitCode {
    let (model, _) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    println!(
        "layers={} macros={} d_model={} vocab={} window={} hip_version={} phi={} isa={} rss_kb={}",
        model.n_layers(),
        model.cfg.n_macros,
        model.cfg.d_model,
        model.cfg.vocab,
        model.cfg.window,
        model.hip_version_sum(),
        model.phi_len(),
        cortex_kernels::active_isa(),
        rss_kb()
    );
    ExitCode::SUCCESS
}

fn cmd_sleep(args: &[String]) -> ExitCode {
    let (mut model, model_path) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let steps: usize = flag(args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let eps: f32 = flag(args, "--eps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-4);
    let lr: f32 = flag(args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5e-6);
    let seed: u64 = flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let mut fsm = RuntimeFsm::default();
    match run_sleep_cycle(&mut model, &model_path, steps, eps, lr, seed, &mut fsm) {
        Ok(s) => {
            println!(
                "slept ce {:.6} -> {:.6} wal_eps={} dedup={}+{} snap={:?} fsm={}",
                s.before,
                s.after,
                s.wal_eps,
                s.dedup_w1,
                s.dedup_w2,
                s.snap_ver,
                fsm.state().as_str()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("sleep: {e}");
            ExitCode::from(1)
        }
    }
}

fn cmd_reset(args: &[String]) -> ExitCode {
    let (model, model_path) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let target_ver: Option<u64> = flag(args, "--version").and_then(|s| s.parse().ok());
    match run_reset_mem(&model, &model_path, target_ver) {
        Ok(rb) => {
            println!("hip_version={} rollback={rb:?}", model.hip_version_sum());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("reset-mem: {e}");
            ExitCode::from(1)
        }
    }
}

fn cmd_init_mini(args: &[String]) -> ExitCode {
    let out = match flag(args, "--out") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("missing --out");
            return ExitCode::from(2);
        }
    };
    let seed: u32 = flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let cfg = match flag(args, "--shape") {
        Some("256") => CortexConfig::test_256(),
        Some("micro") => CortexConfig::micro(),
        Some("tiny") => CortexConfig::mezo_tiny(),
        Some("step1") => CortexConfig::step1_stories(),
        _ => CortexConfig::mini(),
    };
    let model = CortexModel::from_seed(cfg, seed);
    if let Err(e) = save_ctxb(&model, &out) {
        eprintln!("{e}");
        return ExitCode::from(1);
    }
    println!("wrote {}", out.display());
    ExitCode::SUCCESS
}

fn cmd_serve(args: &[String]) -> ExitCode {
    let (model, model_path) = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    model.attach_wal_dir(&wal_dir_for(&model_path));
    let bind = flag(args, "--bind")
        .unwrap_or("127.0.0.1:43123")
        .to_string();
    let unix = flag(args, "--unix").map(|s| s.to_string());
    let server = cortex_runtime::api::Server {
        model: Arc::new(RwLock::new(model)),
        model_path,
    };
    if let Some(sock) = unix {
        let server = cortex_runtime::api::Server {
            model: Arc::clone(&server.model),
            model_path: server.model_path.clone(),
        };
        let sock_path = PathBuf::from(sock);
        std::thread::spawn(move || {
            if let Err(e) = cortex_runtime::api::serve_unix(&sock_path, server) {
                eprintln!("unix serve: {e}");
            }
        });
    }
    if let Err(e) = cortex_runtime::api::serve_tcp(&bind, server) {
        eprintln!("tcp serve: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
