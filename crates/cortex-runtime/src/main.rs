//! cortex-cli — local BitSSM + hippocampus runtime.
//!
//! ```text
//! cortex-cli generate --model toy.ctxb --prompt "hi" --max-tokens 32 \
//!     --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 --seed 1
//! cortex-cli logits --model toy.ctxb --ids 1,2,3
//! cortex-cli /status --model toy.ctxb
//! cortex-cli /sleep --model toy.ctxb
//! cortex-cli /reset-mem --model toy.ctxb
//! cortex-cli serve --model toy.ctxb --bind 127.0.0.1:43123 --unix /tmp/cortex.sock
//! ```

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use cortex_runtime::ctxb::{load_ctxb, save_ctxb};
use cortex_runtime::expert::{
    list_packages, load_package, save_package, tiny_adapter, ExpertMeta, ExpertPackage, ROUTER_ABI,
};
use cortex_runtime::generate::{format_tick, next_token, rss_kb, Session};
use cortex_runtime::model::{CortexConfig, CortexModel};
use cortex_runtime::tokenizer::Tokenizer;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::from(2);
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
        "logits" => cmd_logits(&args[1..]),
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
     [--top-p 0.9] [--repetition-penalty 1.1] [--seed 1]\n  \
     encode --vocab MERGES --input TEXT --output TOKENS.u32\n  \
     export-expert --domain NAME --license MIT --d-model N --vocab N --out FILE.cxex\n  \
     install-expert --pack FILE.cxex --dir DIR --d-model N --vocab N\n  \
     list-experts --dir DIR\n  \
     logits --model FILE --ids 1,2,3\n  \
     /status --model FILE\n  \
     /sleep --model FILE\n  \
     /reset-mem --model FILE\n  \
     serve --model FILE [--bind 127.0.0.1:43123] [--unix PATH]\n  \
     init-mini --out FILE\n\n\
Sampling: temperature 0 is greedy; otherwise softmax(logits/T) over the top-p nucleus,\n\
with a repetition penalty on tokens already emitted.\n\
/sleep runs one in-place MeZO step on anchor sequences (no autograd).\n\
/reset-mem restores hippocampus θ to its deterministic init."
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

fn require_model(args: &[String]) -> Result<CortexModel, ExitCode> {
    let path = flag(args, "--model").ok_or_else(|| {
        eprintln!("missing --model");
        ExitCode::from(2)
    })?;
    load_ctxb(std::path::Path::new(path)).map_err(|e| {
        eprintln!("load {path}: {e}");
        ExitCode::from(1)
    })
}

fn cmd_generate(args: &[String]) -> ExitCode {
    let mut model = match require_model(args) {
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
    let tokenizer = match flag(args, "--vocab") {
        Some(path) => match Tokenizer::from_merge_file(std::path::Path::new(path)) {
            Ok(tok) => tok,
            Err(e) => {
                eprintln!("vocab {path}: {e}");
                return ExitCode::from(1);
            }
        },
        None => Tokenizer::byte_level(),
    };
    let mut session = Session::with_tokenizer(
        &mut model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        penalty,
        seed,
        tokenizer,
    );
    let mut n = 0usize;
    while let Some(tick) = next_token(&mut session) {
        println!("{}", format_tick(&tick));
        n += 1;
    }
    println!("tokens={n}");
    println!("continuation={}", session.continuation_text());
    println!("rss_kb={}", rss_kb());
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
    let tok = match Tokenizer::from_merge_file(std::path::Path::new(vocab)) {
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
    if let Err(err) = save_package(std::path::Path::new(out), &pkg) {
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
    let pkg = match load_package(std::path::Path::new(pack)) {
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
    let dest = std::path::Path::new(dir).join(format!("{}.cxex", pkg.meta.domain));
    if let Err(err) = save_package(&dest, &pkg) {
        eprintln!("install: {err}");
        return ExitCode::from(1);
    }
    println!("installed {}", dest.display());
    ExitCode::SUCCESS
}

fn cmd_list_experts(args: &[String]) -> ExitCode {
    let dir = flag(args, "--dir").unwrap_or("experts");
    let found = list_packages(std::path::Path::new(dir));
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
    let model = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let ids = flag(args, "--ids").unwrap_or("1");
    let tokens: Vec<u32> = ids
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mut state = model.new_state();
    for (t, id) in tokens.iter().enumerate() {
        let logits = model.step(*id, &mut state);
        let line: Vec<String> = logits.iter().map(|v| format!("{v:.8}")).collect();
        println!("pos={t} id={id} {}", line.join(" "));
    }
    ExitCode::SUCCESS
}

fn cmd_status(args: &[String]) -> ExitCode {
    let model = match require_model(args) {
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
    let mut model = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let path = flag(args, "--model").unwrap();
    let seqs = vec![vec![1u32, 2, 3, 4], vec![4, 3, 2, 1], vec![1, 1, 2, 2]];
    let (before, after) = model.mezo_steps(&seqs, 1, 0.05, 0.05, 7);
    if let Err(e) = save_ctxb(&model, std::path::Path::new(path)) {
        eprintln!("save after sleep: {e}");
        return ExitCode::from(1);
    }
    println!("slept ce {before:.6} -> {after:.6}");
    ExitCode::SUCCESS
}

fn cmd_reset(args: &[String]) -> ExitCode {
    let model = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    model.reset_memory_to_init();
    let path = flag(args, "--model").unwrap();
    if let Err(e) = save_ctxb(&model, std::path::Path::new(path)) {
        eprintln!("save after reset: {e}");
        return ExitCode::from(1);
    }
    println!("hip_version={}", model.hip_version_sum());
    ExitCode::SUCCESS
}

fn cmd_init_mini(args: &[String]) -> ExitCode {
    let out = match flag(args, "--out") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("missing --out");
            return ExitCode::from(2);
        }
    };
    let cfg = if flag(args, "--shape") == Some("256") {
        CortexConfig::test_256()
    } else if flag(args, "--shape") == Some("tiny") {
        CortexConfig::mezo_tiny()
    } else {
        CortexConfig::mini()
    };
    let model = CortexModel::from_seed(cfg, 1);
    if let Err(e) = save_ctxb(&model, &out) {
        eprintln!("{e}");
        return ExitCode::from(1);
    }
    println!("wrote {}", out.display());
    ExitCode::SUCCESS
}

fn cmd_serve(args: &[String]) -> ExitCode {
    let model = match require_model(args) {
        Ok(m) => m,
        Err(c) => return c,
    };
    let path = flag(args, "--model").unwrap().to_string();
    let bind = flag(args, "--bind")
        .unwrap_or("127.0.0.1:43123")
        .to_string();
    let unix = flag(args, "--unix").map(|s| s.to_string());
    let server = cortex_runtime::api::Server {
        model: std::sync::Arc::new(std::sync::Mutex::new(model)),
        model_path: PathBuf::from(&path),
    };
    if let Some(sock) = unix {
        let server = cortex_runtime::api::Server {
            model: std::sync::Arc::clone(&server.model),
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
