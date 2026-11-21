//! Local HTTP + Unix-socket endpoint (`RwLock`-shared concurrent inference).
//!
//! Routes:
//! * `GET /status`
//! * `POST /sleep`
//! * `POST /reset-mem`
//! * `POST /v1/chat/completions` (OpenAI-style, `stream: true` emits SSE)
//!
//! Concurrent inference requests acquire a shared `RwLock::read()` guard and
//! run with independent `ModelState` + lock-free `SharedWeightArena` hippocampus
//! reads — zero global `Mutex` serialization across chat completions.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use cortex_hippocampus::SnapshotStore;

use crate::generate::{stream_tokens, Session};
use crate::model::CortexModel;

pub struct Server {
    pub model: Arc<RwLock<CortexModel>>,
    pub model_path: PathBuf,
}

pub fn serve_tcp(addr: &str, server: Server) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let server = Arc::new(server);
    eprintln!("cortex-cli listening on http://{addr}");
    for conn in listener.incoming() {
        let stream = conn?;
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            let _ = handle_tcp(stream, &server);
        });
    }
    Ok(())
}

#[cfg(unix)]
pub fn serve_unix(path: &std::path::Path, server: Server) -> std::io::Result<()> {
    use std::os::unix::net::UnixListener;
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    let server = Arc::new(server);
    eprintln!("cortex-cli listening on unix:{}", path.display());
    for conn in listener.incoming() {
        let stream = conn?;
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            let _ = handle_unix(stream, &server);
        });
    }
    Ok(())
}

fn handle_tcp(mut stream: TcpStream, server: &Server) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buf = vec![0u8; 64 * 1024];
    let n = stream.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]).to_string();
    let resp = dispatch(&req, server);
    stream.write_all(resp.as_bytes())?;
    Ok(())
}

#[cfg(unix)]
fn handle_unix(mut stream: std::os::unix::net::UnixStream, server: &Server) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buf = vec![0u8; 64 * 1024];
    let n = stream.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]).to_string();
    let resp = dispatch(&req, server);
    stream.write_all(resp.as_bytes())?;
    Ok(())
}

fn snap_dir_for(model_path: &std::path::Path) -> PathBuf {
    model_path.with_extension("snaps")
}

fn wal_dir_for(model_path: &std::path::Path) -> PathBuf {
    model_path.with_extension("wal")
}

fn dispatch(req: &str, server: &Server) -> String {
    let mut lines = req.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
    if method == "GET" && path == "/status" {
        return text_ok(&status_text(server));
    }
    if method == "POST" && path == "/reset-mem" {
        let model = server.model.read().expect("model read");
        let sdir = snap_dir_for(&server.model_path);
        if let Ok(store) = SnapshotStore::new(&sdir) {
            let vs = store.list_versions();
            if vs.len() >= 2 {
                let prev = vs[vs.len() - 2];
                if model.rollback_hippocampus(&store, prev).is_ok() {
                    return text_ok(&format!("memory rolled back to snapshot v{prev}\n"));
                }
            }
        }
        model.reset_memory_to_init();
        return text_ok("memory reset\n");
    }
    if method == "POST" && path == "/sleep" {
        let mut model = server.model.write().expect("model write");
        let mut replay = cortex_sleep::ReplayBuffer::new();
        let wal_root = wal_dir_for(&server.model_path);
        for m in 0..model.cfg.n_macros {
            let wf = wal_root.join(format!("macro_{m}")).join("episodes.wal");
            if wf.exists() {
                let _ = replay.load_wal(&wf);
            }
        }
        for seq in anchor_seqs(model.cfg.vocab) {
            replay.add_anchor_tokens(seq);
        }
        let seqs = replay.mixed_token_sequences(10);
        let (before, after) = model.mezo_steps(&seqs, 5, 1e-4, 5e-6, 1);
        let sdir = snap_dir_for(&server.model_path);
        let snap_store = SnapshotStore::new(&sdir).ok();
        if let Some(ref store) = snap_store {
            if store.list_versions().is_empty() {
                let pre: Vec<_> = model.mem.iter().map(|m| m.hip.snapshot()).collect();
                let _ = store.save_macros(&pre);
            }
        }
        let (d1, d2, ver) = model
            .consolidate_all_hippocampus(snap_store.as_ref())
            .unwrap_or((0, 0, None));
        if server.model_path.exists() {
            let _ = crate::ctxb::save_ctxb(&model, &server.model_path);
        }
        return text_ok(&format!(
            "slept ce {before:.4} -> {after:.4} wal_eps={} dedup={}+{} snap={:?}\n",
            replay.wal_episodes.len(),
            d1,
            d2,
            ver
        ));
    }
    if method == "POST" && path == "/v1/chat/completions" {
        let prompt = json_string_field(body, "content").unwrap_or_default();
        let tools_accepted = crate::harness::tools_mentioned(body);
        let stream = body.replace(' ', "").contains("\"stream\":true");
        let max_tokens = json_u32_field(body, "max_tokens").unwrap_or(16).min(64);
        // Shared read lock — concurrent sessions run in parallel without blocking!
        let model = server.model.read().expect("model read");
        let mut session = Session::new(&model, &prompt, max_tokens as usize, 0.0, 1.0, 1.0, 1);
        let pieces = stream_tokens(&mut session);
        let text: String = pieces.iter().copied().map(token_text).collect();
        if stream {
            let mut sse = String::new();
            for id in &pieces {
                let piece = json_escape(&token_text(*id));
                sse.push_str(&format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{piece}\"}}}}]}}\n\n"
                ));
            }
            sse.push_str("data: [DONE]\n\n");
            return sse_ok(&sse);
        }
        let json = format!(
            "{{\"object\":\"chat.completion\",\"tools_accepted\":{tools_accepted},\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}}}}]}}",
            json_escape(&text)
        );
        return json_ok(&json);
    }
    text_err("not found")
}

fn status_text(server: &Server) -> String {
    let model = server.model.read().expect("model read");
    format!(
        "layers={} d_model={} vocab={} hip_version={} isa={} model={}\n",
        model.n_layers(),
        model.cfg.d_model,
        model.cfg.vocab,
        model.hip_version_sum(),
        cortex_kernels::active_isa(),
        server.model_path.display()
    )
}

fn anchor_seqs(vocab: usize) -> Vec<Vec<u32>> {
    let cap = vocab.max(2) as u32;
    vec![
        vec![1 % cap, 2 % cap, 3 % cap, 4 % cap],
        vec![4 % cap, 3 % cap, 2 % cap, 1 % cap],
    ]
}

fn token_text(id: u32) -> String {
    if (32..127).contains(&id) {
        char::from_u32(id).unwrap_or('?').to_string()
    } else {
        format!("<{id}>")
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_string_field(body: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let idx = body.find(&pattern)?;
    let rest = &body[idx + pattern.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = rest[1..].chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(n);
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn json_u32_field(body: &str, key: &str) -> Option<u32> {
    let pattern = format!("\"{key}\"");
    let idx = body.find(&pattern)?;
    let rest = &body[idx + pattern.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

fn text_ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn json_ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn sse_ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn text_err(msg: &str) -> String {
    format!(
        "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        msg.len(),
        msg
    )
}
