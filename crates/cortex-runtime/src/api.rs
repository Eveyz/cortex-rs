//! Local HTTP + Unix-socket endpoint.
//!
//! Routes:
//! * `GET /status`
//! * `POST /sleep`
//! * `POST /reset-mem`
//! * `POST /v1/chat/completions` (OpenAI-style, `stream: true` emits SSE)
//!
//! No cloud client. The handler is std-only so the binary does not pull an
//! async runtime.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::generate::{stream_tokens, Session};
use crate::model::CortexModel;

pub struct Server {
    pub model: Arc<Mutex<CortexModel>>,
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
        server.model.lock().expect("model").reset_memory_to_init();
        return text_ok("memory reset\n");
    }
    if method == "POST" && path == "/sleep" {
        let mut model = server.model.lock().expect("model");
        let seqs = anchor_seqs();
        let (before, after) = model.mezo_steps(&seqs, 1, 0.05, 0.1, 1);
        return text_ok(&format!("slept ce {before:.4} -> {after:.4}\n"));
    }
    if method == "POST" && path == "/v1/chat/completions" {
        let prompt = json_string_field(body, "content").unwrap_or_default();
        let stream = body.replace(' ', "").contains("\"stream\":true");
        let max_tokens = json_u32_field(body, "max_tokens").unwrap_or(16).min(64);
        let mut model = server.model.lock().expect("model");
        let mut session = Session::new(&mut model, &prompt, max_tokens as usize, 0.0, 1.0, 1.0, 1);
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
            "{{\"object\":\"chat.completion\",\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}}}}]}}",
            json_escape(&text)
        );
        return json_ok(&json);
    }
    text_err("not found")
}

fn status_text(server: &Server) -> String {
    let model = server.model.lock().expect("model");
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

fn anchor_seqs() -> Vec<Vec<u32>> {
    vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1]]
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
