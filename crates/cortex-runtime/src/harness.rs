//! Tool harness: parse a call, check the schema, run a built-in, inject the response.
//!
//! Built-ins are fixtures (`echo`, `add`, `read_fixture`). `read_fixture` only
//! reads a relative path under a caller-supplied root.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::tokenizer::TOOL_CALL_CLOSE;
use crate::tokenizer::TOOL_CALL_OPEN;

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub required: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolResult {
    Ok(String),
    Err(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarnessStop {
    Answer(String),
    Clarification(String),
    ChitChat(String),
    ErrorRecovered(String),
    StepLimit,
}

#[derive(Clone, Debug)]
pub struct HarnessReport {
    pub stop: HarnessStop,
    pub calls: usize,
    pub errors: usize,
}

pub struct Registry {
    root: PathBuf,
}

impl Registry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        match call.name.as_str() {
            "echo" => match call.arguments.get("text") {
                Some(text) => ToolResult::Ok(text.clone()),
                None => ToolResult::Err("echo requires text".into()),
            },
            "add" => {
                let a = call.arguments.get("a").and_then(|s| s.parse::<i64>().ok());
                let b = call.arguments.get("b").and_then(|s| s.parse::<i64>().ok());
                match (a, b) {
                    (Some(a), Some(b)) => ToolResult::Ok((a + b).to_string()),
                    _ => ToolResult::Err("add requires integer a and b".into()),
                }
            }
            "read_fixture" => {
                let Some(name) = call.arguments.get("name") else {
                    return ToolResult::Err("read_fixture requires name".into());
                };
                match safe_join(&self.root, name) {
                    Ok(path) => match fs::read_to_string(&path) {
                        Ok(text) => ToolResult::Ok(text),
                        Err(err) => ToolResult::Err(err.to_string()),
                    },
                    Err(err) => ToolResult::Err(err),
                }
            }
            other => ToolResult::Err(format!("unknown tool {other}")),
        }
    }
}

fn safe_join(root: &Path, name: &str) -> Result<PathBuf, String> {
    if name.is_empty() || name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err("fixture name is not a single relative path".into());
    }
    Ok(root.join(name))
}

pub fn validate(spec: &ToolSpec, call: &ToolCall) -> Result<(), String> {
    if spec.name != call.name {
        return Err(format!(
            "schema is for {}, call is {}",
            spec.name, call.name
        ));
    }
    for key in &spec.required {
        if !call.arguments.contains_key(key) {
            return Err(format!("missing required {key}"));
        }
    }
    Ok(())
}

pub fn parse_tool_call(text: &str) -> Result<ToolCall, String> {
    let Some(start) = text.find(TOOL_CALL_OPEN) else {
        return Err("no tool_call".into());
    };
    let rest = &text[start + TOOL_CALL_OPEN.len()..];
    let Some(end) = rest.find(TOOL_CALL_CLOSE) else {
        return Err("unclosed tool_call".into());
    };
    parse_call_json(rest[..end].trim())
}

fn parse_call_json(input: &str) -> Result<ToolCall, String> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        i: 0,
    };
    p.skip();
    let obj = p.object()?;
    let name = obj
        .get("name")
        .cloned()
        .ok_or_else(|| "missing name".to_string())?;
    let mut arguments = BTreeMap::new();
    if let Some(raw) = obj.get("arguments") {
        if raw.starts_with('{') {
            let mut q = Parser {
                bytes: raw.as_bytes(),
                i: 0,
            };
            arguments = q.object()?;
        }
    }
    Ok(ToolCall { name, arguments })
}

struct Parser<'a> {
    bytes: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn skip(&mut self) {
        while self.i < self.bytes.len() && self.bytes[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn object(&mut self) -> Result<BTreeMap<String, String>, String> {
        self.skip();
        self.eat(b'{')?;
        let mut map = BTreeMap::new();
        loop {
            self.skip();
            if self.peek() == Some(b'}') {
                self.i += 1;
                break;
            }
            let key = self.string()?;
            self.skip();
            self.eat(b':')?;
            self.skip();
            let value = self.value()?;
            map.insert(key, value);
            self.skip();
            if self.peek() == Some(b',') {
                self.i += 1;
                continue;
            }
            if self.peek() == Some(b'}') {
                self.i += 1;
                break;
            }
            return Err("expected comma or end of object".into());
        }
        Ok(map)
    }

    fn value(&mut self) -> Result<String, String> {
        self.skip();
        match self.peek() {
            Some(b'"') => self.string(),
            Some(b'{') => {
                let start = self.i;
                let mut depth = 0i32;
                while self.i < self.bytes.len() {
                    match self.bytes[self.i] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            self.i += 1;
                            if depth == 0 {
                                break;
                            }
                            continue;
                        }
                        _ => {}
                    }
                    self.i += 1;
                }
                Ok(String::from_utf8_lossy(&self.bytes[start..self.i]).into_owned())
            }
            Some(b) if b.is_ascii_digit() || b == b'-' => {
                let start = self.i;
                while self.i < self.bytes.len()
                    && (self.bytes[self.i].is_ascii_digit()
                        || matches!(self.bytes[self.i], b'.' | b'-'))
                {
                    self.i += 1;
                }
                Ok(String::from_utf8_lossy(&self.bytes[start..self.i]).into_owned())
            }
            _ => Err("unsupported json value".into()),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        while self.i < self.bytes.len() {
            let b = self.bytes[self.i];
            self.i += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let n = self.bytes.get(self.i).copied().ok_or("bad escape")?;
                    self.i += 1;
                    out.push(n as char);
                }
                _ => out.push(b as char),
            }
        }
        Err("unterminated string".into())
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.i).copied()
    }

    fn eat(&mut self, byte: u8) -> Result<(), String> {
        self.skip();
        if self.peek() == Some(byte) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {}", byte as char))
        }
    }
}

/// One scripted assistant turn. The transcript includes prior tool responses.
pub fn run_scripted(
    tools: &[ToolSpec],
    registry: &Registry,
    mut assistant: impl FnMut(&str) -> String,
    max_steps: usize,
) -> HarnessReport {
    let mut transcript = String::new();
    let mut calls = 0usize;
    let mut errors = 0usize;
    for _ in 0..max_steps {
        let reply = assistant(&transcript);
        if let Ok(call) = parse_tool_call(&reply) {
            calls += 1;
            let spec = tools.iter().find(|t| t.name == call.name);
            let result = match spec {
                None => ToolResult::Err(format!("tool {} is not advertised", call.name)),
                Some(spec) => match validate(spec, &call) {
                    Ok(()) => registry.execute(&call),
                    Err(err) => ToolResult::Err(err),
                },
            };
            let body = match &result {
                ToolResult::Ok(text) => text.clone(),
                ToolResult::Err(err) => {
                    errors += 1;
                    format!("error: {err}")
                }
            };
            transcript.push_str(&reply);
            transcript.push_str("<|tool_response|>");
            transcript.push_str(&body);
            transcript.push_str("</|tool_response|>");
            continue;
        }
        if calls == 0 && looks_like_question(&reply) {
            return HarnessReport {
                stop: HarnessStop::Clarification(reply),
                calls,
                errors,
            };
        }
        if calls == 0 {
            return HarnessReport {
                stop: HarnessStop::ChitChat(reply),
                calls,
                errors,
            };
        }
        if errors > 0 {
            return HarnessReport {
                stop: HarnessStop::ErrorRecovered(reply),
                calls,
                errors,
            };
        }
        return HarnessReport {
            stop: HarnessStop::Answer(reply),
            calls,
            errors,
        };
    }
    HarnessReport {
        stop: HarnessStop::StepLimit,
        calls,
        errors,
    }
}

fn looks_like_question(text: &str) -> bool {
    let t = text.trim();
    t.ends_with('?') || t.ends_with('？') || t.to_ascii_lowercase().contains("which")
}

pub fn load_tool_file(text: &str) -> Vec<ToolSpec> {
    // One tool per line: `name req1,req2`
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let mut parts = line.split_whitespace();
            let name = parts.next()?.to_string();
            let required = parts
                .next()
                .unwrap_or("")
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            Some(ToolSpec { name, required })
        })
        .collect()
}

pub fn tools_mentioned(body: &str) -> usize {
    let Some(at) = body.find("\"tools\"") else {
        return 0;
    };
    body[at..].matches("\"name\"").count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_spec() -> ToolSpec {
        ToolSpec {
            name: "add".into(),
            required: vec!["a".into(), "b".into()],
        }
    }

    #[test]
    fn success_path_runs_add() {
        let dir = std::env::temp_dir();
        let registry = Registry::new(&dir);
        let tools = vec![add_spec()];
        let mut turn = 0;
        let report = run_scripted(
            &tools,
            &registry,
            |_| {
                turn += 1;
                if turn == 1 {
                    "<|tool_call|>{\"name\":\"add\",\"arguments\":{\"a\":2,\"b\":3}}</|tool_call|>"
                        .into()
                } else {
                    "sum is 5".into()
                }
            },
            4,
        );
        assert_eq!(report.calls, 1);
        assert_eq!(report.errors, 0);
        assert_eq!(report.stop, HarnessStop::Answer("sum is 5".into()));
    }

    #[test]
    fn error_recovery_path() {
        let registry = Registry::new(std::env::temp_dir());
        let tools = vec![add_spec()];
        let mut turn = 0;
        let report = run_scripted(
            &tools,
            &registry,
            |_| {
                turn += 1;
                if turn == 1 {
                    "<|tool_call|>{\"name\":\"add\",\"arguments\":{\"a\":2}}</|tool_call|>".into()
                } else {
                    "I still need b. Using add with 2 and 0 is not valid, so I will ask later."
                        .into()
                }
            },
            4,
        );
        assert_eq!(report.calls, 1);
        assert_eq!(report.errors, 1);
        assert!(matches!(report.stop, HarnessStop::ErrorRecovered(_)));
    }

    #[test]
    fn clarification_does_not_call() {
        let registry = Registry::new(std::env::temp_dir());
        let tools = vec![add_spec()];
        let report = run_scripted(
            &tools,
            &registry,
            |_| "Which two integers should I add?".into(),
            3,
        );
        assert_eq!(report.calls, 0);
        assert!(matches!(report.stop, HarnessStop::Clarification(_)));
    }

    #[test]
    fn chit_chat_with_tools_available_does_not_call() {
        let registry = Registry::new(std::env::temp_dir());
        let tools = vec![add_spec()];
        let report = run_scripted(&tools, &registry, |_| "Hello. Nice to meet you.".into(), 3);
        assert_eq!(report.calls, 0);
        assert!(matches!(report.stop, HarnessStop::ChitChat(_)));
    }

    #[test]
    fn read_fixture_stays_in_root() {
        let root = std::env::temp_dir().join(format!("cortex-fix-{}", std::process::id()));
        let _ = fs::create_dir_all(&root);
        fs::write(root.join("note.txt"), "hello fixture").unwrap();
        let registry = Registry::new(&root);
        let ok = registry.execute(&ToolCall {
            name: "read_fixture".into(),
            arguments: BTreeMap::from([("name".into(), "note.txt".into())]),
        });
        assert_eq!(ok, ToolResult::Ok("hello fixture".into()));
        let bad = registry.execute(&ToolCall {
            name: "read_fixture".into(),
            arguments: BTreeMap::from([("name".into(), "../note.txt".into())]),
        });
        assert!(matches!(bad, ToolResult::Err(_)));
        let _ = fs::remove_dir_all(&root);
    }
}
