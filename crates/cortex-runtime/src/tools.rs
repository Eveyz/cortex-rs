//! Tool calling + agent harness: the "hands" of Cortex-Local.
//!
//! The model on its own only generates text. This module gives it a
//! way to act: a small registry of tools, a parser for the
//! `<tool_call>` blocks the model emits, and an `AgentLoop` that
//! runs generate -> parse -> execute -> feed the result back until
//! the model answers directly or runs out of steps.
//!
//! # Contract with the Python distillation data
//!
//! The tool set, the system prompt rendered by
//! [`ToolRegistry::render_system_prompt`], and the trace format
//!
//! ```text
//! <tool_call>{"name": "...", "arguments": {...}}</tool_call>
//! <tool_result name="...">...</tool_result>
//! ```
//!
//! are shared verbatim with `tools/distill_generate.py` (category
//! `tooluse`). The test `test_render_system_prompt_exact` pins the
//! prompt text so the two sides cannot drift apart silently. If you
//! change the wording here, change the Python side to match, and
//! vice versa.
//!
//! # Safety
//!
//! File tools are jailed to a workspace directory: absolute paths
//! and `..` escapes are rejected, and symlinks that point outside
//! are resolved and re-checked. There is deliberately no shell tool.
//! The harness runs on the user's own machine, on their own files,
//! so the workspace default (`./cortex-work`) is a fresh directory
//! rather than the current one.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Minimal JSON (std only: no serde in this workspace)
// ---------------------------------------------------------------------------

/// A JSON value. Objects keep insertion order (a `Vec` of pairs).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonValue>),
    Obj(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        self.skip_ws();
        match self.next() {
            Some(b) if b == c => Ok(()),
            other => Err(format!(
                "expected '{}', found {:?}",
                c as char,
                other.map(|b| b as char)
            )),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_obj(),
            Some(b'[') => self.parse_arr(),
            Some(b'"') => Ok(JsonValue::Str(self.parse_string()?)),
            Some(b't') => self.parse_lit("true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_lit("false", JsonValue::Bool(false)),
            Some(b'n') => self.parse_lit("null", JsonValue::Null),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_num(),
            other => Err(format!(
                "unexpected character {:?}",
                other.map(|b| b as char)
            )),
        }
    }

    fn parse_lit(&mut self, lit: &str, value: JsonValue) -> Result<JsonValue, String> {
        for &b in lit.as_bytes() {
            match self.next() {
                Some(x) if x == b => {}
                _ => return Err(format!("invalid literal, expected {lit}")),
            }
        }
        Ok(value)
    }

    fn parse_num(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(
            self.peek(),
            Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
        ) {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number".to_string())?;
        text.parse::<f64>()
            .map(JsonValue::Num)
            .map_err(|_| format!("invalid number: {text}"))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.next() {
                None => return Err("unterminated string".to_string()),
                Some(b'"') => break,
                Some(b'\\') => match self.next() {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'b') => out.push('\u{0008}'),
                    Some(b'f') => out.push('\u{000C}'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'u') => {
                        let hi = self.parse_hex4()?;
                        if (0xD800..0xDC00).contains(&hi) {
                            // High surrogate: expect a low surrogate.
                            if self.next() == Some(b'\\') && self.next() == Some(b'u') {
                                let lo = self.parse_hex4()?;
                                let c = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(c).ok_or("invalid surrogate pair")?);
                            } else {
                                return Err("lone high surrogate".to_string());
                            }
                        } else {
                            out.push(char::from_u32(hi).ok_or("invalid \\u escape")?);
                        }
                    }
                    Some(other) => {
                        return Err(format!("invalid escape: \\{}", other as char));
                    }
                    None => return Err("unterminated escape".to_string()),
                },
                Some(b) if b < 0x20 => {
                    return Err("unescaped control character in string".to_string());
                }
                Some(b) => {
                    // UTF-8 continuation bytes are consumed as part of
                    // the raw byte sequence; collect the full char.
                    let len = utf8_len(b);
                    let mut buf = vec![b];
                    for _ in 1..len {
                        match self.next() {
                            Some(c) => buf.push(c),
                            None => return Err("unterminated string".to_string()),
                        }
                    }
                    out.push_str(
                        std::str::from_utf8(&buf)
                            .map_err(|_| "invalid UTF-8 in string".to_string())?,
                    );
                }
            }
        }
        Ok(out)
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let b = self.next().ok_or("truncated \\u escape")?;
            let d = (b as char)
                .to_digit(16)
                .ok_or("invalid hex digit in \\u escape")?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn parse_obj(&mut self) -> Result<JsonValue, String> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(b':')?;
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => break,
                other => {
                    return Err(format!(
                        "expected ',' or '}}' in object, found {:?}",
                        other.map(|b| b as char)
                    ));
                }
            }
        }
        Ok(JsonValue::Obj(pairs))
    }

    fn parse_arr(&mut self) -> Result<JsonValue, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => break,
                other => {
                    return Err(format!(
                        "expected ',' or ']' in array, found {:?}",
                        other.map(|b| b as char)
                    ));
                }
            }
        }
        Ok(JsonValue::Arr(items))
    }
}

/// Leading-byte length of a UTF-8 sequence (1 for ASCII).
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

pub fn parse_json(s: &str) -> Result<JsonValue, String> {
    let mut p = JsonParser::new(s);
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err("trailing characters after JSON value".to_string());
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Tool calls: parsing
// ---------------------------------------------------------------------------

/// One parsed `<tool_call>` block.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: JsonValue,
}

/// Scan `text` for `<tool_call>...</tool_call>` blocks. Returns the
/// calls that parsed plus one error string per malformed block (the
/// loop feeds parse errors back to the model as a tool result, so it
/// gets a chance to fix the format).
pub fn parse_tool_calls(text: &str) -> (Vec<ToolCall>, Vec<String>) {
    let normalized = text
        .replace("<|tool_call|>", "<tool_call>")
        .replace("</|tool_call|>", "</tool_call>");
    let mut calls = Vec::new();
    let mut errors = Vec::new();
    let mut rest = normalized.as_str();
    while let Some(start) = rest.find("<tool_call>") {
        let inner_start = start + "<tool_call>".len();
        let Some(end) = rest[inner_start..].find("</tool_call>") else {
            errors.push("unterminated <tool_call> block".to_string());
            break;
        };
        let inner = &rest[inner_start..inner_start + end];
        match parse_json(inner) {
            Ok(v @ JsonValue::Obj(_)) => match v.get("name").and_then(JsonValue::as_str) {
                Some(name) => {
                    let args = match v.get("arguments") {
                        Some(a) => a.clone(),
                        None => JsonValue::Obj(Vec::new()),
                    };
                    if matches!(args, JsonValue::Obj(_)) {
                        calls.push(ToolCall {
                            name: name.to_string(),
                            arguments: args,
                        });
                    } else {
                        errors.push("tool call \"arguments\" is not an object".to_string());
                    }
                }
                None => errors.push("tool call has no string \"name\"".to_string()),
            },
            Ok(_) => errors.push("tool call is not a JSON object".to_string()),
            Err(e) => errors.push(format!("tool call JSON parse error: {e}")),
        }
        rest = &rest[inner_start + end + "</tool_call>".len()..];
    }
    (calls, errors)
}

/// Text up to and including the last `</tool_call>` or `</|tool_call|>`;
/// the whole input when there is no complete call block.
fn truncate_after_last_tool_call(text: &str) -> &str {
    let a = text.rfind("</tool_call>").map(|e| e + "</tool_call>".len());
    let b = text
        .rfind("</|tool_call|>")
        .map(|e| e + "</|tool_call|>".len());
    match (a, b) {
        (Some(x), Some(y)) => &text[..x.max(y)],
        (Some(x), None) => &text[..x],
        (None, Some(y)) => &text[..y],
        (None, None) => text,
    }
}

/// Remove all `<tool_result ...>...</tool_result>` blocks (used to
/// clean a final answer that hallucinated tool results).
fn strip_tool_results(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<tool_result") {
        out.push_str(&rest[..start]);
        let after_tag = &rest[start..];
        match after_tag.find("</tool_result>") {
            Some(end) => rest = &after_tag[end + "</tool_result>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// A tool the model can call. Object-safe so the registry can hold
/// `Box<dyn Tool>`.
pub trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// The parameter schema as shown in the system prompt, e.g.
    /// `{"expression": "string"}`.
    fn parameters(&self) -> &str;
    fn required(&self) -> &[&str];
    fn execute(&self, args: &JsonValue, workspace: &Path) -> Result<String, String>;
}

/// Jail `raw` (a model-supplied path) inside `workspace`: absolute
/// paths and `..` escapes are rejected, and symlinks that point
/// outside are resolved and re-checked.
fn jail(workspace: &Path, raw: &str) -> Result<PathBuf, String> {
    let ws = workspace
        .canonicalize()
        .map_err(|e| format!("workspace not accessible: {e}"))?;
    let mut out = ws.clone();
    for comp in Path::new(raw).components() {
        match comp {
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("absolute paths not allowed: {raw}"));
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if out == ws {
                    return Err(format!("path escapes workspace: {raw}"));
                }
                out.pop();
            }
            Component::Normal(c) => out.push(c),
        }
    }
    if out.exists() {
        let real = out
            .canonicalize()
            .map_err(|e| format!("cannot resolve path: {e}"))?;
        if real != ws && !real.starts_with(&ws) {
            return Err(format!("path escapes workspace: {raw}"));
        }
        return Ok(real);
    }
    Ok(out)
}

// -- calculator ------------------------------------------------------------

fn eval_expr(expr: &str) -> Result<f64, String> {
    let tokens = tokenize(expr)?;
    let mut p = ExprParser { tokens, pos: 0 };
    let v = p.parse_expr()?;
    if p.pos != p.tokens.len() {
        return Err(format!("unexpected trailing input in {expr:?}"));
    }
    Ok(v)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    LParen,
    RParen,
}

fn tokenize(expr: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '+' => {
                tokens.push(Token::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(Token::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Star);
                chars.next();
            }
            '/' => {
                tokens.push(Token::Slash);
                chars.next();
            }
            '%' => {
                tokens.push(Token::Percent);
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Num(
                    num.parse::<f64>()
                        .map_err(|_| format!("invalid number: {num}"))?,
                ));
            }
            other => return Err(format!("unexpected character: {other:?}")),
        }
    }
    Ok(tokens)
}

struct ExprParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl ExprParser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut v = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.pos += 1;
                    v += self.parse_term()?;
                }
                Some(Token::Minus) => {
                    self.pos += 1;
                    v -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut v = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.pos += 1;
                    v *= self.parse_factor()?;
                }
                Some(Token::Slash) => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    if rhs == 0.0 {
                        return Err("division by zero".to_string());
                    }
                    v /= rhs;
                }
                Some(Token::Percent) => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    if rhs == 0.0 {
                        return Err("modulo by zero".to_string());
                    }
                    // Truncated remainder, matching Python math.fmod
                    // (the generator side uses the same rule).
                    v %= rhs;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    fn parse_factor(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some(Token::Num(n)) => {
                let n = *n;
                self.pos += 1;
                Ok(n)
            }
            Some(Token::LParen) => {
                self.pos += 1;
                let v = self.parse_expr()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.pos += 1;
                        Ok(v)
                    }
                    _ => Err("missing closing parenthesis".to_string()),
                }
            }
            Some(Token::Minus) => {
                self.pos += 1;
                Ok(-self.parse_factor()?)
            }
            Some(t) => Err(format!("unexpected token: {t:?}")),
            None => Err("unexpected end of expression".to_string()),
        }
    }
}

/// Integral results print as integers (42, not 42.0) -- the same
/// rule as the Python generator, so traces stay consistent.
fn format_number(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }
    fn description(&self) -> &str {
        "Evaluate an arithmetic expression (+, -, *, /, %, parentheses)."
    }
    fn parameters(&self) -> &str {
        "{\"expression\": \"string\"}"
    }
    fn required(&self) -> &[&str] {
        &["expression"]
    }
    fn execute(&self, args: &JsonValue, _workspace: &Path) -> Result<String, String> {
        let expr = args
            .get("expression")
            .and_then(JsonValue::as_str)
            .ok_or("calculator: \"expression\" must be a string")?;
        Ok(format_number(eval_expr(expr)?))
    }
}

// -- file tools ------------------------------------------------------------

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> &str {
        "Read a text file inside the workspace."
    }
    fn parameters(&self) -> &str {
        "{\"path\": \"string\"}"
    }
    fn required(&self) -> &[&str] {
        &["path"]
    }
    fn execute(&self, args: &JsonValue, workspace: &Path) -> Result<String, String> {
        let path = args
            .get("path")
            .and_then(JsonValue::as_str)
            .ok_or("read_file: \"path\" must be a string")?;
        let target = jail(workspace, path)?;
        let text = fs::read_to_string(&target).map_err(|e| format!("read error: {e}"))?;
        Ok(text.chars().take(4000).collect())
    }
}

pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }
    fn description(&self) -> &str {
        "Write text to a file inside the workspace."
    }
    fn parameters(&self) -> &str {
        "{\"path\": \"string\", \"content\": \"string\"}"
    }
    fn required(&self) -> &[&str] {
        &["path", "content"]
    }
    fn execute(&self, args: &JsonValue, workspace: &Path) -> Result<String, String> {
        let path = args
            .get("path")
            .and_then(JsonValue::as_str)
            .ok_or("write_file: \"path\" must be a string")?;
        let content = args
            .get("content")
            .and_then(JsonValue::as_str)
            .ok_or("write_file: \"content\" must be a string")?;
        let target = jail(workspace, path)?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("write error: {e}"))?;
        }
        fs::write(&target, content).map_err(|e| format!("write error: {e}"))?;
        Ok(format!("wrote {} ({} bytes)", path, content.len()))
    }
}

pub struct ListDirTool;

impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }
    fn description(&self) -> &str {
        "List files in a directory inside the workspace."
    }
    fn parameters(&self) -> &str {
        "{\"path\": \"string\"}"
    }
    fn required(&self) -> &[&str] {
        &["path"]
    }
    fn execute(&self, args: &JsonValue, workspace: &Path) -> Result<String, String> {
        let path = args
            .get("path")
            .and_then(JsonValue::as_str)
            .ok_or("list_dir: \"path\" must be a string")?;
        let target = jail(workspace, path)?;
        let mut names = Vec::new();
        for entry in fs::read_dir(&target).map_err(|e| format!("list error: {e}"))? {
            let entry = entry.map_err(|e| format!("list error: {e}"))?;
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        if names.is_empty() {
            Ok("(empty)".to_string())
        } else {
            Ok(names.join("\n"))
        }
    }
}

// -- time ------------------------------------------------------------------

/// UTC date from days since the Unix epoch (Howard Hinnant's
/// civil-from-days algorithm). Std has no date formatting; this
/// keeps the time tool dependency-free and identical to the Python
/// side (which also reports UTC).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn utc_now_string() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        tod / 3600,
        (tod % 3600) / 60
    )
}

pub struct GetTimeTool;

impl Tool for GetTimeTool {
    fn name(&self) -> &str {
        "get_time"
    }
    fn description(&self) -> &str {
        "Get the current date and time (UTC)."
    }
    fn parameters(&self) -> &str {
        "{}"
    }
    fn required(&self) -> &[&str] {
        &[]
    }
    fn execute(&self, _args: &JsonValue, _workspace: &Path) -> Result<String, String> {
        Ok(utc_now_string())
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: vec![
                Box::new(CalculatorTool),
                Box::new(ReadFileTool),
                Box::new(WriteFileTool),
                Box::new(ListDirTool),
                Box::new(GetTimeTool),
            ],
        }
    }

    pub fn with_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    /// The system prompt, rendered from the registered tools. This
    /// text must match `render_tool_system_prompt()` in
    /// `tools/distill_generate.py` exactly (see the pinned test).
    pub fn render_system_prompt(&self) -> String {
        let mut s = String::from(
            "You are Cortex, a helpful assistant with access to tools.\nAvailable tools:\n",
        );
        for t in &self.tools {
            s.push_str(&format!(
                "- {}: {} Parameters: {}\n",
                t.name(),
                t.description(),
                t.parameters()
            ));
        }
        s.push_str("When you need a tool, emit exactly one block and nothing else after it:\n");
        s.push_str("<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n");
        s.push_str("The tool result will appear as <tool_result name=\"tool_name\">...</tool_result>. Then continue. When you can answer directly, answer without a tool call.");
        s
    }

    /// Validate a parsed call against the tool's schema and run it.
    /// Tool errors are returned as `Err` and the loop feeds them back
    /// to the model as a `<tool_result>` so it can self-correct.
    pub fn execute(&self, call: &ToolCall, workspace: &Path) -> Result<String, String> {
        let tool = self
            .find(&call.name)
            .ok_or_else(|| format!("unknown tool: {}", call.name))?;
        if !matches!(call.arguments, JsonValue::Obj(_)) {
            return Err("tool arguments must be a JSON object".to_string());
        }
        for req in tool.required() {
            if call.arguments.get(req).is_none() {
                return Err(format!("missing required parameter: {req}"));
            }
        }
        tool.execute(&call.arguments, workspace)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

pub struct AgentOutcome {
    pub transcript: String,
    pub answer: String,
    /// Tool rounds executed (one round may contain several calls).
    pub steps: usize,
    pub tool_calls: usize,
    /// True when the model answered without a tool call.
    pub finished: bool,
}

/// Runs the generate -> parse -> execute loop around any text
/// generator. `generate` is a closure over the real model (see
/// `cortex-cli --agent`); tests inject a scripted generator.
///
/// Transcript shape (identical to the distillation traces):
///
/// ```text
/// <system prompt with tools>
///
/// User: <request>
/// Assistant: <model output with <tool_call> blocks>
/// <tool_result name="...">...</tool_result>
/// Assistant: <continuation / final answer>
/// ```
///
/// After each generation, anything the model wrote after the last
/// `</tool_call>` is discarded -- that is where hallucinated tool
/// results would live; only real executed results enter the
/// transcript.
pub struct AgentLoop {
    pub registry: ToolRegistry,
    pub workspace: PathBuf,
    pub max_steps: usize,
}

impl AgentLoop {
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            registry: ToolRegistry::new(),
            workspace,
            max_steps: 5,
        }
    }

    pub fn system_prompt(&self) -> String {
        self.registry.render_system_prompt()
    }

    pub fn run(&self, request: &str, generate: &mut dyn FnMut(&str) -> String) -> AgentOutcome {
        let mut transcript = format!("{}\n\nUser: {}\nAssistant:", self.system_prompt(), request);
        let mut steps = 0usize;
        let mut calls_total = 0usize;
        loop {
            let out = generate(&transcript);
            let (calls, errors) = parse_tool_calls(&out);
            if calls.is_empty() && errors.is_empty() {
                transcript.push_str(&out);
                let answer = strip_tool_results(&out).trim().to_string();
                return AgentOutcome {
                    transcript,
                    answer,
                    steps,
                    tool_calls: calls_total,
                    finished: true,
                };
            }
            transcript.push_str(truncate_after_last_tool_call(&out));
            if steps >= self.max_steps {
                let answer = strip_tool_results(&out).trim().to_string();
                return AgentOutcome {
                    transcript,
                    answer,
                    steps,
                    tool_calls: calls_total,
                    finished: false,
                };
            }
            for call in &calls {
                match self.registry.execute(call, &self.workspace) {
                    Ok(res) => transcript.push_str(&format!(
                        "\n<tool_result name=\"{}\">{}</tool_result>",
                        call.name, res
                    )),
                    Err(e) => transcript.push_str(&format!(
                        "\n<tool_result name=\"{}\">error: {}</tool_result>",
                        call.name, e
                    )),
                }
                calls_total += 1;
            }
            for e in &errors {
                transcript.push_str(&format!(
                    "\n<tool_result name=\"parse_error\">{e}</tool_result>"
                ));
            }
            steps += 1;
            transcript.push_str("\nAssistant:");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_workspace() -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("cortex-tools-test-{}-{n}", std::process::id()));
        fs::create_dir_all(&dir).expect("create test workspace");
        dir
    }

    #[test]
    fn test_json_parser() {
        let v = parse_json(
            r#"{"name": "calculator", "arguments": {"expression": "12*3+5"}, "n": 1.5, "ok": true, "x": null, "arr": [1, "a"]}"#,
        )
        .expect("parse");
        assert_eq!(
            v.get("name").and_then(JsonValue::as_str),
            Some("calculator")
        );
        assert_eq!(
            v.get("arguments")
                .and_then(|a| a.get("expression"))
                .and_then(JsonValue::as_str),
            Some("12*3+5")
        );
        assert_eq!(v.get("n"), Some(&JsonValue::Num(1.5)));
        assert_eq!(v.get("ok"), Some(&JsonValue::Bool(true)));
        assert_eq!(v.get("x"), Some(&JsonValue::Null));
        assert_eq!(
            v.get("arr"),
            Some(&JsonValue::Arr(vec![
                JsonValue::Num(1.0),
                JsonValue::Str("a".to_string())
            ]))
        );
        // Escapes and unicode.
        let s = parse_json(r#""a\nb\u4e2d""#).expect("parse string");
        assert_eq!(s.as_str(), Some("a\nb中"));
        // Errors.
        assert!(parse_json("{").is_err());
        assert!(parse_json("{\"a\": }").is_err());
        assert!(parse_json("{\"a\": 1} trailing").is_err());
        assert!(parse_json("").is_err());
    }

    #[test]
    fn test_calculator() {
        assert_eq!(eval_expr("12*3+5").unwrap(), 41.0);
        assert_eq!(eval_expr("(2+3)*4").unwrap(), 20.0);
        assert_eq!(eval_expr("10%3").unwrap(), 1.0);
        assert_eq!(eval_expr("7/2").unwrap(), 3.5);
        assert_eq!(eval_expr("-5+3").unwrap(), -2.0);
        assert_eq!(eval_expr("2.5*4").unwrap(), 10.0);
        assert!(eval_expr("1/0").is_err());
        assert!(eval_expr("1%0").is_err());
        assert!(eval_expr("2**3").is_err());
        assert!(eval_expr("").is_err());
        assert!(eval_expr("1+").is_err());
        assert!(eval_expr("(1+2").is_err());
        assert_eq!(format_number(41.0), "41");
        assert_eq!(format_number(3.5), "3.5");
        assert_eq!(format_number(-2.0), "-2");
    }

    #[test]
    fn test_jail() {
        let ws = test_workspace();
        let _guard = TestDirGuard(ws.clone());
        assert!(jail(&ws, "notes/a.txt").is_ok());
        assert!(jail(&ws, "a/../b.txt").is_ok());
        assert!(jail(&ws, "../escape.txt").is_err());
        assert!(jail(&ws, "/etc/passwd").is_err());
        assert!(jail(&ws, "a/../../escape.txt").is_err());
        // Symlink pointing outside is caught when it exists.
        #[cfg(unix)]
        {
            let outside = test_workspace();
            let _guard2 = TestDirGuard(outside.clone());
            let target = outside.join("secret.txt");
            fs::write(&target, "secret").unwrap();
            let link = ws.join("link.txt");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert!(jail(&ws, "link.txt").is_err());
        }
    }

    struct TestDirGuard(PathBuf);
    impl Drop for TestDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_parse_tool_calls() {
        let text = "Let me compute.\n<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"12*3+5\"}}</tool_call>\nand also\n<tool_call>{\"name\": \"get_time\", \"arguments\": {}}</tool_call>";
        let (calls, errors) = parse_tool_calls(text);
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "calculator");
        assert_eq!(
            calls[0]
                .arguments
                .get("expression")
                .and_then(JsonValue::as_str),
            Some("12*3+5")
        );
        assert_eq!(calls[1].name, "get_time");
        // Malformed JSON -> error, no call.
        let (calls, errors) = parse_tool_calls("<tool_call>{not json</tool_call>");
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
        // Unterminated block -> error.
        let (calls, errors) = parse_tool_calls("<tool_call>{\"name\": \"x\"");
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
        // Missing name -> error.
        let (calls, errors) = parse_tool_calls("<tool_call>{\"arguments\": {}}</tool_call>");
        assert!(calls.is_empty());
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_render_system_prompt_exact() {
        // Pinned against tools/distill_generate.py's
        // render_tool_system_prompt(). If this fails, one side
        // changed the wording without the other.
        let expected = [
            "You are Cortex, a helpful assistant with access to tools.",
            "Available tools:",
            "- calculator: Evaluate an arithmetic expression (+, -, *, /, %, parentheses). Parameters: {\"expression\": \"string\"}",
            "- read_file: Read a text file inside the workspace. Parameters: {\"path\": \"string\"}",
            "- write_file: Write text to a file inside the workspace. Parameters: {\"path\": \"string\", \"content\": \"string\"}",
            "- list_dir: List files in a directory inside the workspace. Parameters: {\"path\": \"string\"}",
            "- get_time: Get the current date and time (UTC). Parameters: {}",
            "When you need a tool, emit exactly one block and nothing else after it:",
            "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>",
            "The tool result will appear as <tool_result name=\"tool_name\">...</tool_result>. Then continue. When you can answer directly, answer without a tool call.",
        ]
        .join("\n");
        assert_eq!(ToolRegistry::new().render_system_prompt(), expected);
    }

    #[test]
    fn test_civil_from_days() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20000), (2024, 10, 4));
        assert_eq!(civil_from_days(20454), (2026, 1, 1));
        let now = utc_now_string();
        assert_eq!(now.len(), 16);
        assert_eq!(&now[4..5], "-");
        assert_eq!(&now[7..8], "-");
        assert_eq!(&now[10..11], " ");
        assert_eq!(&now[13..14], ":");
    }

    #[test]
    fn test_file_tools_roundtrip() {
        let ws = test_workspace();
        let _guard = TestDirGuard(ws.clone());
        let reg = ToolRegistry::new();
        let write = ToolCall {
            name: "write_file".to_string(),
            arguments: parse_json("{\"path\": \"notes/x.txt\", \"content\": \"hello\"}").unwrap(),
        };
        assert_eq!(
            reg.execute(&write, &ws).unwrap(),
            "wrote notes/x.txt (5 bytes)"
        );
        let read = ToolCall {
            name: "read_file".to_string(),
            arguments: parse_json("{\"path\": \"notes/x.txt\"}").unwrap(),
        };
        assert_eq!(reg.execute(&read, &ws).unwrap(), "hello");
        let list = ToolCall {
            name: "list_dir".to_string(),
            arguments: parse_json("{\"path\": \"notes\"}").unwrap(),
        };
        assert_eq!(reg.execute(&list, &ws).unwrap(), "x.txt");
        // Escape rejected.
        let esc = ToolCall {
            name: "read_file".to_string(),
            arguments: parse_json("{\"path\": \"../escape.txt\"}").unwrap(),
        };
        assert!(reg.execute(&esc, &ws).is_err());
        // Unknown tool + missing parameter.
        let unknown = ToolCall {
            name: "shell".to_string(),
            arguments: parse_json("{}").unwrap(),
        };
        assert!(reg.execute(&unknown, &ws).is_err());
        let missing = ToolCall {
            name: "calculator".to_string(),
            arguments: parse_json("{}").unwrap(),
        };
        assert!(reg.execute(&missing, &ws).is_err());
    }

    #[test]
    fn test_agent_loop_tool_roundtrip() {
        let ws = test_workspace();
        let _guard = TestDirGuard(ws.clone());
        let agent = AgentLoop::new(ws);
        let mut scripted = 0usize;
        let mut generate = |_prompt: &str| -> String {
            scripted += 1;
            match scripted {
                1 => "Let me calculate.\n<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"12*3+5\"}}</tool_call>\n<tool_result name=\"calculator\">999</tool_result>\nThe answer is 999.".to_string(),
                _ => "The total is 41.".to_string(),
            }
        };
        let outcome = agent.run("what is 12*3+5?", &mut generate);
        // The hallucinated 999 result must NOT survive: the real
        // executed result (41) is what enters the transcript.
        assert!(outcome
            .transcript
            .contains("<tool_result name=\"calculator\">41</tool_result>"));
        assert!(!outcome.transcript.contains("999"));
        assert_eq!(outcome.answer, "The total is 41.");
        assert_eq!(outcome.steps, 1);
        assert_eq!(outcome.tool_calls, 1);
        assert!(outcome.finished);
    }

    #[test]
    fn test_agent_loop_no_tool() {
        let ws = test_workspace();
        let _guard = TestDirGuard(ws.clone());
        let agent = AgentLoop::new(ws);
        let mut generate = |_prompt: &str| -> String { "Paris.".to_string() };
        let outcome = agent.run("capital of France?", &mut generate);
        assert_eq!(outcome.answer, "Paris.");
        assert_eq!(outcome.steps, 0);
        assert_eq!(outcome.tool_calls, 0);
        assert!(outcome.finished);
    }

    #[test]
    fn test_agent_loop_max_steps() {
        let ws = test_workspace();
        let _guard = TestDirGuard(ws.clone());
        let mut agent = AgentLoop::new(ws);
        agent.max_steps = 2;
        let mut generate = |_prompt: &str| -> String {
            "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1+1\"}}</tool_call>".to_string()
        };
        let outcome = agent.run("loop forever", &mut generate);
        assert!(!outcome.finished);
        assert_eq!(outcome.steps, 2);
        assert_eq!(outcome.tool_calls, 2);
    }
}
