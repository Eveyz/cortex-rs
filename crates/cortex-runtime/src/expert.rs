//! L2 micro-expert slots and shareable packages.
//!
//! A package is a ternary adapter plus compatibility metadata. It cannot carry
//! L1 hippocampus weights or a private WAL. Inference routes to one mounted
//! slot; the others stay inactive.

use std::fs;
use std::path::{Path, PathBuf};

use crate::ctxb::crc32;

#[derive(Clone, Debug)]
pub struct ExpertMeta {
    pub name: String,
    pub domain: String,
    pub license: String,
    pub d_model: u32,
    pub vocab: u32,
    pub router_abi: u32,
}

#[derive(Clone, Debug)]
pub struct ExpertAdapter {
    pub rows: u32,
    pub cols: u32,
    pub tiles: Vec<u8>,
}

#[derive(Clone, Debug)]
struct Mounted {
    meta: ExpertMeta,
    adapter: ExpertAdapter,
    last_used: u64,
}

pub struct ExpertRouter {
    slots: Vec<Option<Mounted>>,
    cap: usize,
    disk_budget: usize,
    clock: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertError {
    Cap,
    Disk,
    ForbiddenPrivate,
    BadMagic,
    Incompatible,
    Checksum,
    Empty,
}

impl ExpertRouter {
    pub fn new(cap: usize, disk_budget: usize) -> Self {
        Self {
            slots: (0..cap).map(|_| None).collect(),
            cap,
            disk_budget,
            clock: 0,
        }
    }

    pub fn mounted(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub fn disk_used(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|m| m.adapter.tiles.len())
            .sum()
    }

    /// Mount, evicting the least-recently-used slot when the cap or disk budget
    /// would be exceeded. Returns the slot index.
    pub fn mount(
        &mut self,
        meta: ExpertMeta,
        adapter: ExpertAdapter,
    ) -> Result<usize, ExpertError> {
        if adapter.tiles.is_empty() {
            return Err(ExpertError::Empty);
        }
        if contains_private(&adapter.tiles) {
            return Err(ExpertError::ForbiddenPrivate);
        }
        while self.mounted() >= self.cap
            || self.disk_used() + adapter.tiles.len() > self.disk_budget
        {
            if !self.evict_lru() {
                return Err(ExpertError::Disk);
            }
        }
        let idx = self
            .slots
            .iter()
            .position(|s| s.is_none())
            .ok_or(ExpertError::Cap)?;
        self.clock += 1;
        self.slots[idx] = Some(Mounted {
            meta,
            adapter,
            last_used: self.clock,
        });
        Ok(idx)
    }

    pub fn unmount(&mut self, slot: usize) -> bool {
        if slot >= self.slots.len() || self.slots[slot].is_none() {
            return false;
        }
        self.slots[slot] = None;
        true
    }

    /// Sparse route: only the matching domain is marked active. Others stay mounted
    /// but are not returned.
    pub fn route(&mut self, domain: &str) -> Option<usize> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.as_ref().map(|m| m.meta.domain == domain).unwrap_or(false))?;
        self.clock += 1;
        if let Some(slot) = self.slots[idx].as_mut() {
            slot.last_used = self.clock;
        }
        Some(idx)
    }

    pub fn active_domain(&self, slot: usize) -> Option<&str> {
        self.slots
            .get(slot)?
            .as_ref()
            .map(|m| m.meta.domain.as_str())
    }

    fn evict_lru(&mut self) -> bool {
        let mut best: Option<(usize, u64)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(m) = slot {
                if best.map(|(_, t)| m.last_used < t).unwrap_or(true) {
                    best = Some((i, m.last_used));
                }
            }
        }
        if let Some((i, _)) = best {
            self.slots[i] = None;
            true
        } else {
            false
        }
    }
}

pub fn tiny_adapter(domain: &str, salt: u8) -> ExpertAdapter {
    ExpertAdapter {
        rows: 16,
        cols: 16,
        tiles: vec![salt, b'T', domain.len() as u8],
    }
}

/// Packed tile bytes for a target size, without allocating the buffer.
/// One cache line is 64 bytes and 256 ternary weights.
pub fn adapter_plan(target_bytes: usize) -> (u32, u32, usize) {
    let lines = (target_bytes / 64).max(1);
    let cols: u32 = 1024;
    let rows = ((lines * 16) as u32 / (cols / 16)).max(16);
    let packed = (rows as usize / 16) * (cols as usize / 16) * 64;
    (rows, cols, packed)
}

const MAGIC: &[u8; 4] = b"CXEX";
pub const ROUTER_ABI: u32 = 1;

#[derive(Clone, Debug)]
pub struct ExpertPackage {
    pub meta: ExpertMeta,
    pub adapter: ExpertAdapter,
    pub checksum: u32,
}

impl ExpertPackage {
    pub fn export(meta: ExpertMeta, adapter: ExpertAdapter) -> Result<Self, ExpertError> {
        if contains_private(&adapter.tiles) {
            return Err(ExpertError::ForbiddenPrivate);
        }
        let bytes = encode(&meta, &adapter);
        let checksum = crc32(&bytes);
        Ok(Self {
            meta,
            adapter,
            checksum,
        })
    }

    /// Refuse a sidecar that is a hippocampus snapshot or a WAL. The sidecar is
    /// never copied into the package.
    pub fn export_refusing_private(
        meta: ExpertMeta,
        adapter: ExpertAdapter,
        sidecar: &[u8],
    ) -> Result<Self, ExpertError> {
        if contains_private(sidecar) || contains_private(&adapter.tiles) {
            return Err(ExpertError::ForbiddenPrivate);
        }
        Self::export(meta, adapter)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = encode(&self.meta, &self.adapter);
        let sum = crc32(&body);
        body.extend_from_slice(&sum.to_le_bytes());
        body
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ExpertError> {
        if bytes.len() < 8 || &bytes[..4] != MAGIC {
            return Err(ExpertError::BadMagic);
        }
        let (body, sum_bytes) = bytes.split_at(bytes.len() - 4);
        let stored = u32::from_le_bytes(sum_bytes.try_into().unwrap());
        if crc32(body) != stored {
            return Err(ExpertError::Checksum);
        }
        if contains_private(body) {
            return Err(ExpertError::ForbiddenPrivate);
        }
        let (meta, adapter) = decode(body)?;
        Ok(Self {
            meta,
            adapter,
            checksum: stored,
        })
    }

    pub fn install_into(
        &self,
        host_d_model: u32,
        host_vocab: u32,
        router: &mut ExpertRouter,
    ) -> Result<usize, ExpertError> {
        if self.meta.d_model != host_d_model
            || self.meta.vocab != host_vocab
            || self.meta.router_abi != ROUTER_ABI
        {
            return Err(ExpertError::Incompatible);
        }
        router.mount(self.meta.clone(), self.adapter.clone())
    }
}

pub fn contains_private(bytes: &[u8]) -> bool {
    bytes.windows(4).any(|w| w == b"HPSP" || w == b"WAL1")
}

fn encode(meta: &ExpertMeta, adapter: &ExpertAdapter) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, 1);
    put_u32(&mut out, meta.router_abi);
    put_u32(&mut out, meta.d_model);
    put_u32(&mut out, meta.vocab);
    put_str(&mut out, &meta.name);
    put_str(&mut out, &meta.license);
    put_str(&mut out, &meta.domain);
    put_u32(&mut out, adapter.rows);
    put_u32(&mut out, adapter.cols);
    put_u32(&mut out, adapter.tiles.len() as u32);
    out.extend_from_slice(&adapter.tiles);
    out
}

fn decode(bytes: &[u8]) -> Result<(ExpertMeta, ExpertAdapter), ExpertError> {
    if bytes.len() < 20 || &bytes[..4] != MAGIC {
        return Err(ExpertError::BadMagic);
    }
    let mut o = 4usize;
    let _version = read_u32(bytes, &mut o)?;
    let router_abi = read_u32(bytes, &mut o)?;
    let d_model = read_u32(bytes, &mut o)?;
    let vocab = read_u32(bytes, &mut o)?;
    let name = read_str(bytes, &mut o)?;
    let license = read_str(bytes, &mut o)?;
    let domain = read_str(bytes, &mut o)?;
    let rows = read_u32(bytes, &mut o)?;
    let cols = read_u32(bytes, &mut o)?;
    let n = read_u32(bytes, &mut o)? as usize;
    if o + n > bytes.len() {
        return Err(ExpertError::BadMagic);
    }
    let tiles = bytes[o..o + n].to_vec();
    Ok((
        ExpertMeta {
            name,
            domain,
            license,
            d_model,
            vocab,
            router_abi,
        },
        ExpertAdapter { rows, cols, tiles },
    ))
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn read_u32(bytes: &[u8], o: &mut usize) -> Result<u32, ExpertError> {
    if *o + 4 > bytes.len() {
        return Err(ExpertError::BadMagic);
    }
    let v = u32::from_le_bytes(bytes[*o..*o + 4].try_into().unwrap());
    *o += 4;
    Ok(v)
}

fn read_str(bytes: &[u8], o: &mut usize) -> Result<String, ExpertError> {
    let n = read_u32(bytes, o)? as usize;
    if *o + n > bytes.len() {
        return Err(ExpertError::BadMagic);
    }
    let s = String::from_utf8_lossy(&bytes[*o..*o + n]).into_owned();
    *o += n;
    Ok(s)
}

pub fn save_package(path: &Path, package: &ExpertPackage) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, package.to_bytes())
}

pub fn load_package(path: &Path) -> Result<ExpertPackage, ExpertError> {
    let bytes = fs::read(path).map_err(|_| ExpertError::BadMagic)?;
    ExpertPackage::from_bytes(&bytes)
}

pub fn list_packages(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cxex") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_hippocampus::HippocampusWeights;

    fn meta(domain: &str) -> ExpertMeta {
        ExpertMeta {
            name: domain.into(),
            domain: domain.into(),
            license: "MIT".into(),
            d_model: 512,
            vocab: 4096,
            router_abi: ROUTER_ABI,
        }
    }

    #[test]
    fn route_activates_only_the_selected_slot_and_evicts() {
        let mut router = ExpertRouter::new(2, 1024);
        let a = router.mount(meta("math"), tiny_adapter("math", 1)).unwrap();
        let b = router.mount(meta("code"), tiny_adapter("code", 2)).unwrap();
        assert_eq!(router.route("math"), Some(a));
        assert_eq!(router.active_domain(a), Some("math"));
        assert_eq!(router.active_domain(b), Some("code"));
        assert_ne!(router.route("math"), router.route("code"));
        router.mount(meta("law"), tiny_adapter("law", 3)).unwrap();
        assert_eq!(router.mounted(), 2);
        assert!(router.route("math").is_none() || router.route("code").is_none());
        let still = router.mounted();
        assert_eq!(still, 2);
    }

    #[test]
    fn package_round_trip_rejects_hippocampus_and_wal() {
        let adapter = tiny_adapter("math", 7);
        let pkg = ExpertPackage::export(meta("math"), adapter.clone()).unwrap();
        let bytes = pkg.to_bytes();
        assert!(
            !contains_private(&bytes),
            "clean package picked up a private magic"
        );
        let back = ExpertPackage::from_bytes(&bytes).unwrap();
        assert_eq!(back.meta.domain, "math");
        assert_eq!(back.adapter.tiles, adapter.tiles);
        assert_eq!(back.checksum, pkg.checksum);

        let hip = HippocampusWeights::new(4, 4, 4).to_bytes();
        assert!(contains_private(&hip));
        let err = ExpertPackage::export_refusing_private(meta("math"), adapter, &hip);
        assert_eq!(err.err(), Some(ExpertError::ForbiddenPrivate));

        let mut poisoned = bytes.clone();
        poisoned.extend_from_slice(b"WAL1");
        assert!(ExpertPackage::from_bytes(&poisoned).is_err());
    }

    #[test]
    fn adapter_plan_is_in_the_shareable_band() {
        let (_rows, _cols, packed) = adapter_plan(24 * 1024 * 1024);
        assert!(packed >= 20 * 1024 * 1024 && packed <= 50 * 1024 * 1024);
    }

    #[test]
    fn install_checks_compatibility() {
        let pkg = ExpertPackage::export(meta("math"), tiny_adapter("math", 1)).unwrap();
        let mut router = ExpertRouter::new(2, 1024);
        assert!(pkg.install_into(512, 4096, &mut router).is_ok());
        assert!(pkg.install_into(256, 4096, &mut router).is_err());
    }
}
