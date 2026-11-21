//! Append-only CRC32-verified episode WAL (`WAL1`) + `memmap2` versioned
//! hippocampus snapshots (`HPSP`).
//!
//! * High-surprise episodes are written immediately with a per-record IEEE
//!   CRC32 checksum; torn tails on crash recovery are cleanly truncated without
//!   corrupting earlier episodes.
//! * `SnapshotStore` manages numbered / multi-macro snapshots mapped via
//!   `memmap2` for instant rollback (`/reset-mem`).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cortex_kernels::crc32_ieee;
use memmap2::Mmap;

use crate::mlp::HippocampusWeights;

#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    pub surprise: f32,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub context: Vec<u32>,
}

pub struct WalArchive {
    log_path: PathBuf,
    snap_dir: PathBuf,
    log: File,
    last_fsync: Instant,
    fsync_every: Duration,
    pending_since_sync: bool,
}

impl WalArchive {
    pub fn create(dir: impl AsRef<Path>, fsync_every: Duration) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let log_path = dir.join("episodes.wal");
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&log_path)?;
        Ok(Self {
            log_path,
            snap_dir: dir.to_path_buf(),
            log,
            last_fsync: Instant::now(),
            fsync_every,
            pending_since_sync: false,
        })
    }

    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    pub fn snap_dir(&self) -> &Path {
        &self.snap_dir
    }

    pub fn fsync_every(&self) -> Duration {
        self.fsync_every
    }

    pub fn append(&mut self, ep: &Episode) -> std::io::Result<()> {
        write_episode(&mut self.log, ep)?;
        self.pending_since_sync = true;
        self.maybe_fsync(false)?;
        Ok(())
    }

    pub fn maybe_fsync(&mut self, force: bool) -> std::io::Result<()> {
        if !self.pending_since_sync && !force {
            return Ok(());
        }
        if force || self.last_fsync.elapsed() >= self.fsync_every {
            self.log.flush()?;
            self.log.sync_all()?;
            self.last_fsync = Instant::now();
            self.pending_since_sync = false;
        }
        Ok(())
    }

    /// Read all valid records from `episodes.wal`.
    pub fn read_all(&self) -> std::io::Result<Vec<Episode>> {
        Self::read_file(&self.log_path).map(|(eps, _)| eps)
    }

    /// Read all valid records from an arbitrary WAL file, returning
    /// `(episodes, truncated_tail_detected)`.
    pub fn read_file(path: &Path) -> std::io::Result<(Vec<Episode>, bool)> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), false)),
            Err(e) => return Err(e),
        };
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut out = Vec::new();
        let mut o = 0usize;
        while o < buf.len() {
            if o + 8 > buf.len() || &buf[o..o + 4] != b"WAL1" {
                return Ok((out, true));
            }
            o += 4;
            let len = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
            o += 4;
            if o + len + 4 > buf.len() {
                return Ok((out, true));
            }
            let rec = &buf[o..o + len];
            let crc_stored = u32::from_le_bytes(buf[o + len..o + len + 4].try_into().unwrap());
            o += len + 4;
            if crc32_ieee(rec) != crc_stored {
                return Ok((out, true));
            }
            match decode_episode(rec) {
                Ok(ep) => out.push(ep),
                Err(_) => return Ok((out, true)),
            }
        }
        Ok((out, false))
    }

    /// Write single-layer θ to `snap_{version}.bin` and return the path.
    pub fn write_snapshot(
        &self,
        weights: &HippocampusWeights,
        unix_secs: u64,
    ) -> std::io::Result<PathBuf> {
        let path = self
            .snap_dir
            .join(format!("snap_{:016x}.bin", weights.version));
        let mut payload = weights.to_bytes();
        payload.extend_from_slice(&unix_secs.to_le_bytes());
        let mut f = File::create(&path)?;
        f.write_all(&payload)?;
        f.sync_all()?;
        Ok(path)
    }

    /// Map a snapshot with `memmap2` and restore θ.
    pub fn load_snapshot(path: &Path) -> std::io::Result<(HippocampusWeights, u64)> {
        let f = File::open(path)?;
        let map = unsafe { Mmap::map(&f)? };
        if map.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "short snapshot",
            ));
        }
        let body_len = map.len() - 8;
        let weights = HippocampusWeights::from_bytes(&map[..body_len])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut ts = [0u8; 8];
        ts.copy_from_slice(&map[body_len..]);
        let unix_secs = u64::from_le_bytes(ts);
        let _ = map[0];
        Ok((weights, unix_secs))
    }
}

/// Multi-macro versioned hippocampus snapshot store (`hippo-v{ver}.snap`),
/// supporting save + rollback across ALL macro blocks of `CortexModel`.
pub struct SnapshotStore {
    pub dir: PathBuf,
}

impl SnapshotStore {
    pub fn new(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn path_for(&self, version: u64) -> PathBuf {
        self.dir.join(format!("hippo-v{version}.snap"))
    }

    pub fn list_versions(&self) -> Vec<u64> {
        let mut vs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                if let Some(name) = e.file_name().to_str() {
                    if let Some(rest) = name
                        .strip_prefix("hippo-v")
                        .and_then(|r| r.strip_suffix(".snap"))
                    {
                        if let Ok(v) = rest.parse::<u64>() {
                            vs.push(v);
                        }
                    }
                }
            }
        }
        vs.sort_unstable();
        vs
    }

    /// Save a vector of macro hippocampus weights (`[n_macros]`) as the next
    /// monotonically increasing snapshot version. Uses `HPSP` container magic
    /// + IEEE CRC32.
    pub fn save_macros(&self, macros: &[HippocampusWeights]) -> std::io::Result<u64> {
        let version = self.list_versions().last().copied().unwrap_or(0) + 1;
        let mut body = Vec::new();
        body.extend_from_slice(&(macros.len() as u32).to_le_bytes());
        for w in macros {
            let chunk = w.to_bytes();
            body.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            body.extend_from_slice(&chunk);
        }
        let crc = crc32_ieee(&body);
        let path = self.path_for(version);
        let mut f = File::create(&path)?;
        f.write_all(b"HPSP")?;
        f.write_all(&0xFFFF_0001u32.to_le_bytes())?; // multi-macro container tag
        f.write_all(&version.to_le_bytes())?;
        f.write_all(&body)?;
        f.write_all(&crc.to_le_bytes())?;
        f.sync_all()?;
        Ok(version)
    }

    /// Load a multi-macro snapshot via `memmap2`.
    pub fn load_macros(&self, version: u64) -> std::io::Result<Vec<HippocampusWeights>> {
        let f = File::open(self.path_for(version))?;
        let map = unsafe { Mmap::map(&f)? };
        if map.len() < 20 || &map[0..4] != b"HPSP" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad multi-macro snapshot magic",
            ));
        }
        let tag = u32::from_le_bytes(map[4..8].try_into().unwrap());
        if tag != 0xFFFF_0001 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unexpected snapshot container tag",
            ));
        }
        let body = &map[16..map.len() - 4];
        let crc_stored = u32::from_le_bytes(map[map.len() - 4..].try_into().unwrap());
        if crc32_ieee(body) != crc_stored {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "multi-macro snapshot CRC32 mismatch",
            ));
        }
        let count = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
        let mut off = 4usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            if off + 4 > body.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "truncated macro snapshot entry",
                ));
            }
            let len = u32::from_le_bytes(body[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + len > body.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "truncated macro snapshot payload",
                ));
            }
            let w = HippocampusWeights::from_bytes(&body[off..off + len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            off += len;
            out.push(w);
        }
        Ok(out)
    }

    pub fn rollback_macros(&self, version: u64) -> std::io::Result<Vec<HippocampusWeights>> {
        self.load_macros(version)
    }
}

fn write_episode(w: &mut impl Write, ep: &Episode) -> std::io::Result<()> {
    let mut body = Vec::new();
    body.extend_from_slice(&ep.surprise.to_le_bytes());
    push_f32_vec(&mut body, &ep.key);
    push_f32_vec(&mut body, &ep.value);
    body.extend_from_slice(&(ep.context.len() as u32).to_le_bytes());
    for t in &ep.context {
        body.extend_from_slice(&t.to_le_bytes());
    }
    let crc = crc32_ieee(&body);
    w.write_all(b"WAL1")?;
    w.write_all(&(body.len() as u32).to_le_bytes())?;
    w.write_all(&body)?;
    w.write_all(&crc.to_le_bytes())?;
    Ok(())
}

fn push_f32_vec(out: &mut Vec<u8>, xs: &[f32]) {
    out.extend_from_slice(&(xs.len() as u32).to_le_bytes());
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn decode_episode(rec: &[u8]) -> Result<Episode, String> {
    let mut o = 0usize;
    if rec.len() < 4 {
        return Err("short episode".into());
    }
    let surprise = f32::from_le_bytes(rec[o..o + 4].try_into().unwrap());
    o += 4;
    let key = take_f32(rec, &mut o)?;
    let value = take_f32(rec, &mut o)?;
    if o + 4 > rec.len() {
        return Err("short context".into());
    }
    let n = u32::from_le_bytes(rec[o..o + 4].try_into().unwrap()) as usize;
    o += 4;
    let mut context = Vec::with_capacity(n);
    for _ in 0..n {
        if o + 4 > rec.len() {
            return Err("truncated context".into());
        }
        context.push(u32::from_le_bytes(rec[o..o + 4].try_into().unwrap()));
        o += 4;
    }
    if o != rec.len() {
        return Err("trailing bytes in episode".into());
    }
    Ok(Episode {
        surprise,
        key,
        value,
        context,
    })
}

fn take_f32(rec: &[u8], o: &mut usize) -> Result<Vec<f32>, String> {
    if *o + 4 > rec.len() {
        return Err("truncated vec".into());
    }
    let n = u32::from_le_bytes(rec[*o..*o + 4].try_into().unwrap()) as usize;
    *o += 4;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if *o + 4 > rec.len() {
            return Err("truncated f32".into());
        }
        out.push(f32::from_le_bytes(rec[*o..*o + 4].try_into().unwrap()));
        *o += 4;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn wal_append_torn_tail_recovery_and_snapshot_rollback() {
        let dir = std::env::temp_dir().join(format!(
            "cortex-wal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut wal = WalArchive::create(&dir, Duration::from_secs(5)).unwrap();
        let ep = Episode {
            surprise: 0.4,
            key: vec![0.25, -0.5],
            value: vec![1.0, 2.0, 3.0],
            context: vec![7, 8, 9],
        };
        wal.append(&ep).unwrap();
        wal.maybe_fsync(true).unwrap();

        // Append a torn tail to simulate mid-write crash.
        {
            let mut f = OpenOptions::new()
                .append(true)
                .open(wal.log_path())
                .unwrap();
            f.write_all(b"WAL1\x20\x00\x00\x00\xAA\xBB").unwrap();
        }
        let (got, truncated) = WalArchive::read_file(wal.log_path()).unwrap();
        assert!(truncated, "torn tail must be flagged");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].context, vec![7, 8, 9]);

        // Multi-macro SnapshotStore save + rollback.
        let store = SnapshotStore::new(&dir).unwrap();
        let mut w_a = HippocampusWeights::new(8, 8, 8);
        w_a.w1.as_mut_slice()[3] = 0.1234;
        w_a.version = 2;
        let v1 = store.save_macros(&[w_a.clone(), w_a.clone()]).unwrap();
        assert_eq!(v1, 1);

        let mut w_b = w_a.clone();
        w_b.w1.as_mut_slice()[3] = -9.0;
        let v2 = store.save_macros(&[w_b.clone(), w_b]).unwrap();
        assert_eq!(v2, 2);

        let rolled = store.rollback_macros(1).unwrap();
        assert_eq!(rolled.len(), 2);
        assert!((rolled[1].w1.as_slice()[3] - 0.1234).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
