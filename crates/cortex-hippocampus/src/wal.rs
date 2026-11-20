//! Append-only episode WAL plus mmap snapshots.
//!
//! Records are written as soon as they are queued. `fsync` runs when the
//! configured cadence elapses (default 5s) or when the caller forces it.
//! Snapshots are immutable files; the loader maps them with `memmap2` and
//! copies θ back into the live weights. Restoring an older file does not
//! rewrite it (the previous version stays on disk for another rollback).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use memmap2::Mmap;

use crate::mlp::HippocampusWeights;

#[derive(Clone, Debug)]
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

    pub fn read_all(&self) -> std::io::Result<Vec<Episode>> {
        let mut f = File::open(&self.log_path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut out = Vec::new();
        let mut o = 0usize;
        while o + 8 <= buf.len() {
            if &buf[o..o + 4] != b"WAL1" {
                break;
            }
            o += 4;
            let len = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
            o += 4;
            if o + len > buf.len() {
                break;
            }
            let rec = &buf[o..o + len];
            o += len;
            out.push(
                decode_episode(rec)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            );
        }
        Ok(out)
    }

    /// Write θ to `snap_{version}.bin` and return the path. The file is synced.
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

    /// Map a snapshot and restore θ. Uses `memmap2` (read-only map).
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
        // Touch the mapping so a CoW-style private view is actually resident.
        let _ = map[0];
        Ok((weights, unix_secs))
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
    w.write_all(b"WAL1")?;
    w.write_all(&(body.len() as u32).to_le_bytes())?;
    w.write_all(&body)?;
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
    fn wal_append_and_snapshot_rollback() {
        let dir = std::env::temp_dir().join(format!(
            "cortex-wal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut wal = WalArchive::create(&dir, Duration::from_secs(5)).unwrap();
        assert_eq!(wal.fsync_every(), Duration::from_secs(5));
        let ep = Episode {
            surprise: 0.4,
            key: vec![0.25, -0.5],
            value: vec![1.0, 2.0, 3.0],
            context: vec![7, 8, 9],
        };
        wal.append(&ep).unwrap();
        wal.maybe_fsync(true).unwrap();
        let got = wal.read_all().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].context, vec![7, 8, 9]);
        assert!((got[0].key[1] + 0.5).abs() < 1e-6);

        let mut weights = HippocampusWeights::new(8, 8, 8);
        let marker = 0.1234f32;
        weights.w1.as_mut_slice()[3] = marker;
        weights.version = 2;
        let path = wal.write_snapshot(&weights, 1_700_000_000).unwrap();
        weights.w1.as_mut_slice()[3] = -9.0;
        weights.version = 9;
        let (restored, ts) = WalArchive::load_snapshot(&path).unwrap();
        assert_eq!(ts, 1_700_000_000);
        assert_eq!(restored.version, 2);
        assert!((restored.w1.as_slice()[3] - marker).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
