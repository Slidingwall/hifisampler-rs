use fs4::FileExt;
use ndarray::Array2;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fs::{self, create_dir_all, rename, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};
const MEL_MIN: f32 = -20.7232658369464104;
const MEL_DEQUANT_SCALE: f32 = 3.6630311072770638742656595712215e-4;
const MEL_QUANT_SCALE: f32 = 2729.9795462107227172561518593534;
const HNSEP_DEQUANT_SCALE: f32 = 0.015625;
const HNSEP_QUANT_SCALE: f32 = 64.0;
macro_rules! defer {
    ($($stmt:stmt);* $(;)?) => {
        struct Defer<F: FnOnce()>(Option<F>);
        impl<F: FnOnce()> Drop for Defer<F> {
            fn drop(&mut self) { self.0.take().map(|f| f()); }
        }
        let _defer = Defer(Some(|| { $($stmt);* }));
    };
}
#[derive(Debug, Default)]
struct CrossProcessLockManager {
    lock_files: Mutex<HashMap<PathBuf, Arc<File>>>,
}
impl CrossProcessLockManager {
    fn get_lock_file(&self, path: &Path) -> Arc<File> {
        let lock_path = path.with_extension("lock");
        let mut lock_files = self.lock_files.lock().unwrap();
        if let Some(file) = lock_files.get(path) {
            return file.clone();
        }
        if let Some(parent) = lock_path.parent() {
            create_dir_all(parent).unwrap();
        }
        let file = File::options().read(true).write(true).create(true).open(&lock_path).unwrap();
        let file_arc = Arc::new(file);
        lock_files.insert(path.to_path_buf(), file_arc.clone());
        file_arc
    }
    fn acquire_shared(&self, path: &Path) {
        (&*self.get_lock_file(path)).lock_shared().unwrap();
    }
    fn acquire_exclusive(&self, path: &Path, timeout: Duration) {
        let lock_file = self.get_lock_file(path);
        let start = Instant::now();
        loop {
            if let Ok(()) = (&*lock_file).try_lock() {
                return;
            }
            if start.elapsed() >= timeout {
                panic!("Acquire exclusive lock timeout ({}ms): {:?}", timeout.as_millis(), path);
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
    fn release(&self, path: &Path) {
        (&*self.get_lock_file(path)).unlock().unwrap();
    }
}
#[derive(Debug, Default)]
pub struct CacheManager {
    lock_manager: CrossProcessLockManager,
}
impl CacheManager {
    fn validate_file_path(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            create_dir_all(parent).unwrap();
        }
    }
    pub fn load_features_cache(&self, path: &Path, force_gen: bool) -> Option<(Array2<f32>, f32)> {
        if force_gen || !path.exists() {
            return None;
        }
        self.lock_manager.acquire_shared(path);
        defer! { self.lock_manager.release(path); }
        let data = fs::read(path).map_err(|e| warn!("Open cache {} failed: {}", path.display(), e)).ok()?;
        if data.len() < 8 {
            warn!("Invalid cache file: {}", path.display());
            return None;
        }
        let cols = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let scale = f32::from_le_bytes(data[4..8].try_into().unwrap());
        if data.len() < 8 + 128 * cols * 2 {
            warn!("Truncated cache file: {}", path.display());
            return None;
        }
        let mut mel_origin = Array2::zeros((128, cols));
        for (m, chunk) in mel_origin.iter_mut().zip(data[8..].chunks_exact(2)) {
            *m = MEL_MIN + u16::from_le_bytes(chunk.try_into().unwrap()) as f32 * MEL_DEQUANT_SCALE;
        }
        info!("Cache loaded: {}", path.display());
        Some((mel_origin, scale))
    }
    pub fn load_hnsep_cache(&self, path: &Path, force_gen: bool) -> Option<Array2<f32>> {
        if force_gen || !path.exists() {
            return None;
        }
        self.lock_manager.acquire_shared(path);
        defer! { self.lock_manager.release(path); }
        let data = fs::read(path).map_err(|e| warn!("Open HNSEP cache {} failed: {}", path.display(), e)).ok()?;
        if data.len() < 4 {
            warn!("Invalid HNSEP cache file: {}", path.display());
            return None;
        }
        let cols = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        if data.len() < 4 + 1025 * cols * 2 {
            warn!("Truncated HNSEP cache file: {}", path.display());
            return None;
        }
        let mut mag = Array2::zeros((1025, cols));
        for (m, chunk) in mag.iter_mut().zip(data[4..].chunks_exact(2)) {
            *m = u16::from_le_bytes(chunk.try_into().unwrap()) as f32 * HNSEP_DEQUANT_SCALE;
        }
        info!("HNSEP cache loaded: {}", path.display());
        Some(mag)
    }
    pub fn save_features_cache(&self, path: &Path, (mel_origin, scale): &(Array2<f32>, f32)) {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! { self.lock_manager.release(path); }
        let cols = mel_origin.ncols();
        let mut buf = Vec::with_capacity(8 + 128 * cols * 2);
        buf.extend_from_slice(&(cols as u32).to_le_bytes());
        buf.extend_from_slice(&scale.to_le_bytes());
        buf.extend(mel_origin.iter().flat_map(|&x| 
            (((x - MEL_MIN) * MEL_QUANT_SCALE).round().clamp(0.0, 65535.0) as u16).to_le_bytes()
        ));
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, &buf).unwrap();
        rename(&tmp_path, path).unwrap();
        info!("Features saved to: {}", path.display());
    }
    pub fn save_hnsep_cache(&self, path: &Path, mag: &Array2<f32>) {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! { self.lock_manager.release(path); }
        let cols = mag.ncols();
        let mut buf = Vec::with_capacity(4 + 1025 * cols * 2);
        buf.extend_from_slice(&(cols as u32).to_le_bytes());
        buf.extend(mag.iter().flat_map(|&x| 
            ((x * HNSEP_QUANT_SCALE).round().clamp(0.0, 65535.0) as u16).to_le_bytes()
        ));
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, &buf).unwrap();
        rename(&tmp_path, path).unwrap();
        info!("HNSEP cache saved to: {}", path.display());
    }
}
pub static CACHE_MANAGER: Lazy<CacheManager> = Lazy::new(CacheManager::default);