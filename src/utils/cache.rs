use anyhow::{anyhow, Context, Result};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ndarray::{Array0, Array2, Array3};
use ndarray_npy::{read_npy, write_npy, NpzReader, NpzWriter};
use tracing::{info, warn};
use once_cell::sync::Lazy;
use fs2::FileExt;
macro_rules! defer {
    ($($stmt:stmt);* $(;)?) => {
        let _defer = {
            struct Defer<F: FnOnce()>(Option<F>);
            impl<F: FnOnce()> Drop for Defer<F> {
                fn drop(&mut self) {
                    if let Some(f) = self.0.take() {
                        f();
                    }
                }
            }
            Defer(Some(|| { $($stmt);* }))
        };
    };
}
#[derive(Debug, Clone)]
pub struct Features {
    pub mel_origin: Array2<f64>,
    pub scale: f64,
}
#[derive(Debug, Default)]
struct CrossProcessLockManager {
    lock_files: Mutex<HashMap<PathBuf, Arc<File>>>,
}
impl CrossProcessLockManager {
    fn get_lock_file(&self, path: &Path) -> Result<Arc<File>> {
        let lock_path = path.with_extension("lock");
        let mut lock_files = self.lock_files.lock()
            .map_err(|e| anyhow!("Lock manager poisoned: {}", e))?;
        if let Some(file) = lock_files.get(path) {
            return Ok(file.clone());
        }
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Create lock dir: {:?}", parent))?;
        }
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .with_context(|| format!("Open lock file: {:?}", lock_path))?;
        let file_arc = Arc::new(file);
        lock_files.insert(path.to_path_buf(), file_arc.clone());
        Ok(file_arc)
    }
    fn acquire_shared(&self, path: &Path) -> Result<()> {
        let lock_file = self.get_lock_file(path)?;
        (&*lock_file).lock_shared()
            .with_context(|| format!("Acquire shared lock: {:?}", path))?;
        Ok(())
    }
    fn acquire_exclusive(&self, path: &Path, timeout: Duration) -> Result<()> {
        let lock_file = self.get_lock_file(path)?;
        let start = Instant::now();
        loop {
            match (&*lock_file).try_lock_exclusive() {
                Ok(()) => return Ok(()),
                Err(_) => {
                    if start.elapsed() >= timeout {
                        return Err(anyhow!(
                            "Acquire exclusive lock timeout ({}ms): {:?}",
                            timeout.as_millis(),
                            path
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
    fn release(&self, path: &Path) -> Result<()> {
        let lock_file = self.get_lock_file(path)?;
        (&*lock_file).unlock()
            .with_context(|| format!("Release lock: {:?}", path.with_extension("lock")))?;
        Ok(())
    }
}
#[derive(Debug, Default)]
pub struct CacheManager {
    lock_manager: CrossProcessLockManager,
}
impl CacheManager {
    fn validate_file_path(&self, path: &Path) -> Result<()> {
        if path.exists() && path.is_dir() {
            return Err(anyhow!("Path {:?} is a directory (expected file)", path));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Create parent dir: {:?}", parent))?;
        }
        Ok(())
    }
    pub fn load_features_cache(&self, path: &Path, force_generate: bool) -> Result<Option<Features>> {
        if force_generate || !path.exists() {
            return Ok(None);
        }
        self.validate_file_path(path)?;
        self.lock_manager.acquire_shared(path)?;
        defer! {
            let _ = self.lock_manager.release(path);
        }
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                warn!("Open cache {} failed: {}", path.display(), e);
                return Ok(None);
            }
        };
        let mut reader = match NpzReader::new(file) {
            Ok(r) => r,
            Err(e) => {
                warn!("Read NPZ {} failed: {}", path.display(), e);
                return Ok(None);
            }
        };
        let scale_arr: Array0<f64> = reader.by_name("scale")
            .with_context(|| format!("Read 'scale' from {}", path.display()))?;
        let mel_origin = reader.by_name("mel_origin")
            .with_context(|| format!("Read 'mel_origin' from {}", path.display()))?;
        info!("Cache loaded: {}", path.display());
        Ok(Some(Features { mel_origin, scale: scale_arr.into_scalar() }))
    }
    pub fn load_hnsep_cache(&self, path: &Path, force_generate: bool) -> Result<Option<Array3<f64>>> {
        if force_generate || !path.exists() {
            return Ok(None);
        }
        self.validate_file_path(path)?;
        self.lock_manager.acquire_shared(path)?;
        defer! {
            let _ = self.lock_manager.release(path);
        }
        let data = read_npy::<_, Array3<f64>>(path)
            .with_context(|| format!("Read hnsep cache {}", path.display()))?;
        info!("Hnsep cache loaded: {}", path.display());
        Ok(Some(data))
    }
    pub fn save_features_cache(&self, path: &Path, features: &Features) -> Result<Option<Features>> {
        self.validate_file_path(path)?;
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5))?;
        defer! {
            let _ = self.lock_manager.release(path);
        }
        if path.exists() {
            info!("Cache exists, reuse: {}", path.display());
            return self.load_features_cache(path, false);
        }
        let temp_path = path.with_extension("tmp");
        if temp_path.exists() {
            fs::remove_file(&temp_path)
                .with_context(|| format!("Remove temp file {:?}", temp_path))?;
        }
        let file = File::create(&temp_path)
            .with_context(|| format!("Create temp file {:?}", temp_path))?;
        let mut writer = NpzWriter::new(file);
        if features.mel_origin.is_empty() {
            return Err(anyhow!("Empty mel_origin cannot be saved"));
        }
        writer.add_array("mel_origin", &features.mel_origin)
            .with_context(|| "Write mel_origin to NPZ")?;
        writer.add_array("scale", &Array0::from_elem((), features.scale))
            .with_context(|| "Write scale to NPZ")?;
        writer.finish()
            .with_context(|| "Finalize NPZ writer")?;
        fs::rename(&temp_path, path)
            .with_context(|| format!("Rename {:?} → {:?}", temp_path, path))?;
        info!("Features saved to: {}", path.display());
        Ok(Some(features.clone()))
    }
    pub fn save_hnsep_cache(&self, path: &Path, data: &Array3<f64>) -> Result<Option<Array3<f64>>> {
        self.validate_file_path(path)?;
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5))?;
        defer! {
            let _ = self.lock_manager.release(path);
        }
        if path.exists() {
            info!("Hnsep cache exists, reuse: {}", path.display());
            return self.load_hnsep_cache(path, false);
        }
        if data.is_empty() {
            return Err(anyhow!("Empty hnsep data"));
        }
        let temp_path = path.with_extension("tmp");
        if temp_path.exists() {
            fs::remove_file(&temp_path)?;
        }
        write_npy(&temp_path, data)
            .with_context(|| format!("Write hnsep temp file {:?}", temp_path))?;
        fs::rename(&temp_path, path)
            .with_context(|| format!("Rename {:?} → {:?}", temp_path, path))?;
        info!("Hnsep saved to: {}", path.display());
        Ok(Some(data.clone()))
    }
}
pub static CACHE_MANAGER: Lazy<CacheManager> = Lazy::new(CacheManager::default);