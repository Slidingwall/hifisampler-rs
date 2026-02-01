use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use ndarray::{Array0, Array1, Array2};
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
    fn get_lock_file(&self, path: &Path) -> Arc<File> {
        let lock_path = path.with_extension("lock");
        let mut lock_files = self.lock_files.lock().unwrap();
        if let Some(file) = lock_files.get(path) {
            return file.clone();
        }
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .open(&lock_path)
            .unwrap();
        let file_arc = Arc::new(file);
        lock_files.insert(path.to_path_buf(), file_arc.clone());
        file_arc
    }
    fn acquire_shared(&self, path: &Path) {
        let lock_file = self.get_lock_file(path);
        (&*lock_file).lock_shared().unwrap();
    }
    fn acquire_exclusive(&self, path: &Path, timeout: Duration) {
        let lock_file = self.get_lock_file(path);
        let start = Instant::now();
        loop {
            match (&*lock_file).try_lock_exclusive() {
                Ok(()) => return,
                Err(_) => {
                    if start.elapsed() >= timeout {
                        panic!("Acquire exclusive lock timeout ({}ms): {:?}", timeout.as_millis(), path);
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
    fn release(&self, path: &Path) {
        let lock_file = self.get_lock_file(path);
        (&*lock_file).unlock().unwrap();
    }
}
#[derive(Debug, Default)]
pub struct CacheManager {
    lock_manager: CrossProcessLockManager,
}
impl CacheManager {
    fn validate_file_path(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
    }
    pub fn load_features_cache(&self, path: &Path, force_generate: bool) -> Option<Features> {
        if force_generate || !path.exists() {
            return None;
        }
        self.validate_file_path(path);
        self.lock_manager.acquire_shared(path);
        defer! {
            let _ = self.lock_manager.release(path);
        }
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                warn!("Open cache {} failed: {}", path.display(), e);
                return None;
            }
        };
        let mut reader = match NpzReader::new(file) {
            Ok(r) => r,
            Err(e) => {
                warn!("Read NPZ {} failed: {}", path.display(), e);
                return None;
            }
        };
        let scale_arr: Array0<f64> = reader.by_name("scale").unwrap();
        let mel_origin = reader.by_name("mel_origin").unwrap();
        info!("Cache loaded: {}", path.display());
        Some(Features { mel_origin, scale: scale_arr.into_scalar() })
    }
    pub fn load_hnsep_cache(&self, path: &Path, force_generate: bool) -> Option<Vec<f64>> {
        if force_generate || !path.exists() {
            return None;
        }
        self.validate_file_path(path);
        self.lock_manager.acquire_shared(path);
        defer! {
            let _ = self.lock_manager.release(path);
        }
        let arr1_data = read_npy::<_, Array1<f64>>(path).unwrap();
        let vec_data = arr1_data.to_vec();
        info!("Hnsep cache loaded: {} (length: {})", path.display(), vec_data.len());
        Some(vec_data)
    }
    pub fn save_features_cache(&self, path: &Path, features: &Features) -> Option<Features> {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! {
            let _ = self.lock_manager.release(path);
        }
        if path.exists() {
            info!("Cache exists, reuse: {}", path.display());
            return self.load_features_cache(path, false);
        }
        let temp_path = path.with_extension("tmp");
        let file = File::create(&temp_path).unwrap();
        let mut writer = NpzWriter::new(file);
        writer.add_array("mel_origin", &features.mel_origin).unwrap();
        writer.add_array("scale", &Array0::from_elem((), features.scale)).unwrap();
        writer.finish().unwrap();
        fs::rename(&temp_path, path).unwrap();
        info!("Features saved to: {}", path.display());
        Some(features.clone())
    }
    pub fn save_hnsep_cache(&self, path: &Path, data: Vec<f64>) -> Option<Vec<f64>> {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! {
            let _ = self.lock_manager.release(path);
        }
        if path.exists() {
            info!("Hnsep cache exists, reuse: {}", path.display());
            return self.load_hnsep_cache(path, false);
        }
        let temp_path = path.with_extension("tmp");
        let arr1_data = Array1::from_vec(data);
        write_npy(&temp_path, &arr1_data).unwrap();
        fs::rename(&temp_path, path).unwrap();
        info!("Hnsep saved to: {} (length: {})", path.display(), arr1_data.len());
        Some(arr1_data.to_vec())
    }
}
pub static CACHE_MANAGER: Lazy<CacheManager> = Lazy::new(CacheManager::default);