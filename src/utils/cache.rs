use fs2::FileExt;
use ndarray::{Array0, Array2};
use ndarray_npy::{NpzReader, NpzWriter};
use once_cell::sync::Lazy;
use oxifft::Complex;
use std::collections::HashMap;
use std::fs::{create_dir_all, rename, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};
macro_rules! defer {
    ($($stmt:stmt);* $(;)?) => {
        struct Defer<F: FnOnce()>(Option<F>);
        impl<F: FnOnce()> Drop for Defer<F> {
            fn drop(&mut self) { self.0.take().map(|f| f()); }
        }
        let _defer = Defer(Some(|| { $($stmt);* }));
    };
}
#[derive(Debug, Clone)]
pub struct Features {
    pub mel_origin: Array2<f32>,
    pub scale: f32,
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
        let lock_file = self.get_lock_file(path);
        (&*lock_file).lock_shared().unwrap();
    }
    fn acquire_exclusive(&self, path: &Path, timeout: Duration) {
        let lock_file = self.get_lock_file(path);
        let start = Instant::now();
        loop {
            if let Ok(()) = (&*lock_file).try_lock_exclusive() {
                return;
            }
            if start.elapsed() >= timeout {
                panic!("Acquire exclusive lock timeout ({}ms): {:?}", timeout.as_millis(), path);
            } else {
                thread::sleep(Duration::from_millis(10));
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
            create_dir_all(parent).unwrap();
        }
    }
    pub fn load_features_cache(&self, path: &Path, force_gen: bool) -> Option<Features> {
        if force_gen || !path.exists() {
            return None;
        }
        self.lock_manager.acquire_shared(path);
        defer! { self.lock_manager.release(path); }
        let file = File::open(path).map_err(|e| warn!("Open cache {} failed: {}", path.display(), e)).ok()?;
        let mut reader = NpzReader::new(file).map_err(|e| warn!("Read NPZ {} failed: {}", path.display(), e)).ok()?;
        let scale_arr: Array0<f32> = reader.by_name("scale").unwrap();
        let mel_origin = reader.by_name("mel_origin").unwrap();
        info!("Cache loaded: {}", path.display());
        Some(Features { mel_origin, scale: scale_arr.into_scalar() })
    }
    pub fn load_hnsep_cache(&self, path: &Path, force_gen: bool) -> Option<Array2<Complex<f32>>> {
        if force_gen || !path.exists() {
            return None;
        }
        self.lock_manager.acquire_shared(path);
        defer! { self.lock_manager.release(path); }
        let file = File::open(path)
            .map_err(|e| warn!("Open HNSEP cache {} failed: {}", path.display(), e))
            .ok()?;
        let mut reader = NpzReader::new(file)
            .map_err(|e| warn!("Read HNSEP NPZ {} failed: {}", path.display(), e))
            .ok()?;
        let real: Array2<f32> = reader.by_name("real").unwrap();
        let imag: Array2<f32> = reader.by_name("imag").unwrap();
        let dim = real.dim();
        let spec = Array2::from_shape_fn(dim, |(i, j)| {
            Complex::new(real[(i, j)], imag[(i, j)])
        });
        info!("HNSEP cache loaded: {}", path.display());
        Some(spec)
    }
    pub fn save_features_cache(&self, path: &Path, features: &Features) {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! { self.lock_manager.release(path); }
        let tmp_path = path.with_extension("tmp");
        let file = File::create(&tmp_path).unwrap();
        let mut writer = NpzWriter::new(file);
        writer.add_array("mel_origin", &features.mel_origin).unwrap();
        writer.add_array("scale", &Array0::from_elem((), features.scale)).unwrap();
        writer.finish().unwrap();
        rename(&tmp_path, path).unwrap();
        info!("Features saved to: {}", path.display());
    }
    pub fn save_hnsep_cache(&self, path: &Path, spec: &Array2<Complex<f32>>) {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! { self.lock_manager.release(path); }
        let tmp_path = path.with_extension("tmp");
        let file = File::create(&tmp_path).unwrap();
        let mut writer = NpzWriter::new(file);
        let real = spec.map(|c| c.re);
        let imag = spec.map(|c| c.im);
        writer.add_array("real", &real).unwrap();
        writer.add_array("imag", &imag).unwrap();
        writer.finish().unwrap();
        rename(&tmp_path, path).unwrap();
        info!("HNSEP cache saved: {}", path.display());
    }
}
pub static CACHE_MANAGER: Lazy<CacheManager> = Lazy::new(CacheManager::default);