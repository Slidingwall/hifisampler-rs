pub mod hnsep;
pub mod hifigan;
use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
use once_cell::sync::OnceCell;
use ndarray::{Array2, Array3,s};
use lele::tensor::TensorView;
use crate::consts::HIFI_CONFIG;
use crate::model::hifigan::{HiFiGAN, HiFiGANWorkspace};
use crate::model::hnsep::{HNSEP, HNSEPWorkspace};
pub struct VocoderContext {
    pub model: HiFiGAN<'static>,
    pub workspace: HiFiGANWorkspace,
}
impl VocoderContext {
    pub fn run(&mut self, mel: &Array2<f32>, f0: &[f32]) -> Vec<f32> {
        let (n_frames, n_mels) = mel.dim();  
        self.model.forward_with_workspace(
            &mut self.workspace,
            TensorView::from_slice(mel.as_slice().unwrap(), vec![1, n_frames, n_mels]),
            TensorView::from_slice(f0, vec![1, f0.len()]),
        ).data.as_ref().to_vec()
    }
}
pub struct RemoverContext {
    pub model: HNSEP<'static>,
    pub workspace: HNSEPWorkspace,
}
impl RemoverContext {
    pub fn run(&mut self, spec: &Array3<f32>) -> Array2<f32> {
        let (ch, bins, frames) = spec.dim();
        assert_eq!(ch, 2);
        let pad_frames = (frames + 15) / 16 * 16;
        let input_tensor = if pad_frames == frames {
            TensorView::from_slice(spec.as_slice().unwrap(), vec![1, 2, bins, frames])
        } else {
            let mut padded = Array3::zeros((2, bins, pad_frames));
            padded.slice_mut(s![.., .., 0..frames]).assign(spec);
            TensorView::from_owned(padded.into_raw_vec_and_offset().0, vec![1, 2, bins, pad_frames])
        };
        let binding = self.model.forward_with_workspace(&mut self.workspace, input_tensor);
        let output = binding.data.as_ref();
        let half = bins * pad_frames;
        let mut mag_out = Array2::zeros((bins, frames));
        for i in 0..bins {
            for j in 0..frames {
                let idx = i * pad_frames + j;
                mag_out[(i, j)] = output[idx].hypot(output[half + idx]);
            }
        }
        mag_out
    }
}
static VOCODER_POOL: OnceCell<Vec<Mutex<VocoderContext>>> = OnceCell::new();
static REMOVER_POOL: OnceCell<Vec<Mutex<RemoverContext>>> = OnceCell::new();
static NEXT_VOCODER: AtomicUsize = AtomicUsize::new(0);
static NEXT_REMOVER: AtomicUsize = AtomicUsize::new(0);
pub fn initialize_models(max_workers: usize) {
    if !HIFI_CONFIG.vocoder_path.exists() {
        tracing::error!("HiFiGAN model not found at: {}", HIFI_CONFIG.vocoder_path.display());
    }
    if !HIFI_CONFIG.hnsep_path.exists() {
        tracing::error!("HNSEP model not found at: {}", HIFI_CONFIG.hnsep_path.display());
    }
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pool_size = max_workers.min(cpu_cores).max(1);
    tracing::info!("Creating model pool with size = {}", pool_size);
    let vocoder_data = Box::leak(std::fs::read(&HIFI_CONFIG.vocoder_path).expect("Failed to read vocoder weights").into_boxed_slice());
    let _ = VOCODER_POOL.set(
        (0..pool_size)
            .map(|_| Mutex::new(VocoderContext {
                model: HiFiGAN::new(vocoder_data),
                workspace: HiFiGANWorkspace::new(),
            }))
            .collect()
    );
    let remover_data = Box::leak(std::fs::read(&HIFI_CONFIG.hnsep_path).expect("Failed to read remover weights").into_boxed_slice());
    let _ = REMOVER_POOL.set(
        (0..pool_size)
            .map(|_| Mutex::new(RemoverContext {
                model: HNSEP::new(remover_data),
                workspace: HNSEPWorkspace::new(),
            }))
            .collect()
    );
    tracing::info!("All models initialized successfully (weights loaded once).");
}
pub fn get_vocoder() -> &'static Mutex<VocoderContext> {
    let pool = VOCODER_POOL.get().expect("Vocoder pool not initialized");
    &pool[NEXT_VOCODER.fetch_add(1, Ordering::Relaxed) % pool.len()]
}
pub fn get_remover() -> &'static Mutex<RemoverContext> {
    let pool = REMOVER_POOL.get().expect("Remover pool not initialized");
    &pool[NEXT_REMOVER.fetch_add(1, Ordering::Relaxed) % pool.len()]
}