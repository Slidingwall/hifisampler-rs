use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::Value };
use ndarray::{Array2, Axis};
#[derive(Debug)]
pub struct HiFiGANLoader {
    session: Session,
}
impl HiFiGANLoader {
    pub fn new(model_path: &PathBuf) -> Self {
        Self {
            session: Session::builder().unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
                .commit_from_file(model_path).unwrap()
        }
    }
    pub fn run(&mut self, mel: Array2<f32>, f0: Vec<f32>) -> Vec<f32> {
        let (n_mels, n_frames) = mel.dim();
        let mel_flat: Vec<f32> = mel.axis_iter(Axis(1)).flat_map(|col| col.to_vec()).collect();
        self.session.run(
                vec![
                    ("mel", Value::from_array(([1, n_frames, n_mels], mel_flat)).unwrap()), 
                    ("f0", Value::from_array(([1, f0.len()], f0)).unwrap())
            ]).unwrap()
            .get("waveform").unwrap()
            .try_extract_tensor::<f32>().unwrap()
            .1
            .to_vec()
    }
}