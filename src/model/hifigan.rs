use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::Value };
use ndarray::Array2;
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
        self.session.run(
            vec![
                ("mel", Value::from_array(mel.t().to_shape((1, n_frames, n_mels)).unwrap().to_owned()).unwrap()), 
                ("f0", Value::from_array(([1, f0.len()], f0)).unwrap())
            ]).unwrap().get("waveform").unwrap()
            .try_extract_tensor::<f32>().unwrap().1.to_vec()
    }
}