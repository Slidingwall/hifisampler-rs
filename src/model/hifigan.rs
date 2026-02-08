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
    pub fn run(&mut self, mel: Array2<f64>, f0: &[f64]) -> Vec<f64> {
        let (n_mels, n_frames) = mel.dim();
        let mel_f32: Vec<f32> = mel
            .axis_iter(Axis(1))
            .flat_map(|col| col) 
            .map(|&x| x as f32) 
            .collect();
        let f0_f32: Vec<f32> = f0.into_iter().map(|&x| x as f32).collect();
        let mel_tensor = Value::from_array(([1, n_frames as i64, n_mels as i64], mel_f32)).unwrap();
        let f0_tensor = Value::from_array(([1, f0.len() as i64], f0_f32)).unwrap();
        self.session.run(vec![("mel", mel_tensor), ("f0", f0_tensor)]).unwrap()
            .get("waveform").unwrap()
            .try_extract_tensor::<f32>().unwrap()
            .1
            .into_iter()
            .map(|x| *x as f64) 
            .collect()
    }
}