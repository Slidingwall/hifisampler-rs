use std::{collections::HashMap, path::PathBuf};
use anyhow::{anyhow, Result};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::Array2;
#[derive(Debug)]
pub struct HiFiGANLoader {
    session: Session,
}
impl HiFiGANLoader {
    pub fn new(model_path: &PathBuf) -> Result<Self> {
        Session::builder()
            .map_err(|_| anyhow!("Failed to create ONNX session builder"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|_| anyhow!("Failed to set graph optimization level (Level3)"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to load model from path {:?}: {}", model_path, e))
            .map(|session| Self { session })
    }
    pub fn run(&mut self, mel: Array2<f64>, f0: &[f64]) -> Result<Vec<f64>> {
        let (n_mels, n_frames) = mel.dim();
        if n_frames != f0.len() {
            return Err(anyhow!(
                "Mel frame count ({}) does not match F0 length ({})",
                n_frames,
                f0.len()
            ));
        }
        let mel_vec_f32: Vec<f32> = mel
            .permuted_axes((1, 0))  
            .as_standard_layout()   
            .as_slice()             
            .ok_or_else(|| anyhow!("Mel array is not in contiguous memory layout"))?
            .iter()
            .map(|&x| x as f32)     
            .collect();
        let f0_vec_f32: Vec<f32> = f0
            .iter()
            .map(|&x| x as f32)     
            .collect();
        let mel_tensor = Value::from_array(
            ([1i64, n_frames as i64, n_mels as i64], mel_vec_f32)
        ).map_err(|_| anyhow!("Failed to create mel input tensor"))?;
        let f0_tensor = Value::from_array(
            ([1i64, f0.len() as i64], f0_vec_f32)
        ).map_err(|_| anyhow!("Failed to create F0 input tensor"))?;
        let inputs: HashMap<_, _> = [("mel", mel_tensor), ("f0", f0_tensor)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let audio_data_f64 = self.session.run(inputs)
            .map_err(|e| anyhow!("Model inference failed: {}", e))?
            .get("waveform")
            .ok_or_else(|| anyhow!("Output node 'waveform' not found in model"))?
            .try_extract_tensor::<f32>()
            .map_err(|_| anyhow!("Failed to extract waveform tensor (expected f32)"))?
            .1
            .into_iter()
            .map(|x| *x as f64)
            .collect();
        Ok(audio_data_f64)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_model_load() {
        let model_path = PathBuf::from("./model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx");
        let result = HiFiGANLoader::new(&model_path);
        assert!(result.is_ok(), "Model load failed: {:?}", result.unwrap_err());
    }
}