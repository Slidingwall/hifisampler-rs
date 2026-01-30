use std::{collections::HashMap, path::PathBuf};
use anyhow::{anyhow, Result};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::{Array2, Array3, Array4, Zip};
use rustfft::num_complex::Complex;
use tracing::debug;
use crate::{consts, utils::stft::*};
const SEG_LENGTH: usize = 32 * consts::HOP_SIZE;
const OUTPUT_BIN: usize = consts::FFT_SIZE / 2 + 1;
fn validate_shape(actual: &[usize], expected: &[usize], name: &str) -> Result<()> {
    actual.eq(expected)
        .then_some(())
        .ok_or_else(|| anyhow!(
            "Invalid {} shape: expected {:?}, got {:?}",
            name, expected, actual
        ))
}
#[derive(Debug)]
pub struct HNSEPLoader {
    session: Session,
}
impl HNSEPLoader {
    pub fn new(model_path: &PathBuf) -> Result<Self> {
        Session::builder()
            .map_err(|_| anyhow!("Failed to create ONNX session builder"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|_| anyhow!("Failed to set session optimization level"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to load model from path {:?}: {}", model_path, e))
            .map(|session| Self { session })
    }
    pub fn run(&mut self, wave: &[f64]) -> Result<Array3<f64>> {
        let original_len = wave.len();
        if original_len == 0 {
            return Err(anyhow!("Input audio length cannot be zero"));
        }
        debug!("Starting HNSEP processing (original audio length: {})", original_len);
        let tl_pad = ((SEG_LENGTH * (((original_len + consts::HOP_SIZE - 1) / SEG_LENGTH + 1) - 1) 
            - (original_len + consts::HOP_SIZE)) / 2 / consts::HOP_SIZE) * consts::HOP_SIZE;
        let tr_pad = SEG_LENGTH * (((original_len + consts::HOP_SIZE - 1) / SEG_LENGTH + 1)) 
            - (original_len + consts::HOP_SIZE) - tl_pad;
        let mut x_padded = Vec::with_capacity(tl_pad + original_len + tr_pad);
        x_padded.extend(std::iter::repeat(0.0).take(tl_pad));
        x_padded.extend_from_slice(wave);
        x_padded.extend(std::iter::repeat(0.0).take(tr_pad));
        debug!(
            "Padded audio: left={}, right={}, total_len={}",
            tl_pad, tr_pad, x_padded.len()
        );
        let spec = stft_core(&x_padded, Some(consts::FFT_SIZE), Some(consts::HOP_SIZE))
            .map_err(|e| anyhow!("STFT failed: {}", e))?;
        let (n_freq, t_spec) = (spec.nrows(), spec.ncols());
        validate_shape(&[n_freq], &[OUTPUT_BIN], "STFT frequency bins")?;
        debug!("STFT completed: freq_bins={}, time_frames={}", n_freq, t_spec);
        let (real, imag): (Vec<_>, Vec<_>) = spec.iter().map(|&c| (c.re, c.im)).unzip();
        let target_t_spec = ((t_spec + 15) / 16) * 16;
        debug!("Padded time frames: original={}, target={}", t_spec, target_t_spec);
        let binding = Array4::from_shape_fn(
            (1, 2, OUTPUT_BIN, target_t_spec),
            |(_, c, f, t)| -> f32 {
                if t < t_spec {
                    match c {
                        0 => real[f + t * OUTPUT_BIN] as f32,
                        1 => imag[f + t * OUTPUT_BIN] as f32,
                        _ => 0.0,
                    }
                } else {
                    0.0
                }
            },
        );
        let onnx_input = binding;
        let input_value = Value::from_array(
            (
                onnx_input.shape().iter().map(|&d| d as i64).collect::<Vec<_>>(),
                onnx_input.as_slice().unwrap()
                    .to_vec()
            ),
        )
        .map_err(|e| anyhow!("Failed to create input tensor (shape {:?}): {}", onnx_input.shape(), e))?;
        debug!("Running ONNX inference (input shape {:?}, dtype=f32)", onnx_input.shape());
        let outputs = self.session.run(HashMap::from([("input", input_value)]))
            .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
        let (output_shape, output_data) = outputs.get("output")
            .ok_or_else(|| anyhow!("Missing output node 'output'"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract output tensor: {}", e))?;
        debug!("Output tensor: shape {:?}, data_len={}, dtype=f32", output_shape, output_data.len());
        let mask = Zip::from(
            &Array2::from_shape_fn((OUTPUT_BIN, t_spec), |(f, t)| {
                output_data[f + t * OUTPUT_BIN] as f64
            })
        )
        .and(
            &Array2::from_shape_fn((OUTPUT_BIN, t_spec), |(f, t)| {
                output_data[OUTPUT_BIN * target_t_spec + f + t * OUTPUT_BIN] as f64
            })
        )
        .map_collect(|&re, &im| Complex::new(re, im));
        debug!("Complex mask built (shape {:?})", mask.shape());
        let x_pred_padded = istft_core(
            &(spec * &mask),
            (t_spec - 1) * consts::HOP_SIZE + consts::FFT_SIZE,
            Some(consts::FFT_SIZE),
            Some(consts::HOP_SIZE),
        )
        .map_err(|e| anyhow!("ISTFT failed: {}", e))?;
        debug!("ISTFT completed (output_len={})", x_pred_padded.len());
        let target_end = tl_pad + original_len;
        if target_end > x_pred_padded.len() {
            return Err(anyhow!(
                "ISTFT output too short: required {} samples (start={}, end={}), got {}",
                target_end, tl_pad, target_end, x_pred_padded.len()
            ));
        }
        Array3::from_shape_vec(
            (1, 1, original_len),
            x_pred_padded[tl_pad..target_end].to_vec()
        )
        .map_err(|e| anyhow!("Failed to build final output: {}", e))
        .and_then(|output| Ok(output))
    }
}
#[cfg(test)]
mod tests {
    use ndarray::s;
    use super::*;
    use lazy_static::lazy_static;
    use std::path::PathBuf;
    lazy_static! {
        static ref MODEL_PATH: PathBuf = PathBuf::from("./model/hnsep_model.onnx");
    }
    const TEST_SAMPLE_RATE: usize = 44100;
    const TEST_DURATION: f64 = 2.0;
    fn generate_sine_audio(freq: f64, duration: f64, sample_rate: usize) -> Vec<f64> {
        (0..(duration * sample_rate as f64) as usize)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() * 0.5
            })
            .collect()
    }
    fn generate_noise_audio(duration: f64, sample_rate: usize) -> Vec<f64> {
        (0..(duration * sample_rate as f64) as usize)
            .map(|i| {
                let hash = (i as u64).wrapping_mul(11400714819323198485).to_le_bytes()[0] as f64;
                (hash / 255.0) - 0.5
            })
            .collect()
    }
    #[test]
    fn test_model_load_success() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        let loader = HNSEPLoader::new(&MODEL_PATH);
        assert!(loader.is_ok(), "Model load failed: {:?}", loader.as_ref().err());
        Ok(())
    }
    #[test]
    fn test_run_normal_sine_audio() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        let audio = generate_sine_audio(440.0, TEST_DURATION, TEST_SAMPLE_RATE);
        let original_len = audio.len();
        HNSEPLoader::new(&MODEL_PATH)?
            .run(&audio)
            .map_err(|e| anyhow!("Normal audio processing failed: {}", e))
            .and_then(|output| {
                assert_eq!(
                    output.shape(),
                    &[1, 1, original_len],
                    "Output shape mismatch: expected {:?}, got {:?}",
                    [1, 1, original_len],
                    output.shape()
                );
                Ok(())
            })
    }
    #[test]
    fn test_run_noise_audio() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        let audio = generate_noise_audio(TEST_DURATION, TEST_SAMPLE_RATE);
        let original_len = audio.len();
        HNSEPLoader::new(&MODEL_PATH)?
            .run(&audio)
            .map_err(|e| anyhow!("Noise audio processing failed: {}", e))
            .and_then(|output| {
                assert_eq!(output.shape(), &[1, 1, original_len]);
                Ok(())
            })
    }
    #[test]
    fn test_run_audio_length_exact_16x() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        use crate::consts::{HOP_SIZE, FFT_SIZE};
        let x_padded_len = (16 - 1) * HOP_SIZE + FFT_SIZE;
        let t1 = x_padded_len;
        let t_pad = SEG_LENGTH * ((t1 - 1) / SEG_LENGTH + 1) - t1;
        let tl_pad = (t_pad / 2 / HOP_SIZE) * HOP_SIZE;
        let tr_pad = t_pad - tl_pad;
        let original_len = x_padded_len - tl_pad - tr_pad;
        let audio = generate_sine_audio(
            880.0,
            original_len as f64 / TEST_SAMPLE_RATE as f64,
            TEST_SAMPLE_RATE
        );
        HNSEPLoader::new(&MODEL_PATH)?
            .run(&audio)
            .map_err(|e| anyhow!("16x length audio processing failed: {}", e))
            .and_then(|output| {
                assert_eq!(output.shape(), &[1, 1, original_len]);
                Ok(())
            })
    }
    #[test]
    fn test_run_audio_length_non_16x() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        use crate::consts::{HOP_SIZE, FFT_SIZE};
        let x_padded_len = (17 - 1) * HOP_SIZE + FFT_SIZE;
        let t1 = x_padded_len;
        let t_pad = SEG_LENGTH * ((t1 - 1) / SEG_LENGTH + 1) - t1;
        let tl_pad = (t_pad / 2 / HOP_SIZE) * HOP_SIZE;
        let tr_pad = t_pad - tl_pad;
        let original_len = x_padded_len - tl_pad - tr_pad;
        let audio = generate_sine_audio(
            660.0,
            original_len as f64 / TEST_SAMPLE_RATE as f64,
            TEST_SAMPLE_RATE
        );
        HNSEPLoader::new(&MODEL_PATH)?
            .run(&audio)
            .map_err(|e| anyhow!("Non-16x length audio processing failed: {}", e))
            .and_then(|output| {
                assert_eq!(output.shape(), &[1, 1, original_len]);
                Ok(())
            })
    }
    #[test]
    fn test_run_empty_audio() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        let err = HNSEPLoader::new(&MODEL_PATH)?.run(&[]).unwrap_err();
        assert!(err.to_string().contains("Input audio length cannot be zero"), "Incorrect error message: {}", err);
        Ok(())
    }
    #[test]
    fn test_stft_freq_bin_mismatch() -> Result<()> {
        use crate::consts::{FFT_SIZE, HOP_SIZE};
        let audio = generate_sine_audio(440.0, 0.5, TEST_SAMPLE_RATE);
        let mut x_padded = Vec::new();
        x_padded.extend(std::iter::repeat(0.0).take(1024));
        x_padded.extend_from_slice(&audio);
        x_padded.extend(std::iter::repeat(0.0).take(1024));
        let spec = stft_core(&x_padded, Some(FFT_SIZE), Some(HOP_SIZE))?;
        let wrong_spec = spec.slice(s![0..FFT_SIZE/2, ..]).to_owned();
        let err = validate_shape(&[wrong_spec.nrows()], &[OUTPUT_BIN], "STFT frequency bins").unwrap_err();
        assert!(err.to_string().contains("Invalid STFT frequency bins shape"), "Incorrect error message: {}", err);
        Ok(())
    }
    #[test]
    fn test_istft_output_too_short() -> Result<()> {
        if !MODEL_PATH.exists() {
            eprintln!("Warning: Test model file not found, skipping test");
            return Ok(());
        }
        let audio = generate_sine_audio(440.0, 0.1, TEST_SAMPLE_RATE);
        let original_len = audio.len();
        match HNSEPLoader::new(&MODEL_PATH)?.run(&audio) {
            Ok(output) => assert_eq!(output.shape(), &[1, 1, original_len]),
            Err(err) => {
                eprintln!("ISTFT short audio error: {}", err);
                assert!(
                    err.to_string().contains("ISTFT output too short") || err.to_string().contains("spectrum slice failed"),
                    "Incorrect error type: {}",
                    err
                );
            }
        }
        Ok(())
    }
}