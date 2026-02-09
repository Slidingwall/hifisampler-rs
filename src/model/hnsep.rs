use std::path::PathBuf;
use ort::{ session::{Session, builder::GraphOptimizationLevel}, value::Value };
use ndarray::{Array4, azip, s};
use oxifft::Complex;
use crate::{consts::{FFT_SIZE, HOP_SIZE}, utils::stft::*};
const SEG_LENGTH: usize = 32 * HOP_SIZE;
const OUTPUT_BIN: usize = FFT_SIZE / 2 + 1;
#[derive(Debug)]
pub struct HNSEPLoader {
    session: Session,
}
impl HNSEPLoader {
    pub fn new(model_path: &PathBuf) -> Self {
        Self {
            session: Session::builder().unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
                .commit_from_file(model_path).unwrap()
        }
    }
    pub fn run(&mut self, wave: &[f32]) -> Vec<f32> {
        let orig_len = wave.len();
        let total_pad = SEG_LENGTH * (((orig_len + HOP_SIZE - 1) / SEG_LENGTH) + 1) - (orig_len + HOP_SIZE); 
        let left = (total_pad / 2 / HOP_SIZE) * HOP_SIZE; 
        let mut x_pad = Vec::with_capacity(orig_len + total_pad);
        x_pad.resize(left, 0.0);
        x_pad.extend_from_slice(wave);
        x_pad.resize(orig_len + total_pad, 0.0);
        let mut spec = stft_core(&x_pad, FFT_SIZE, HOP_SIZE);
        let t_spec = spec.ncols();
        let target_t_spec = ((t_spec + 15) / 16) * 16;
        let mut arr4 = Array4::from_elem((1, 2, OUTPUT_BIN, target_t_spec), 0.0);
        azip!((index (c, f, t), val in &mut arr4.slice_mut(s![0, .., .., ..t_spec])) {
            *val = if c as usize == 0 { spec[(f, t)].re } else { spec[(f, t)].im };
        });
        let input_value = Value::from_array(
            (
                [1, 2, OUTPUT_BIN, target_t_spec],
                arr4.into_raw_vec_and_offset().0
            ),
        ).unwrap();
        let outputs = self.session.run(vec![("input", input_value)]).unwrap();
        let output_data = outputs.get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap()
            .1;
        let bin_idx = OUTPUT_BIN * target_t_spec;
        azip!((index (f, t), sc_val in &mut spec) {
                let t_offset = t * OUTPUT_BIN;
                *sc_val *= Complex::new(
                    output_data[f + t_offset],
                    output_data[bin_idx + f + t_offset]
                );
            });
        x_pad.clear();
        x_pad = istft_core(
            &spec,
            (t_spec - 1) * HOP_SIZE + FFT_SIZE,
            FFT_SIZE,
            HOP_SIZE,
        );
        x_pad.drain(0..left);
        x_pad.truncate(orig_len);
        x_pad
    }
}
