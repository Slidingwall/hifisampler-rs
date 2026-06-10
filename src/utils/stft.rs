use crate::consts::{FFT_SIZE, HOP_SIZE};
use ndarray::{ArrayView1, Array3, Axis ,parallel::prelude::*};
use once_cell::sync::Lazy;
use phastft::{planner::PlannerR2c32, r2c_fft_f32_with_planner};
use crate::utils::hann_window::HANN_WINDOW;
static FFT_PLANNER: Lazy<PlannerR2c32> = Lazy::new(|| PlannerR2c32::new(FFT_SIZE));
thread_local! {
    static REAL_BUF: std::cell::RefCell<[f32; FFT_SIZE]> = std::cell::RefCell::new([0.0; FFT_SIZE]);
    static RE_BUF: std::cell::RefCell<[f32; 1025]> = std::cell::RefCell::new([0.0; 1025]);
    static IM_BUF: std::cell::RefCell<[f32; 1025]> = std::cell::RefCell::new([0.0; 1025]);
}
pub fn stft_core(signal: &[f32]) -> Array3<f32> {
    let freq_bins = FFT_SIZE / 2 + 1;
    let n_frames = (signal.len() + HOP_SIZE - 1) / HOP_SIZE;
    let planner = &*FFT_PLANNER;
    let window = &HANN_WINDOW;
    let mut spec = Array3::zeros((2, freq_bins, n_frames));
    spec.axis_iter_mut(Axis(2))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame_idx, mut frame_view)| {
            let start = frame_idx * HOP_SIZE;
            let slice_end = (start + FFT_SIZE).min(signal.len());
            let slice_len = slice_end - start;
            REAL_BUF.with(|cell| {
                let mut real_input = cell.borrow_mut();
                for (i, (&s, &w)) in signal[start..slice_end].iter().zip(window.iter()).enumerate() {
                    real_input[i] = s * w;
                }
                for i in slice_len..FFT_SIZE {
                    real_input[i] = 0.0;
                }
                RE_BUF.with(|re_cell| {
                    IM_BUF.with(|im_cell| {
                        let mut spec_re = re_cell.borrow_mut();
                        let mut spec_im = im_cell.borrow_mut();
                        r2c_fft_f32_with_planner(&real_input[..], &mut spec_re[..], &mut spec_im[..], planner);
                        frame_view.row_mut(0).assign(&ArrayView1::from(&*spec_re));
                        frame_view.row_mut(1).assign(&ArrayView1::from(&*spec_im));
                    });
                });
            });
        });
    spec
}