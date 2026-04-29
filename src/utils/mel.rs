use crate::{consts::FFT_SIZE, utils::{interp::spec_interp, mel_basis::MEL_BASIS_DATA}};
use ndarray::{Array2, Axis, azip};
const TARGET_BINS: usize = FFT_SIZE / 2 + 1;
pub fn mel(spec:&Array2<f32>,key_shift:f32)->Array2<f32>{
    let (inf, ot) = spec.dim();
    let mut mel_spec = Array2::zeros((128, ot));
    let target_time = ((ot-1)as f32 *4.).round() as usize +1;
    let mut process_mel = |data: &Array2<f32>| {
        azip!((mut row in mel_spec.axis_iter_mut(Axis(0)), filter in &MEL_BASIS_DATA) {
            row.iter_mut().enumerate().for_each(|(t, val)| {
                *val = filter.iter().filter(|&&(f,_)| f < TARGET_BINS).map(|&(f,w)| data[(f,t)]*w).sum();
            });
        });
    };
    if key_shift.abs() < 1e-6 {
        process_mel(spec);
    } else {
        let fs = (-key_shift /12.).exp2();
        let tf = (FFT_SIZE as f32 * fs).round() as usize /2 +1;
        let mut sf = spec_interp(spec,(tf,ot),Axis(0),|f| {
            let x = f as f32 *(inf as f32/tf as f32);
            (x.floor() as isize, x.fract())
        });
        sf = ndarray::concatenate(Axis(0),&[sf.view(),Array2::zeros((TARGET_BINS-sf.nrows(),ot)).view()]).unwrap();
        let amp_scale = FFT_SIZE as f32 / (FFT_SIZE as f32 * fs).round() as f32;
        sf.iter_mut().for_each(|v| *v = v.exp() * amp_scale);
        process_mel(&sf);
    }
    spec_interp(&mel_spec, (128, target_time), Axis(1), |t| {
        let x = t as f32 /4.;
        (x.floor() as isize, x.fract())
    })
}