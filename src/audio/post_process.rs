use bs1770::{ChannelLoudnessMeter, gated_mean};
use ndarray::{Array2, Axis, azip};
use oxifft::Complex;
use crate::{
    consts::{FFT_SIZE, HOP_SIZE, HIFI_CONFIG, SAMPLE_RATE},
    utils::{stft::{stft_core, istft_core}, reflect_pad_1d}, 
};
pub fn pre_emphasis_base_tension(wave: &mut Vec<f64>, b: f64) {
    let orig_len = wave.len();
    let orig_max = wave.iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(1.0); 
    let padded_len = ((orig_len + HOP_SIZE - 1) / HOP_SIZE) * HOP_SIZE;
    wave.resize(padded_len, 0.0);
    let comp_spec = stft_core(&*wave, FFT_SIZE, HOP_SIZE);
    let mut spec_amp = Array2::zeros(comp_spec.dim());
    let mut spec_phase = Array2::zeros(comp_spec.dim());
    azip!((amp_val in &mut spec_amp, &c in &comp_spec) {
        *amp_val = c.norm();
    });
    azip!((phase_val in &mut spec_phase, &c in &comp_spec) {
        *phase_val = c.arg();
    });
    spec_amp.mapv_inplace(|x| x.max(1e-9).ln());
    spec_amp.axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(j, mut bin)| {
            let filter = (b * (1.0 - (SAMPLE_RATE as f64 * j as f64) / (FFT_SIZE / 1500 + 3000) as f64)).clamp(-2.0, 2.0);
            bin.iter_mut().for_each(|amp_db| *amp_db += filter);
        });
    let mut comp_spec_istft = Array2::from_elem((FFT_SIZE / 2 + 1, comp_spec.ncols()), Complex::zero());
    azip!((comp_val in &mut comp_spec_istft, &phase in &spec_phase, &amp_db in &spec_amp) {
        let amp = amp_db.exp(); 
        *comp_val = Complex::new(amp * phase.cos(), amp * phase.sin());
    });
    let mut filtered_wave = istft_core(&comp_spec_istft, wave.len(), FFT_SIZE, HOP_SIZE);
    let filtered_max = filtered_wave.iter()
        .map(|x| x.abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(1.0);
    let gain = (orig_max / filtered_max) * ((b / -15.0).max(0.0) + 1.0);
    wave.truncate(orig_len);
    wave.iter_mut()
        .zip(filtered_wave.drain(0..orig_len)) 
        .for_each(|(w, fw)| *w = fw * gain);
}
fn rms_db(audio: &[f64]) -> f64 {
    let sum_sq: f64 = audio.iter()
        .map(|&x| x * x)
        .sum();
    let rms = (sum_sq / audio.len() as f64).sqrt();
    if rms < 1e-10 {
        f64::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    }
}
fn linear_fade(length: usize, fade_in: bool, sample_rate: f64) -> Vec<f64> {
    let fade_len = ((0.2 * sample_rate) as usize).min(length / 4);
    let mut fade = Vec::with_capacity(length); 
    if fade_in {
        fade.extend(std::iter::repeat(1.0).take(length));
        for i in 0..fade_len {
            let val = (i as f64) / ((fade_len - 1).max(1)) as f64;
            fade[i] = val;
        }
    } else {
        fade.extend(std::iter::repeat(1.0).take(length - fade_len));
        for i in 0..fade_len {
            let val = (i as f64) / ((fade_len - 1).max(1)) as f64;
            fade.push(val);
        }
    }
    fade
}
pub fn loudness_norm(
    wave: &mut Vec<f64>,
    sample_rate: f64,
    target: f64,
    norm_strength: u8,
) {
    let orig_len = wave.len();
    if orig_len == 0 {
        return;
    }
    let min_len = (0.4 * sample_rate) as usize;
    let (val_start, val_end, need_restore) = if HIFI_CONFIG.trim_silence {
        let frame_len = (0.02 * sample_rate) as usize;
        let hop_len = (0.01 * sample_rate) as usize;
        if frame_len > orig_len {
            (0, orig_len, false)
        } else {
            let mut start = None;
            let mut end = 0;
            let max_i = orig_len.saturating_sub(frame_len);
            for i in (0..=max_i).step_by(hop_len) {
                if rms_db(&wave[i..i + frame_len]) > HIFI_CONFIG.silence_threshold {
                    start.get_or_insert(i);
                    end = i;
                }
            }
            match start {
                Some(s) => {
                    (s, ((end / hop_len + 1 + ((0.1 * sample_rate) as usize / hop_len)) * hop_len + frame_len)
                        .min(orig_len), true)
                }
                None => (0, orig_len, false),
            }
        }
    } else {
        (0, orig_len, false)
    };
    let val_len = val_end - val_start;
    if val_len == 0 {
        return;
    }
    if val_len < min_len {
        reflect_pad_1d(wave, 0, min_len - val_len);
    }
    let mut meter = ChannelLoudnessMeter::new(sample_rate as u32);
    meter.push(
        wave[val_start..(val_start + min_len.max(val_len)).min(wave.len())]
            .iter()
            .map(|&x| x as f32)
    );
    let measure = gated_mean(meter.into_100ms_windows().as_ref()).loudness_lkfs() as f64;
    let gain = 10.0f64.powf((target - measure) * norm_strength as f64 * 0.0005);
    wave[val_start..val_end]
        .iter_mut()
        .for_each(|x| *x *= gain);
    if need_restore {
        wave[0..val_start].iter_mut().for_each(|x| *x = 0.0);
        wave[val_end..orig_len].iter_mut().for_each(|x| *x = 0.0);
        let fade_out = linear_fade(val_len, false, sample_rate);
        wave[val_start..val_end]
            .iter_mut()
            .zip(fade_out.iter())
            .for_each(|(w, f)| *w *= f);
    }
    wave.truncate(orig_len);
    wave.iter_mut()
    .for_each(|x| *x = x.clamp(-1.0, 1.0));
}