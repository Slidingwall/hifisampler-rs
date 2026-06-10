pub mod post_process;
mod base_coeff;
use crate::consts::SAMPLE_RATE;
use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use once_cell::sync::Lazy;
use rubato::{Resampler, SincFixedIn, WindowFunction, SincInterpolationParameters, SincInterpolationType};
use std::{fs::File, path::{Path, PathBuf}};
use symphonia::{
    core::{
        audio::{SampleBuffer, SignalSpec},
        io::MediaSourceStream,
        probe::Hint,
    },
    default::{get_codecs, get_probe},
};
const I16_MAX: f32 = i16::MAX as f32;
static COMMON_EXTENSIONS: Lazy<Vec<&str>> = Lazy::new(|| {
    vec!["wav", "flac", "ogg", "mp3", "aac"]
});
fn resample_audio(audio: &[f32], in_sr: u32, out_sr: u32) -> Result<Vec<f32>> {
    let ratio = out_sr as f64 / in_sr as f64; // Due to rubato's API, we need to use f64 for the ratio
    let mut res = Vec::with_capacity((audio.len() as f64 * ratio).ceil() as usize);
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 64,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::Hann,
        },
        256,
        1,
    )?;
    let mut input = Vec::with_capacity(256);
    for chunk in audio.chunks(256) {
        input.clear();
        input.extend_from_slice(chunk);
        let proc_res = resampler.process(&[&input], None)?;
        let output = proc_res.get(0).unwrap();
        res.extend_from_slice(output);
    }
    let final_proc_res = resampler.process(&[&[]], None)?;
    let final_output = final_proc_res.get(0).unwrap();
    res.extend_from_slice(final_output);
    Ok(res)
}
pub fn read_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let mut path = PathBuf::from(path.as_ref());
    if !path.exists() {
        let found = COMMON_EXTENSIONS.iter().find(|&&ext| {
            path.set_extension(ext);
            path.exists()
        });
        if found.is_none() {
            return Err(anyhow!("No supported audio file found (tried extensions: {:?})", COMMON_EXTENSIONS));
        }
    }
    let source = File::open(&path)?;
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut probed = get_probe().format(&Hint::new(), mss, &Default::default(), &Default::default())?;
    let track = probed.format.default_track().ok_or_else(|| anyhow!("No audio track found"))?;
    let spec = SignalSpec {
        channels: track.codec_params.channels.unwrap(),
        rate: track.codec_params.sample_rate.unwrap(),
    };
    let channels = spec.channels.count();
    let channels_f32 = channels as f32;
    let mut decoder = get_codecs().make(&track.codec_params, &Default::default())?;
    let mut audio = Vec::with_capacity(409600);
    let mut sample_buf = SampleBuffer::<f32>::new(4096, spec);
    let track_id = track.id;
    while let Ok(packet) = probed.format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let Some(decoded) = decoder.decode(&packet).ok() else {
            continue;
        };
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();
        channels.eq(&1).then(|| audio.extend_from_slice(samples))
            .unwrap_or_else(|| audio.extend(samples.chunks(channels).map(|frame| frame.iter().sum::<f32>() / channels_f32)));
    }
    if spec.rate == SAMPLE_RATE {
        Ok(audio)
    } else {
        resample_audio(&audio, spec.rate, SAMPLE_RATE)
    }
}
pub fn write_audio<P: AsRef<Path>>(path: P, audio: &[f32]) -> Result<()> {
    let mut writer = WavWriter::new(
        File::create(path.as_ref())?,
        WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int
        },
    )?;
    for &s in audio {
        writer.write_sample((s * I16_MAX) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::{read_audio, write_audio};
    use std::{path::Path, time::Instant};
    #[test]
    fn test_read_write() {
        let test_paths = ["test/01.wav", "test/pjs001.wav"]
            .iter().map(Path::new).collect::<Vec<_>>();
        for path in test_paths {
            println!("Testing: {:?}", path.as_os_str());
            let out_path = path.with_extension("out.wav");
            let now = Instant::now();
            if path.exists() {
                let audio = read_audio(path).expect("Read failed");
                println!("Read time: {:.2?}", now.elapsed());
                write_audio(&out_path, &audio).expect("Write failed");
                println!("Write time: {:.2?}", now.elapsed());
            } else {
                println!("File not found: {:?} (skipped)", path.as_os_str());
            }
        }
    }
}