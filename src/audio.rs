pub mod post_process;
use crate::consts::SAMPLE_RATE;
use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
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
const I16_MAX: f64 = i16::MAX as f64;
fn resample_audio(audio: &[f64], in_sr: u32, out_sr: u32) -> Result<Vec<f64>> {
    let ratio = out_sr as f64 / in_sr as f64;
    let mut res = Vec::with_capacity((audio.len() as f64 * ratio).ceil() as usize);
    let mut resampler = SincFixedIn::<f64>::new(
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
    for chunk in audio.chunks(256) {
        let mut input = Vec::from(chunk);
        input.resize(256, 0.0);
        let proc_res = resampler.process(&[&input], None)?;
        let output = proc_res.get(0).unwrap();
        res.extend_from_slice(output);
    }
    let final_proc_res = resampler.process(&[&[]], None)?;
    let final_output = final_proc_res.get(0).unwrap();
    res.extend_from_slice(final_output);
    Ok(res)
}
pub fn read_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let mut path = PathBuf::from(path.as_ref());
    if !path.exists() {
        let common_extensions = ["wav", "flac", "ogg", "mp3", "aac"];
        let found = common_extensions.iter().find(|&&ext| {
            path.set_extension(ext);
            path.exists()
        });
        if found.is_none() {
            return Err(anyhow!(
                "No supported audio file found (tried extensions: {:?})",
                common_extensions
            ));
        }
    }
    let source = File::open(&path)?;
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut probed = get_probe()
        .format(&Hint::new(), mss, &Default::default(), &Default::default())?;
    let track = probed
        .format
        .default_track()
        .ok_or_else(|| anyhow!("No audio track found"))?;
    let spec = SignalSpec {
        channels: track.codec_params.channels.unwrap(),
        rate: track.codec_params.sample_rate.unwrap(),
    };
    let channels = spec.channels.count();
    let mut decoder = get_codecs()
        .make(&track.codec_params, &Default::default())?;
    let mut audio = Vec::with_capacity(409600);
    let mut sample_buf = SampleBuffer::<f64>::new(4096, spec);
    let track_id = track.id;
    while let Ok(packet) = probed.format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        if let Ok(decoded) = decoder.decode(&packet) {
            sample_buf.copy_interleaved_ref(decoded);
            let samples = sample_buf.samples();
            if channels == 1 {
                audio.extend_from_slice(samples);
            } else {
                audio.extend(samples.chunks(channels).map(|frame| {
                    frame.iter().sum::<f64>() / channels as f64
                }));
            }
        }
    }
    if spec.rate == SAMPLE_RATE {
        Ok(audio)
    } else {
        resample_audio(&audio, spec.rate, SAMPLE_RATE)
    }
}
pub fn write_audio<P: AsRef<Path>>(path: P, audio: &[f64]) -> Result<()> {
    let mut writer = WavWriter::new(
        File::create(path.as_ref())?,
        WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int
        },
    )?;
    audio.iter()
        .map(|&s| (s * I16_MAX) as i16)
        .try_for_each(|sample| writer.write_sample(sample))?;
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
            .iter()
            .map(Path::new)
            .collect::<Vec<_>>();
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