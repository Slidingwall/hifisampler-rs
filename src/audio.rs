pub mod post_process;
mod base_coeff;
use crate::consts::SAMPLE_RATE;
use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use oxiaudio::{decode_file, encode_wav_with_config, downmix_to_mono, AudioBuffer, ChannelLayout, SampleFormat, WavBitDepth};
use std::path::{Path, PathBuf};
const COMMON_EXTENSIONS: Lazy<Vec<&str>> = Lazy::new(|| {
    vec!["wav", "flac", "ogg", "mp3", "aac"]
});
pub fn read_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let mut path = PathBuf::from(path.as_ref());
    if !path.exists() {
        let found = COMMON_EXTENSIONS.iter().any(|&ext| {
            path.set_extension(ext);
            path.exists()
        });
        if !found {
            return Err(anyhow!("no supported audio file found (tried extensions: {:?})", COMMON_EXTENSIONS));
        }
    }
    let buf = decode_file(&path).map_err(|e| anyhow!("audio decode failed: {e}"))?;
    let buf = if buf.sample_rate == SAMPLE_RATE {
        buf
    } else {
        oxiaudio::dsp::resample(&buf, SAMPLE_RATE).map_err(|e| anyhow!("resample failed: {e}"))?
    };
    Ok(downmix_to_mono(&buf).samples)
}
pub fn write_audio<P: AsRef<Path>>(path: P, audio: Vec<f32>) -> Result<()> {
    let buf = AudioBuffer::<f32> {
        samples: audio,
        sample_rate: SAMPLE_RATE,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    encode_wav_with_config(&buf, path.as_ref(), WavBitDepth::I16)
        .map_err(|e| anyhow!("WAV encode failed: {e}"))?;
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
                write_audio(&out_path, audio).expect("Write failed");
                println!("Write time: {:.2?}", now.elapsed());
            } else {
                println!("File not found: {:?} (skipped)", path.as_os_str());
            }
        }
    }
}
