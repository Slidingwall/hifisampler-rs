use anyhow::{anyhow, Result};
use regex::Regex;
use std::collections::HashMap;
#[inline(always)]
fn to_uint6(c: u8) -> u8 {
    match c {
        b'+' => 62,
        b'/' => 63,
        b'0'..=b'9' => c + 4,
        b'A'..=b'Z' => c - 65,
        b'a'..=b'z' => c - 71,
        _ => 0,
    }
}
#[inline(always)]
fn to_int12_stream<S: AsRef<str>>(b64: S) -> Vec<i16> {
    b64.as_ref()
        .as_bytes()
        .chunks_exact(2) 
        .map(|chunk| {
            let uint12 = (to_uint6(chunk[0]) as u16) << 6 | (to_uint6(chunk[1]) as u16);
            ((uint12 << 4) as i16) >> 4 
        })
        .collect()
}
pub fn pitch_string_to_cents(string: &str) -> Result<Vec<f64>> {
    let mut res = Vec::new();
    let parts: Vec<_> = string.split('#').collect();
    let mut idx = 0;
    while idx < parts.len() - 1 {
        let stream = to_int12_stream(parts[idx]);
        res.extend(stream);
        let rle = parts[idx+1].parse::<usize>()
            .map_err(|e| anyhow!("Invalid RLE '{}': {}", parts[idx+1], e))?;
        if rle > 0 {
            let last = res.last().copied()
                .ok_or_else(|| anyhow!("Empty pitch stream for '{}'", parts[idx]))?;
            res.extend(std::iter::repeat(last).take(rle));
        }
        idx += 2;
    }
    if idx < parts.len() {
        res.extend(to_int12_stream(parts[idx]));
    }
    Ok(res.into_iter()
        .map(|x| x as f64 / 100.0)
        .chain(std::iter::once(0.0))
        .collect())
}
#[inline(always)]
pub fn tempo_parser(arg: &str) -> Result<f64> {
    let tempo: f64 = arg[1..].parse()?;
    Ok(tempo)
}
pub fn pitch_parser(arg: &str) -> Result<i32> {
    if let Ok(v) = arg.parse::<i32>() {
        return Ok(v);
    }
    let (note_part, octave_part) = match arg.char_indices().nth(1) {
        Some((_, c)) => {
            if c == '#' {
                (&arg[0..2], &arg[2..])
            } else {
                (&arg[0..1], &arg[1..])
            }
        }
        None => return Err(anyhow!("Invalid pitch format")),
    };
    let note_val = match note_part {
        "C" => 0,
        "C#" => 1,
        "D" => 2,
        "D#" => 3,
        "E" => 4,
        "F" => 5,
        "F#" => 6,
        "G" => 7,
        "G#" => 8,
        "A" => 9,
        "A#" => 10,
        "B" => 11,
        _ => return Err(anyhow!("Invalid note")),
    };
    let octave = octave_part.parse::<i32>()? + 1;
    Ok(octave * 12 + note_val)
}
pub fn flag_parser(s: &str) -> Result<HashMap<String, Option<f64>>> {
    let input = s.replace('/', "");
    static SUPPORTED_FLAGS: &[&str] = &[
        "fe", "fl", "fo", "fv", "fp", "ve", "vo", "g", "t", "vl",
        "A", "B", "G", "P", "S", "p", "R", "D", "C", "Z", "Hv", "Hb", "Ht", "He", "HG"
    ];
    let re = Regex::new(&format!(r"({})([+-]?\d+(\.\d+)?)?", SUPPORTED_FLAGS.join("|")))?;
    let mut flags = HashMap::new();
    for cap in re.captures_iter(&input) {
        let flag = cap.get(1).unwrap().as_str().to_string();
        let value = cap.get(2).map(|m| m.as_str().parse::<f64>().ok()).flatten();
        flags.insert(flag, value); 
    }
    Ok(flags)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_string() {
        let test = "B7CPCVCVCTCQCNCICDB+B5B0BvBrBnBlBk#14#BjBF/++Y8k615d4p4f4l4y5G5f596e7B7l8H8n9D9Z9q9092919y9t9n9f9Y9Q9I9C898584858/9L9b9v+G+f+4/Q/m/5AIATAY#2#AWAUARAOALAHAFACABAA";
        let pitchbend = pitch_string_to_cents(&test).unwrap();
        pitchbend.iter().for_each(|p| println!("{}", p));
    }
    #[test]
    fn test_tempo() {
        let tempo = tempo_parser("!120").unwrap();
        assert_eq!(tempo, 120.);
    }
    #[test]
    fn test_pitch() {
        let pitch = pitch_parser("C4").unwrap();
        assert_eq!(pitch, 60);
        let pitch = pitch_parser("C5").unwrap();
        assert_eq!(pitch, 72);
        let pitch = pitch_parser("A4").unwrap();
        assert_eq!(pitch, 69);
    }
    #[test]
    fn test_parse_empty() -> Result<()> {
        let flags = flag_parser("")?;
        assert!(flags.is_empty()); 
        Ok(())
    }
    #[test]
    fn test_parse_with_values() -> Result<()> {
        let flags = flag_parser("B50Hv70fl0.5G")?;
        assert_eq!(flags.get("B"), Some(&Some(50.0)));
        assert_eq!(flags.get("Hv"), Some(&Some(70.0)));
        assert_eq!(flags.get("fl"), Some(&Some(0.5)));
        assert_eq!(flags.get("G"), Some(&None)); 
        Ok(())
    }
    #[test]
    fn test_parse_flag_without_value() -> Result<()> {
        let flags = flag_parser("GHeMe")?;
        assert_eq!(flags.get("G"), Some(&None));
        assert_eq!(flags.get("He"), Some(&None));
        assert_eq!(flags.get("Me"), Some(&None));
        Ok(())
    }
}