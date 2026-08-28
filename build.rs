use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Windows-only: embed an application icon via the Windows SDK `rc.exe`.
    // On machines without the SDK (e.g. CI runners) this must fail *gracefully*
    // instead of panicking the build script.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Search a few common Windows SDK locations; skip if none exist.
    let rc_candidates = [
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\rc.exe",
        r"D:\Windows Kits\10\bin\10.0.22621.0\x64\rc.exe",
    ];
    let Some(rc) = rc_candidates.iter().copied().find(|p| std::path::Path::new(p).exists()) else {
        println!("cargo:warning=Windows SDK rc.exe not found; skipping icon embedding");
        return;
    };

    if std::fs::write(manifest_dir.join("icon.rc"), "icon ICON favicon.ico").is_err() {
        println!("cargo:warning=failed to write icon.rc; skipping icon embedding");
        return;
    }

    match Command::new(rc).current_dir(&manifest_dir).arg("icon.rc").status() {
        Ok(status) if status.success() => println!("cargo:rustc-link-arg=icon.res"),
        _ => println!("cargo:warning=rc.exe failed; skipping icon embedding"),
    }
}
