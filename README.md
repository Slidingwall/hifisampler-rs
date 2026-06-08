# hifisampler-rs 
(WIP) A Rust alternative backends of UTAU resampler hifisampler.  
（仍在开发）UTAU重采样器Hifisampler的Rust替代后端。

The goal is to unofficially migrate hifisampler to Rust, just like its upstream StrayCat, in order to reduce dependencies, software size, or improve speed. This project is based on [StrayCat-rs](https://github.com/UtaUtaUtau/straycat-rs) and [hifisampler](https://github.com/openhachimi/hifisampler), and some code from [StrayCrab](https://github.com/layetri/straycrab) was also referenced.  
本项目的愿景是将hifisampler非官方地与其上游项目StrayCat一样迁移至rust语言中，以减少其依赖项、软件体积或提升其速度。本项目基于[StrayCat-rs](https://github.com/UtaUtaUtau/straycat-rs)与[hifisampler](https://github.com/openhachimi/hifisampler)，同时还参考了部分[StrayCrab](https://github.com/layetri/straycrab)的代码。  

> [!CAUTION]
> **Please note that this project is not yet completed. Its performance may be inferior to the original Python version.**  
> **请注意，本项目没有开发完成。其性能可能不如原始的Python版本。**  
> 
> Due to certain reasons, I may not be able to allocate time to this project for a long period. Therefore, I have decided to publicly release the semi-finished code in advance, hoping to receive support from other developers.  
> 由于一些原因，我可能很长一段时间无法分出精力在此项目上。因此我决定先行公开半成品的代码，以期待来自其他开发者的支持。  

## Difference 差异
During the refactoring process, this project has also made several modifications to the original workflow of hifisampler. Due to differences in programming languages, libraries, and errors introduced while optimizing the computational pipeline, this project may cause discrepancies against the processing results of the original Python implementation.  
在改写的过程中，本项目还对hifisampler原有的流程进行了一些改动。由于所采用的编程语言和库不同，以及在优化计算流程中引入的误差，本项目的处理结果可能与原来的Python实现有所不同。  
 
Notable changes include:  
主要改动包括：

| Item | Python Version | Rust Version |
|:---:|:---:|:---:|
| Support model formats / 支持的模型格式| PyTorch and ONNX| Only ONNX |
| Configuration file / 配置文件 | config.py, with vocoder‑hardcoded parameters / 含与声码器硬绑定的参数 | hificonfig.ini, without vocoder‑hardcoded parameters / 不含与声码器硬绑定的参数 |
| Features cache / 特征缓存 | 32 bit float npz / 32位浮点NPZ | Quantized 16 bit unsigned integer binary / 量化的16位无符号整数二进制 |
| HNSEP cache / HNSEP缓存| 32 bit float PyTorch tensor / 32位浮点PyTorch张量 | Quantized 16 bit unsigned integer binary / 量化的16位无符号整数二进制 |
| STFT | 3 times for HNSEP model, mel spec and tension / 对HNSEP模型、mel谱与张力处理进行3次STFT | Only once for all processes / 仅对所有流程进行1次STFT |
| ISTFT | 2 times for HNSEP model and tension / 对HNSEP模型和张力处理进行2次ISTFT | No ISTFT / 无ISTFT | 
| Logarithmic conversion / 对数转换 | 2 times for mel spec and tension / 对mel谱和张力处理进行2次对数转换 | Only once for mel spec / 仅对mel谱进行1次对数转换 |
| Exponential conversion / 指数转换 | 1 times for tension / 对张力处理进行1次指数转换 | No exponential conversion / 无指数转换 |

## Using 使用

> [!WARNING]
> Since this project will ultimately be compiled into an executable file, the server-side will be renamed `hifiserver-rust` to distinguish it from the client(`hifisampler`) actually called by UTAU/OpenUTAU. To avoid such confusion in UTAU/OpenUTAU, **it is not recommended to place the server-side and client-side together in the `resampler` folder**.  
> 由于本项目最终会编译为可执行文件，为了与UTAU/OpenUTAU实际调用的客户端(`hifisampler`)作区分，服务器端会被重命名为`hifiserver-rust`。为了避免在UTAU/OpenUTAU中混淆，**不建议您将服务器端与客户端一起放入`resampler`文件夹中**。  

The client is as same as hifisampler. If you are using macOS or Linux, you can temporarily use the client of [StrayCat-server](https://github.com/Astel123457/straycat-server/releases/tag/release).  
客户端与原hifisampler的客户端一致。如果您使用macOS或者Linux，您可以暂时使用[StrayCat-server](https://github.com/Astel123457/straycat-server/releases/tag/release)的客户端。  

`hificonfig.ini` is the server-side configuration file.  
`hificonfig.ini`是服务器端的配置文件。    

You need the following ONNX models:  
您需要以下 ONNX 模型：
- [pc-nsf-hifigan](https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02) (or [Kouon Vocoder](https://github.com/Kouon-Project/Kouon_Vocoder/releases/tag/V2.0.0), the `pc-mini-nsf` version)
- [hnsep(Chinese link)](https://wwbpi.lanzouv.com/igZpn3g2eqxa), a temporary version exported by myself. / 一个我导出的临时版本。

They should be located in the `./model/` folder within the same directory as the server-side, but you can also customize the model's location by modifying `hificonfig.ini`.  
它们应位于与服务器端同目录的`./model/`文件夹内，但您也可以通过修改`hificonfig.ini`来自定义模型的位置。  

## How to compile
 **Note**: By the nature of an UTAU resampler, it is only ideal to build this program in Windows.
 1. Install [rustup](https://rustup.rs/).
 2. Decide whether you want to build with the icon.
    - Build with icon:
        1. Install [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/).
        2. Locate `rc.exe`. It is usually in `C:\Program Files (x86)\Windows Kits\10\bin\<version number>\x64\rc.exe`
        3. Replace the location for `rc.exe` in the build script `build.rs`.
        4. Build with `cargo build -r`
    - Build without icon:
        1. Delete the build script `build.rs`.
        2. Build with `cargo build -r`
 
 I highly encourage building in the other platforms as those builds can be used in [OpenUtau.](https://github.com/stakira/OpenUtau) Build steps for Mac/Linux should be similar, just follow build without icon skipping step 1.

## Supported Flags 支持的Flags

Basically same as Python version.  
与Python版本基本一致。

|Flags|Describe|Range|Default|
|:---:|:---:|:---:|:---:|
|**g**|Gender / formants<br/>性别 / 共振峰|-600~600|0|
|**Hb**|Breath / noise<br/>气息 / 噪波|0~500|100|
|**Hv**|Voice / harmonic<br/>发声 / 谐波|0~150|100|
|**HG**|Vocal fry / growl<br/>怒音 / 嘶吼|0~100|0|
|**P**[^1]|Note level loudness normalize<br/>音符级响度标准化|0~100|100|
|**t**|Pitch shift<br/>音高偏移|-1200~1200|0|
|**Ht**|Tension<br/>张力|-100~100|0|
|**A**|Amplitude<br/>振幅|-100~100|0|
|**G**|Force regenerate cache<br/>强制重生成缓存|bool|false|
|**He**[^2]|Loop mode<br/>循环模式|bool|false|

[^1]: Only effective when `wave_norm` is set to `true` in `hificonfig.ini`, targeting -16 LUFS.  
      仅当`hificonfig.ini`中，`wave_norm`为`true`时有效，以 -16 LUFS 为基准。  
[^2]: Globally enabled when `loop_mode` is set to `true` in `hificonfig.ini`.  
      当`hificonfig.ini`中，`loop_mode`为`true`时全局启用。  

You can download OpenUTAU resampler manifest file from [名無絃](https://bowlroll.net/file/335049).  
您可以下载[名無絃](https://bowlroll.net/file/335049)提供的OpenUTAU重采样器配置文件。  

