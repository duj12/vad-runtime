# FSMN VAD 库

基于FSMN模型的语音活动检测(VAD)库，支持 ONNX Runtime CPU 和 RKNN NPU（RK3588）双后端推理。

## 特性

- 高精度：基于FSMN模型的VAD
- 流式处理：支持实时流式音频输入
- 轻量级：最小依赖，仅需要 ONNX Runtime 和 kaldi-native-fbank
- 双后端：支持 ONNX Runtime CPU 和 RKNN NPU（RK3588）
- 纯净 C API：易于集成到各种语言和框架

## 项目结构

```
vad-runtime/
├── include/               # 头文件
│   ├── vad.h              # 对外C API
│   ├── vad_model.h        # VAD模型抽象接口
│   ├── fsmn_vad_onnx.h    # ONNX后端实现
│   ├── fsmn_vad_rknn.h    # RKNN NPU后端实现
│   ├── e2e_vad.h          # 端点检测状态机（header-only）
│   └── common.h           # 公共定义与常量
├── src/                   # 源文件
│   ├── vad.cpp            # C API实现
│   ├── fsmn_vad_onnx.cpp  # ONNX Runtime推理核心
│   └── fsmn_vad_rknn.cpp  # RKNN NPU推理核心
├── third_party/
│   ├── kaldi-native-fbank/ # Fbank特征提取（源码）
│   └── onnxruntime/        # ONNX Runtime 1.16.3 预编译依赖（随仓库携带，离线可用）
│       ├── linux-x86_64/   #   Linux x86_64
│       ├── linux-aarch64/  #   Linux aarch64（Jetson 等）
│       └── android/        #   Android arm64-v8a / armeabi-v7a
├── prebuilt/              # 各平台预编译 libfsmn_vad.so（可直接取用）
├── model/                 # 模型文件
│   ├── model_quant.onnx   # ONNX量化模型
│   ├── am.mvn             # CMVN归一化参数（Kaldi文本格式）
│   └── config.yaml        # 模型配置
├── examples/
│   ├── demo_onnx.cpp      # ONNX后端使用示例
│   ├── demo_rknn.cpp      # RKNN后端使用示例
│   ├── xmov.raw           # 测试音频（16kHz, 16bit, 单声道）
│   └── xmov.wav           # 测试音频WAV格式
└── CMakeLists.txt         # CMake编译配置
```

## 快速开始

依赖（ONNX Runtime）已随仓库放在 `third_party/onnxruntime/` 下，**构建过程不需要联网**。

### 方式一：直接使用预编译库

`prebuilt/<platform>/libfsmn_vad.so` 已随仓库提供（Linux x86_64 / Linux aarch64）。
运行时把对应平台的 `libonnxruntime.so.1.16.3`（位于
`third_party/onnxruntime/<platform>/lib/`）放在 `libfsmn_vad.so` 同目录即可，
库的 RUNPATH 是 `$ORIGIN`。详见 [`prebuilt/README.md`](prebuilt/README.md)。

### 方式二：一键编译

```bash
make                  # 本机原生编译（x86_64 或 aarch64）
make ARCH=arm64       # 在 x86 上交叉编译 aarch64（需 g++-aarch64-linux-gnu）
make android          # Android arm64-v8a（需 Android NDK）
cd build && ./vad_example_onnx ../model ../examples/xmov.raw
```

### 方式三：手动 CMake

```bash
mkdir -p build && cd build
cmake .. -DENABLE_ONNX=ON -DENABLE_RKNN=OFF
cmake --build . -j"$(nproc)"
./vad_example_onnx ../model ../examples/xmov.raw
```

编译时 ONNX Runtime 会按 `CMAKE_SYSTEM_PROCESSOR` 自动选择：
`x86_64 → linux-x86_64`，`aarch64 → linux-aarch64`。可用 `-DONNXRUNTIME_DIR=<dir>`
指定其它 ONNX Runtime；只有显式加 `-DONNXRUNTIME_DOWNLOAD=ON` 才会联网下载。

### RK3588 Android（推荐 RKNN NPU）

完整流程请看：

- [`docs/ANDROID.md`](docs/ANDROID.md) — Android RKNN/ONNX 构建、adb 部署、调试
- [`docs/RKNN_GUIDE.md`](docs/RKNN_GUIDE.md) — RKNN 模型转换、PC 模拟器验证、Linux/Android 部署注意事项

```bash
export ANDROID_NDK=~/other/android-ndk-r19c
export RKNN_MODEL_ZOO_ROOT=~/workspace/rknn_model_zoo
bash tools/build_android_rknn.sh
bash tools/deploy_android_rknn.sh
```

## C API

对外接口见 [`include/vad.h`](include/vad.h)。示例程序在 [`examples/demo_onnx.cpp`](examples/demo_onnx.cpp) 与 [`examples/demo_rknn.cpp`](examples/demo_rknn.cpp)。

## 参考

- [FunASR](https://github.com/modelscope/FunASR) — 阿里达摩院语音识别工具包
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) — 纯C++ Fbank特征提取
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 跨平台推理引擎
