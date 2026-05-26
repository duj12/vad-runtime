# FSMN VAD 库

基于FSMN模型的语音活动检测(VAD)库，从原始 online-vad 推理代码剥离，去除 ffmpeg 等无关依赖，仅保留核心 VAD 功能。支持 ONNX Runtime CPU 推理，预留 RKNN NPU 加速接口。

## 特性

- 高精度：基于FSMN模型的VAD
- 流式处理：支持实时流式音频输入
- 轻量级：最小依赖，仅需要 ONNX Runtime 和 kaldi-native-fbank
- 双后端：支持 ONNX Runtime CPU 和 RKNN NPU（预留）
- 纯净 C API：易于集成到各种语言和框架

## 项目结构

```
vad-runtime/
├── include/               # 头文件
│   ├── vad.h              # 对外C API
│   ├── vad_model.h        # VAD模型抽象接口
│   ├── fsmn_vad_onnx.h    # ONNX后端实现
│   ├── fsmn_vad_rknn.h    # RKNN后端实现（预留）
│   ├── e2e_vad.h          # 端点检测状态机（header-only）
│   └── common.h           # 公共定义与常量
├── src/                   # 源文件
│   ├── vad.cpp            # C API实现
│   ├── fsmn_vad_onnx.cpp  # ONNX Runtime推理核心
│   └── fsmn_vad_rknn.cpp  # RKNN后端实现（预留）
├── third_party/
│   ├── kaldi-native-fbank/ # Fbank特征提取（源码）
│   └── onnxruntime/        # ONNX Runtime（需自行下载）
├── model/                 # 模型文件
│   ├── model_quant.onnx   # ONNX量化模型
│   ├── am.mvn             # CMVN归一化参数（Kaldi文本格式）
│   └── config.yaml        # 模型配置
├── examples/
│   ├── demo_onnx.cpp      # ONNX后端使用示例
│   ├── demo_rknn.cpp      # RKNN后端使用示例（预留）
│   ├── xmov.raw           # 测试音频（16kHz, 16bit, 单声道）
│   └── xmov.wav           # 测试音频WAV格式
└── CMakeLists.txt         # CMake编译配置
```

## 编译

### 依赖要求

- CMake 3.16+
- C++14 编译器
- ONNX Runtime 1.16.3（下载后放到 `third_party/onnxruntime/`）
- RKNN SDK（可选，用于RKNN后端）

### ONNX Runtime 下载

```bash
# x86_64 Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar xzf onnxruntime-linux-x64-1.16.3.tgz
mv onnxruntime-linux-x64-1.16.3 onnxruntime

# ARM64 (RK3588)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar xzf onnxruntime-linux-aarch64-1.16.3.tgz
mv onnxruntime-linux-aarch64-1.16.3 onnxruntime
```

### x86_64 Linux 编译

```bash
mkdir build && cd build
cmake .. -DENABLE_RKNN=OFF
make -j$(nproc)
```

### RK3588 交叉编译

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DENABLE_ONNX=ON \
    -DENABLE_RKNN=ON \
    -DRKNN_SDK_DIR=/path/to/rknn_sdk
make -j4
```

### 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_ONNX` | ON | 启用ONNX Runtime后端 |
| `ENABLE_RKNN` | OFF | 启用RKNN NPU后端 |
| `BUILD_EXAMPLES` | ON | 编译示例程序 |
| `BUILD_SHARED_LIBS` | ON | 编译为动态库 |

## 快速开始

### 运行示例

```bash
cd build
LD_LIBRARY_PATH=. ./vad_example_onnx /path/to/model /path/to/audio.pcm
```

示例输出：

```
VAD initialized, backend: 1
Read 83968 samples, duration: 5.248s
/path/to/audio.pcm: [[0,-1],[-1,4270]]
```

输出格式：`<音频路径>: [[start_ms,end_ms],[start_ms,end_ms],...]`

在线模式下，分段以部分结果形式返回：
- `[start, -1]` — 语音开始于 `start` ms，尚未结束
- `[-1, end]` — 上段未结束的语音在 `end` ms 结束

### C API

```c
#include "vad.h"

// 初始化
void* handle = vad_init("./model", 16000, VAD_BACKEND_ONNX);

// 处理音频块
int* segments = NULL;
int segments_len = 0;
int ret = vad_process(handle, pcm_data, pcm_len, is_final, &segments, &segments_len);
for (int i = 0; i < segments_len; ++i) {
    int start_ms = segments[i * 2];
    int end_ms = segments[i * 2 + 1];
    printf("[%d,%d]\n", start_ms, end_ms);
}
vad_free_segments(segments);

// 重置状态，处理新音频流
vad_reset(handle);

// 释放资源
vad_destroy(handle);
```

## API 说明

### 后端类型

```c
typedef enum {
    VAD_BACKEND_AUTO = 0,  // 自动选择（目前等同ONNX）
    VAD_BACKEND_ONNX = 1,  // ONNX Runtime CPU
    VAD_BACKEND_RKNN = 2   // RKNN NPU（预留）
} VadBackendType;
```

### 初始化

```c
/**
 * @brief 初始化VAD模型
 * @param model_dir 模型目录，需包含 model_quant.onnx, am.mvn, config.yaml
 * @param sample_rate 采样率，固定16000
 * @param backend 后端类型
 * @return VAD句柄，失败返回NULL
 */
void* vad_init(const char* model_dir, int sample_rate, VadBackendType backend);
```

### 处理音频

```c
/**
 * @brief 处理音频数据
 * @param handle VAD句柄
 * @param pcm_data 16bit PCM数据（16kHz, 单声道）
 * @param pcm_len 采样点个数
 * @param is_final 是否为最后一段
 * @param out_segments 输出分段数组 [start_ms,end_ms,start_ms,end_ms,...]
 * @param out_segments_len 分段个数
 * @return 0成功，其他失败
 */
int vad_process(void* handle, const short* pcm_data, int pcm_len,
                bool is_final, int** out_segments, int* out_segments_len);
```

### 其他

```c
int vad_reset(void* handle);
int vad_destroy(void* handle);
VadBackendType vad_get_backend(void* handle);
void vad_free_segments(int* segments);
```

## 模型文件

模型目录需包含以下文件：

```
model/
├── model_quant.onnx   # ONNX量化模型（5输入5输出）
├── am.mvn             # CMVN参数（Kaldi文本格式）
└── config.yaml        # 模型配置
```

### config.yaml 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `speech_noise_thres` | 语音/噪声判断阈值 | 0.6 |
| `max_end_silence_time` | 尾部静音超时(ms) | 800 |
| `max_single_segment_time` | 最大单段长度(ms) | 20000 |
| `decibel_thres` | 分贝阈值 | -100.0 |
| `snr_thres` | 信噪比阈值 | -100.0 |

### am.mvn 格式

Kaldi 文本格式，包含 `<AddShift>`（负均值）和 `<Rescale>`（缩放因子）两部分，每部分 400 个 float 值（80维fbank × 5帧拼接）。

## RKNN 后端（预留）

RKNN NPU后端接口已预留，当前实现为 stub。如需启用：

1. 安装 [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
2. 将 ONNX 模型转换为 RKNN 格式
3. 设置 `-DENABLE_RKNN=ON -DRKNN_SDK_DIR=/path/to/rknn_sdk` 编译

## 参考

- [FunASR](https://github.com/modelscope/FunASR) — 阿里达摩院语音识别工具包
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) — 纯C++ Fbank特征提取
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 跨平台推理引擎
