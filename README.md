# FSMN VAD 库

基于FSMN模型的语音活动检测(VAD)库，支持 ONNX Runtime CPU 推理，预留 RKNN NPU 加速接口。

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

示例输出（xmov.raw，5.248s，语音 0~4270ms，首帧 360ms + 后续 60ms）：

```
VAD initialized, backend: 1
Read 83968 samples, duration: 5248ms

   0ms: BEGIN    0.014    [0,-1]
 360ms: CONTINUE    0.013
4800ms: END    0.007    [-1,4270]
4860ms: NONE    0.007

=== Summary ===
RTF: 0.007

=== 结果说明 ===
语音起始: WindowDetector(200ms窗口)需要足够语音帧才能确认BEGIN
          首帧360ms可直接确认(sil_to_speech_thres=150ms), 0ms输出BEGIN
          若用固定60ms步长, 需第6个chunk(~360ms)才输出[0,-1],
          前5个chunk为NONE, 实时场景下会被丢弃
语音结束: max_end_silence_time=800ms, 需累积650ms连续静音才确认END
          实际语音止于4270ms, 但END在4800ms输出, 延迟约530ms
```

每行为一个状态变化点，跳过连续相同的状态。时间戳为当前 chunk 实时时间，segment（如 `[0,-1]`）随行显示供参考。

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
| `max_single_segment_time` | 最大单段长度(ms) | 60000 |
| `decibel_thres` | 分贝阈值 | -100.0 |
| `snr_thres` | 信噪比阈值 | -100.0 |

### am.mvn 格式

Kaldi 文本格式，包含 `<AddShift>`（负均值）和 `<Rescale>`（缩放因子）两部分，每部分 400 个 float 值（80维fbank × 5帧拼接）。

## RKNN 后端（预留）

RKNN NPU后端接口已预留，当前实现为 stub。如需启用：

1. 安装 [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2)
2. 将 ONNX 模型转换为 RKNN 格式
3. 设置 `-DENABLE_RKNN=ON -DRKNN_SDK_DIR=/path/to/rknn_sdk` 编译

## 流式输出说明

VAD 内部通过 `WindowDetector`（200ms 窗口）判断语音起止，检测结果存在固有延迟：

- **语音起始（BEGIN）**：需要累积足够多语音帧才能确认，首帧必须超过窗口大小才能立即响应
- **语音结束（END）**：需要累积足够多连续静音帧才能确认（`max_end_silence_time` 参数控制）

### 示例输出

以 `xmov.raw`（5.248s，语音 0~4270ms）为例，首帧 360ms + 后续 60ms 步长：

```
   0ms: BEGIN    0.013    [0,-1]        ← 首帧360ms一次确认, 0ms输出
 360ms: CONTINUE    0.012
4800ms: END    0.006    [-1,4270]       ← 实际语音止于4270ms, 延迟~530ms
4860ms: NONE    0.006
```

### 语音起始延迟

- `WindowDetector` 窗口 = 200ms，`sil_to_speech_thres` = 150ms
- 若固定 60ms 步长输入，首个 chunk 只有 6 帧 LFR 输出，WindowDetector 无法确认
- 通常需要 ~6 个 chunk（~360ms）才返回 `[0, -1]`，前 5 个 chunk 的状态为 NONE
- **实时场景问题**：前几帧的 NONE 直接输出后，真实的 BEGIN（时间戳 0ms）到来时已错过，导致语音头被吃掉
- **解决**：首帧使用大块（如 360ms）确保一次确认，或预先填充静音让内部缓存就绪

### 语音结束延迟

- `max_end_silence_time = 800ms`（config.yaml），需累积 `800-150=650ms` 连续静音才确认结束
- `speech_to_sil_thres = 150ms` 是 WindowDetector 的语音→静音阈值
- END 输出时刻比实际语音结束位置滞后约 `max_end_sil_time - speech_to_sil_thres`，但 segment 的 end_ms 通过 lookback 机制回溯到正确位置
- 如需减少延迟，可调小 `max_end_silence_time`

### 状态定义

| 状态 | 数值 | 含义 | 对应 segment |
|------|------|------|-------------|
| NONE | 0 | 非语音段 | 无 |
| BEGIN | 1 | 检测到语音起始 | `[start_ms, -1]` |
| CONTINUE | 2 | 持续语音中 | 无 |
| END | 3 | 检测到语音结束 | `[-1, end_ms]` |

输出跳过连续相同状态，仅状态变化时打印。时间戳为当前 chunk 的实时时间。

## 参考

- [FunASR](https://github.com/modelscope/FunASR) — 阿里达摩院语音识别工具包
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) — 纯C++ Fbank特征提取
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 跨平台推理引擎
