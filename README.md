# FSMN 离线/在线 VAD 库

一个轻量级的语音活动检测(VAD)库，基于FSMN模型，支持ONNX Runtime CPU推理和RK3588 NPU硬件加速，输入为16kHz采样率的PCM原始音频，输出语音段的起止时间。

## 特性

- 🎯 高精度：基于FSMN模型的VAD，准确率高
- ⚡ 高性能：支持RK3588 NPU硬件加速，推理速度快
- 🔄 流式处理：支持实时流式音频输入
- 📦 轻量级：最小依赖，仅需要ONNX Runtime或RKNN SDK
- 🚀 多平台：支持x86_64 Linux/Windows和ARM64 Linux(RK3588)

## 项目结构

```
vad-runtime/
├── include/            # 头文件
│   ├── vad.h           # 对外C/C++ API
│   ├── vad_model.h     # 抽象模型接口
│   ├── fsmn_vad_onnx.h # ONNX后端实现
│   ├── fsmn_vad_rknn.h # RKNN后端实现
│   ├── e2e_vad.h       # 端点检测算法
│   └── common.h        # 公共定义
├── src/                # 源文件
│   ├── vad.cpp         # API实现
│   ├── fsmn_vad_onnx.cpp
│   ├── fsmn_vad_rknn.cpp
│   └── e2e_vad.cpp
├── third_party/        # 第三方依赖
│   └── kaldi-native-fbank # Fbank特征提取
├── model/              # 模型文件
│   ├── onnx/           # ONNX格式模型
│   └── rknn/           # RKNN格式模型
├── tools/              # 工具脚本
│   └── convert_onnx_to_rknn.py # ONNX转RKNN模型脚本
├── examples/           # 示例代码
│   ├── demo_onnx.cpp   # ONNX后端示例
│   └── demo_rknn.cpp   # RKNN后端示例
└── CMakeLists.txt      # 编译配置
```

## 编译

### 依赖要求

- CMake 3.16+
- C++14 编译器
- ONNX Runtime 1.16+ (可选，用于ONNX后端)
- RKNN SDK 1.4+ (可选，用于RKNN后端)

### x86平台编译(ONNX后端)

```bash
mkdir build && cd build
cmake .. -DENABLE_ONNX=ON -DENABLE_RKNN=OFF
make -j4
```

### RK3588交叉编译(ONNX+RKNN后端)

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

- `ENABLE_ONNX`: 是否启用ONNX Runtime后端，默认ON
- `ENABLE_RKNN`: 是否启用RKNN NPU后端，默认OFF
- `BUILD_EXAMPLES`: 是否编译示例程序，默认ON
- `BUILD_SHARED_LIBS`: 是否编译动态库，默认ON
- `ONNXRUNTIME_DIR`: ONNX Runtime安装目录
- `RKNN_SDK_DIR`: RKNN SDK安装目录(启用RKNN时必填)

## 快速开始

### C API 使用示例

```c
#include "vad.h"

int main() {
    // 初始化VAD，自动选择后端
    void* handle = vad_init("./model", 16000, VAD_BACKEND_AUTO);
    if (!handle) {
        // 初始化失败处理
        return -1;
    }

    // 处理音频数据
    const short* pcm_data = ...; // 16bit 16kHz 单声道PCM数据
    int pcm_len = ...;           // 采样点个数
    bool is_final = false;       // 是否是最后一段

    int* segments = nullptr;
    int segments_len = 0;
    int ret = vad_process(handle, pcm_data, pcm_len, is_final, &segments, &segments_len);
    if (ret == 0 && segments_len > 0) {
        for (int i = 0; i < segments_len; ++i) {
            int start_ms = segments[i * 2];
            int end_ms = segments[i * 2 + 1];
            // 处理语音段
        }
        vad_free_segments(segments);
    }

    // 重置状态，处理新的音频流
    vad_reset(handle);

    // 释放资源
    vad_destroy(handle);
    return 0;
}
```

### C++ API 使用示例

```cpp
#include "vad.h"

int main() {
    // 初始化VAD
    vad::Vad vad("./model", 16000, vad::VAD_BACKEND_AUTO);
    if (!vad.IsValid()) {
        return -1;
    }

    // 处理音频
    std::vector<short> pcm_data = ...;
    auto segments = vad.Process(pcm_data, false);
    for (auto& seg : segments) {
        int start_ms = seg.first;
        int end_ms = seg.second;
        // 处理语音段
    }

    return 0;
}
```

## API 说明

### 后端类型

```c
typedef enum {
    VAD_BACKEND_AUTO = 0,  // 自动选择，RK3588优先NPU，否则CPU
    VAD_BACKEND_ONNX = 1,  // 强制使用ONNX Runtime CPU
    VAD_BACKEND_RKNN = 2   // 强制使用RKNN NPU
} VadBackendType;
```

### 初始化

```c
/**
 * @brief 初始化VAD模型
 * @param model_dir 模型目录路径，包含model.onnx(或model.rknn)、am.mvn、config.yaml
 * @param sample_rate 音频采样率，必须为16000
 * @param backend 后端类型
 * @return VAD句柄，失败返回NULL
 */
void* vad_init(const char* model_dir, int sample_rate = 16000, VadBackendType backend = VAD_BACKEND_AUTO);
```

### 处理音频

```c
/**
 * @brief 处理音频数据，检测语音段
 * @param handle VAD句柄
 * @param pcm_data 16bit PCM音频数据，单声道，采样率16000
 * @param pcm_len 音频数据长度(采样点个数)
 * @param is_final 是否是最后一段数据
 * @param out_segments 输出的语音段数组，每个段为[start_ms, end_ms]，需要调用vad_free_segments释放
 * @param out_segments_len 输出的语音段个数
 * @return 0成功，其他失败
 */
int vad_process(void* handle, const short* pcm_data, int pcm_len, bool is_final, int** out_segments, int* out_segments_len);
```

### 重置状态

```c
/**
 * @brief 重置VAD状态，用于处理新的音频流
 * @param handle VAD句柄
 * @return 0成功，其他失败
 */
int vad_reset(void* handle);
```

### 释放资源

```c
/**
 * @brief 释放VAD资源
 * @param handle VAD句柄
 * @return 0成功，其他失败
 */
int vad_destroy(void* handle);

/**
 * @brief 释放vad_process返回的segments内存
 * @param segments 语音段数组
 */
void vad_free_segments(int* segments);
```

## 模型转换

如果需要在RK3588上使用NPU加速，需要将ONNX模型转换为RKNN格式：

1. 安装RKNN Toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
2. 使用转换脚本：

```bash
# 转换为RKNN模型，不量化
python3 tools/convert_onnx_to_rknn.py model/onnx/model.onnx model/rknn/model.rknn

# 转换为量化模型
python3 tools/convert_onnx_to_rknn.py model/onnx/model.onnx model/rknn/model_quant.rknn --quantize --generate-dataset
```

## 模型文件说明

模型目录需要包含以下文件：

```
model/
├── model.onnx      # ONNX格式模型(用于CPU推理)
├── model.rknn      # RKNN格式模型(用于NPU推理，可选)
├── am.mvn          # CMVN归一化参数
└── config.yaml     # 模型配置文件
```

## 性能指标

| 平台 | 后端 | 实时因子(RTF) | 内存占用 |
|------|------|---------------|----------|
| RK3588 | RKNN NPU | <0.01 | <10MB |
| RK3588 | ONNX CPU | ~0.05 | <20MB |
| x86 i7-10700 | ONNX CPU | ~0.01 | <20MB |

*RTF = 处理时间 / 音频时长，RTF < 1表示可以实时处理*

## 许可证

MIT License，详见LICENSE文件。

## 参考

- [FunASR](https://github.com/alibaba-damo-academy/FunASR): 阿里达摩院语音团队开源的语音识别工具包
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank): 纯C++实现的Kaldi Fbank特征提取
- [RKNN Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2): Rockchip NPU开发工具包