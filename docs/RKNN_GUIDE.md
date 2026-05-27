# RKNN 模型导出与部署指南

FSMN VAD 支持 RKNN NPU 后端（RK3588），提供与 ONNX Runtime 后端完全一致的推理流程。本文档覆盖从 ONNX 导出、RKNN 转换、PC 模拟器验证到板端部署的全流程。

## 完整流程

```
FunASR 导出 ONNX
      ↓
ONNX → RKNN 转换（rknn-toolkit2）
      ↓
PC 模拟器验证 & 基准测试（x86_64）
      ↓
aarch64 交叉编译
      ↓
RK3588 板端部署
```

## 1. 环境准备

| 组件 | 说明 |
|------|------|
| rknn-toolkit2 | Python 包，用于 ONNX → RKNN 转换 + PC 模拟器推理 |
| RKNN SDK | C 头文件 + 动态库，用于交叉编译板端程序 |
| aarch64 交叉编译器 | `gcc-aarch64-linux-gnu` / `g++-aarch64-linux-gnu` |

### 1.1 安装 rknn-toolkit2（WSL / Linux x86_64）

```bash
# 下载 rknn-toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
pip install rknn-toolkit2
# 或从 Rockchip 提供的 whl 安装
pip install rknn_toolkit2-*-cp37-cp37m-linux_x86_64.whl
pip install onnx numpy onnxruntime
```

验证安装：

```bash
python -c "from rknn.api import RKNN; print('OK')"
```

### 1.2 获取 RKNN SDK（交叉编译用）

从 Rockchip 官方获取 RK3588 的 RKNN SDK，含 `rknn_api.h` 和 `librknnrt.so`（aarch64）。目录结构：

```
rknn_sdk/
├── include/
│   └── rknn_api.h
└── lib/
    └── aarch64/
        └── librknnrt.so
```

放置到项目根目录下 `rknn_sdk/`。

### 1.3 安装交叉编译器

```bash
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

## 2. 模型转换

### 2.1 导出 ONNX 模型

使用 FunASR 导出 FSMN VAD 的 ONNX 模型（需 Python 3.10+）：

```python
from funasr import AutoModel

model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
res = model.export(type="onnx", quantize=False)
print(res)
```

导出产物为 `model.onnx`（**非量化 FP32**，约 1.7MB）。

### 2.2 转换为 RKNN

使用 [convert_onnx_to_rknn.py](../tools/convert_onnx_to_rknn.py)：

```bash
python tools/convert_onnx_to_rknn.py model/model.onnx model/model.rknn
```

转换完成后可用 `--verify` 参数在 PC 模拟器上验证：

```bash
python tools/convert_onnx_to_rknn.py model/model.onnx model/model.rknn --verify
```

### 2.3 模型 I/O 说明

FSMN VAD 模型共有 **5 个输入 / 5 个输出**：

| 方向 | 名称 | Shape | 说明 |
|------|------|-------|------|
| Input | `speech` | [1, N, 400] | LFR+CMVN 后的特征，N=1/6/36 |
| Input | `in_cache0~3` | [1, 128, 19, 1] | FSMN 层缓存（NCHW） |
| Output | `logits` | [1, N, 248] | 原始 logits（非 softmax） |
| Output | `out_cache0~3` | [1, 128, 19, 1] | 更新后的缓存（NCHW） |

关键说明：
- **N（帧数）**：支持 1/6/36 三种尺寸，通过 `dynamic_input` 配置。非精确匹配的中间值会自动 padding 到最近的有效尺寸（[原理](#dynamic-input-原理)）
- **Cache 格式**：RKNN 输入必须为 **NHWC** `[1, 19, 1, 128]`（从 NCHW `[1,128,19,1]` 转置），输出为 **NCHW** `[1,128,19,1]`
- **logits**：248 维原始输出，**不需要 softmax**，后端直接取指定维度的值

### 2.4 dynamic_input 原理

RKNN 的 `dynamic_input` 支持在 **3 组预定义尺寸**间切换，每组尺寸对应 min/opt/max：

```python
dynamic_input=[
    [[1, 1, 400],  [1,128,19,1], [1,128,19,1], [1,128,19,1], [1,128,19,1]],  # min  (1帧)
    [[1, 6, 400],  [1,128,19,1], [1,128,19,1], [1,128,19,1], [1,128,19,1]],  # opt  (6帧)
    [[1, 36, 400], [1,128,19,1], [1,128,19,1], [1,128,19,1], [1,128,19,1]],  # max  (36帧)
]
```

- 输入帧数 = 1 → 使用第一组尺寸
- 输入帧数 ≤ 6 → 使用第二组（6帧），不足部分后端自动 padding
- 输入帧数 > 6 → 使用第三组（36帧），不足部分后端自动 padding

C++ 代码中的 padding 逻辑（[`fsmn_vad_rknn.cpp`](../src/fsmn_vad_rknn.cpp)）：

- 按实际帧数选择 1 / 6 / 36，不足部分补零
- **板端 NPU** 须在 `rknn_inputs_set` 前调用 `rknn_set_input_shapes`（查询 `RKNN_QUERY_INPUT_DYNAMIC_RANGE` 后按 shape 组设置）；PC 模拟器由 toolkit 处理，无此步骤

## 3. PC 模拟器验证

rknn-toolkit2 在 x86_64 上提供 PC 模拟器，无需 RK3588 硬件即可验证推理正确性和速度。

### 3.1 基准测试脚本

使用 [benchmark_rknn.py](../tools/benchmark_rknn.py)：

```bash
# 完整测试（速度 + 与 ONNX 对比）
python tools/benchmark_rknn.py --iterations 200

# 仅测速度，跳过 ONNX 对比
python tools/benchmark_rknn.py --no-verify --iterations 500
```

脚本自动执行：从 ONNX 构建 RKNN → PC 模拟器推理 → 输出分析 → 正确性对比 → 速度测试 → RTF 估算。

### 3.2 预期结果

| 项目 | PC 模拟器 | RK3588 NPU（估算）|
|------|-----------|-------------------|
| 6 帧推理 | ~5 ms | < 1 ms |
| 36 帧推理 | ~6 ms | < 1 ms |
| logits 误差 | < 0.002 | 与模拟器一致 |
| cache 误差 | < 0.02 | 与模拟器一致 |

PC 模拟器使用 CPU 模拟 NPU 指令，速度约 5ms/infer，远慢于实际 NPU（~0.3ms）。RTF（6帧 = 60ms 音频）约 0.08，仍有 **6 倍余量**。

## 4. 交叉编译

### 4.1 配置

```bash
mkdir build-arm64 && cd build-arm64
cmake .. \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DENABLE_ONNX=OFF \
    -DENABLE_RKNN=ON \
    -DRKNN_SDK_DIR=/path/to/vad-runtime/rknn_sdk
make -j4
```

### 4.2 编译产物

```
build-arm64/
├── libfsmn_vad.so        # VAD 动态库（RKNN 后端）
└── vad_example_rknn      # 示例程序
```

### 4.3 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_RKNN` | OFF | 启用 RKNN NPU 后端 |
| `ENABLE_ONNX` | ON | 建议关闭，避免需要 ONNX Runtime 头文件 |
| `RKNN_SDK_DIR` | (空) | 旧版布局：`include/` + `lib/aarch64/librknnrt.so` |
| `RKNN_MODEL_ZOO_ROOT` | (空) | rknn_model_zoo 根目录，使用 `3rdparty/rknpu2` |
| `RKNN_RKNPU2_ROOT` | (空) | rknpu2 SDK 根目录（与 model_zoo 内 rknpu2 布局相同） |

## 5. RK3588 板端部署

### 5.0 Android（adb）

完整流程（构建、部署、调试）见 **[ANDROID.md](ANDROID.md)** § RKNN NPU。

### 5.1 Linux（ssh/scp）

```bash
# 创建目录
ssh rk3588 'mkdir -p /home/root/vad/model'

# 推送编译产物
scp build-arm64/vad_example_rknn rk3588:/home/root/vad/
scp build-arm64/libfsmn_vad.so rk3588:/home/root/vad/

# 推送 RKNN 运行时库
scp rknn_sdk/lib/aarch64/librknnrt.so rk3588:/home/root/vad/

# 推送模型和音频
scp model/model.rknn rk3588:/home/root/vad/model/
scp model/am.mvn rk3588:/home/root/vad/model/
scp model/config.yaml rk3588:/home/root/vad/model/
scp examples/xmov.raw rk3588:/home/root/vad/
```

### 5.2 运行（Linux）

```bash
ssh rk3588 'cd /home/root/vad && \
  LD_LIBRARY_PATH=. ./vad_example_rknn ./model ./xmov.raw'
```

### 5.3 预期输出

```
VAD initialized, backend: 2
Read 83968 samples, duration: 5248ms

   0ms: BEGIN    0.001    [0,-1]
 360ms: CONTINUE    0.001
4800ms: END    0.001    [-1,4270]
4860ms: NONE    0.002

=== Summary ===
RTF: 0.001
```

后端号 `2` 表示 RKNN。NPU 推理延迟约 0.3ms（远低于 ONNX CPU 的 ~2ms）。

## 6. 常见问题

### 6.1 load_rknn 后无法在 PC 模拟器运行

```
E init_runtime: RKNN model that loaded by 'load_rknn' not support inference on the simulator
```

**原因**：`load_rknn` 加载的 `.rknn` 模型是已编译的板端格式，PC 模拟器不支持直接运行。

**解决**：使用 `load_onnx` + `build` + `init_runtime(target=None)` 完整流水线，参见 [benchmark_rknn.py](../tools/benchmark_rknn.py) 的 `load_and_build` 函数。

### 6.2 推理时 shape 错误

```
ValueError: _get_dynamic_idx ...
```

**原因**：输入帧数不在 dynamic_input 的 3 组尺寸范围内，或 cache 的 NHWC/NCHW 格式错误。

**解决**：确保输入帧数为 1/6/36 之一，cache 输入为 NHWC `[1,19,1,128]`。

### 6.3 RKNN 输入/输出格式

| 参数 | 输入格式 | 输出格式 |
|------|----------|----------|
| speech | NCHW `[1,N,400]` | — |
| cache 0~3 | **NHWC** `[1,19,1,128]` | **NCHW** `[1,128,19,1]` |
| logits | — | `[1,N,248]` |

Cache 格式不对称（输入 NHWC，输出 NCHW），流式拼接时需转置：

```cpp
// NCHW → NHWC（输入前转换）
for (int c = 0; c < 128; c++)
    for (int h = 0; h < 19; h++)
        nhwc[h * 128 + c] = nchw[c * 19 + h];

// NHWC → NCHW（输出后转换，如需下次输入）
for (int h = 0; h < 19; h++)
    for (int c = 0; c < 128; c++)
        nchw[c * 19 + h] = nhwc[h * 128 + c];
```

### 6.4 交叉编译找不到 rknn_api.h

```
fatal error: rknn_api.h: No such file or directory
```

**解决**：确认 `-DRKNN_SDK_DIR` 指向包含 `include/rknn_api.h` 的目录，或手动设置 `-DRKNN_SDK_DIR=/path/to/vad-runtime/rknn_sdk`。

### 6.5 板端运行报 librknnrt.so 找不到

```
error while loading shared libraries: librknnrt.so
```

**解决**：确保 `librknnrt.so` 推送到板端，并在 `LD_LIBRARY_PATH` 中：

```bash
export LD_LIBRARY_PATH=/home/root/vad:$LD_LIBRARY_PATH
```

## 相关文件

| 文件 | 说明 |
|------|------|
| [convert_onnx_to_rknn.py](convert_onnx_to_rknn.py) | ONNX → RKNN 转换脚本 |
| [benchmark_rknn.py](../tools/benchmark_rknn.py) | PC 模拟器基准测试 |
| [../src/fsmn_vad_rknn.cpp](../src/fsmn_vad_rknn.cpp) | RKNN 推理后端 C++ 实现 |
| [../include/fsmn_vad_rknn.h](../include/fsmn_vad_rknn.h) | RKNN 后端头文件 |
| [../examples/demo_rknn.cpp](../examples/demo_rknn.cpp) | RKNN 示例程序 |
