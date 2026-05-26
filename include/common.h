#ifndef VAD_COMMON_H
#define VAD_COMMON_H

namespace vad {

// VAD默认配置常量
#ifndef MODEL_SAMPLE_RATE
#define MODEL_SAMPLE_RATE 16000
#endif

#ifndef VAD_SILENCE_DURATION
#define VAD_SILENCE_DURATION 800  // 尾部静音阈值(ms)
#endif

#ifndef VAD_MAX_LEN
#define VAD_MAX_LEN 15000         // 最大单段长度(ms)
#endif

#ifndef VAD_SPEECH_NOISE_THRES
#define VAD_SPEECH_NOISE_THRES 0.9  // 语音/噪声阈值
#endif

#ifndef VAD_LFR_M
#define VAD_LFR_M 5  // 帧累积数
#endif

#ifndef VAD_LFR_N
#define VAD_LFR_N 1  // 帧移
#endif

// 模型文件名
#define MODEL_NAME "model_quant.onnx" // ONNX格式模型
#define RKNN_MODEL_NAME "model.rknn"  // RKNN格式模型
#define VAD_CMVN_NAME "am.mvn"        // 归一化参数
#define VAD_CONFIG_NAME "config.yaml" // 配置文件

// VAD后端类型
typedef enum {
    VAD_BACKEND_AUTO = 0,  // 自动选择（RK3588优先NPU，否则CPU）
    VAD_BACKEND_ONNX = 1,  // 强制使用ONNX Runtime CPU
    VAD_BACKEND_RKNN = 2   // 强制使用RKNN NPU
} VadBackendType;

} // namespace vad

#endif // VAD_COMMON_H