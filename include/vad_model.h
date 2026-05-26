#ifndef VAD_MODEL_H
#define VAD_MODEL_H

#include <string>
#include <vector>
#include "common.h"

namespace vad {

class VadModel {
public:
    virtual ~VadModel() = default;

    /**
     * @brief 初始化VAD模型
     * @param model_dir 模型目录路径
     * @param sample_rate 音频采样率，默认16000
     * @param thread_num 线程数，默认1
     */
    virtual void Init(const std::string& model_dir, int sample_rate = MODEL_SAMPLE_RATE, int thread_num = 1) = 0;

    /**
     * @brief 处理音频数据，检测语音段
     * @param pcm_data 16bit PCM音频数据，单声道，采样率16000
     * @param pcm_len 音频数据长度（采样点个数）
     * @param is_final 是否是最后一段数据
     * @return 检测到的语音段列表，每个段为[start_ms, end_ms]
     */
    virtual std::vector<std::pair<int, int>> Process(const short* pcm_data, int pcm_len, bool is_final = false) = 0;

    /**
     * @brief 重置VAD状态，用于新的音频流处理
     */
    virtual void Reset() = 0;

    /**
     * @brief 获取当前使用的后端类型
     * @return 后端类型
     */
    virtual VadBackendType GetBackendType() const = 0;

    /**
     * @brief 获取VAD采样率
     * @return 采样率
     */
    virtual int GetSampleRate() const = 0;

    /**
     * @brief 设置VAD参数
     * @param vad_tail_sil 尾部静音时长(ms)
     * @param vad_max_len 最大语音段长度(ms)
     * @param vad_energy 能量阈值
     * @param vad_snr 信噪比阈值
     */
    virtual void SetConfig(int vad_tail_sil, int vad_max_len, int vad_energy, int vad_snr) = 0;
};

} // namespace vad

#endif // VAD_MODEL_H