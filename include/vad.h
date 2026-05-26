#ifndef VAD_H
#define VAD_H

#include "common.h"
#include <vector>
#include <utility>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化VAD模型
 * @param model_dir 模型目录路径
 * @param sample_rate 音频采样率，默认16000
 * @param backend 后端类型，默认自动选择
 * @return VAD句柄，失败返回NULL
 */
void* vad_init(const char* model_dir, int sample_rate = 16000, VadBackendType backend = VAD_BACKEND_AUTO);

/**
 * @brief 处理音频数据，检测语音段
 * @param handle VAD句柄
 * @param pcm_data 16bit PCM音频数据，单声道，采样率16000
 * @param pcm_len 音频数据长度（采样点个数）
 * @param is_final 是否是最后一段数据
 * @param out_segments 输出的语音段数组，每个段为[start_ms, end_ms]
 * @param out_segments_len 输出的语音段个数
 * @return 0成功，其他失败
 */
int vad_process(void* handle, const short* pcm_data, int pcm_len, bool is_final,
                int** out_segments, int* out_segments_len);

/**
 * @brief 重置VAD状态，用于新的音频流处理
 * @param handle VAD句柄
 * @return 0成功，其他失败
 */
int vad_reset(void* handle);

/**
 * @brief 释放VAD资源
 * @param handle VAD句柄
 * @return 0成功，其他失败
 */
int vad_destroy(void* handle);

/**
 * @brief 获取当前使用的后端类型
 * @param handle VAD句柄
 * @return 后端类型
 */
VadBackendType vad_get_backend(void* handle);

/**
 * @brief 释放process返回的segments内存
 * @param segments 语音段数组
 */
void vad_free_segments(int* segments);

#ifdef __cplusplus
}
#endif

// C++ API
#ifdef __cplusplus
#include <vector>
#include <utility>
#include <memory>

namespace vad {

/**
 * @brief VAD C++ API封装类
 */
class Vad {
public:
    /**
     * @brief 构造函数
     * @param model_dir 模型目录路径
     * @param sample_rate 音频采样率，默认16000
     * @param backend 后端类型，默认自动选择
     */
    explicit Vad(const std::string& model_dir, int sample_rate = 16000,
                 VadBackendType backend = VAD_BACKEND_AUTO) {
        handle_ = vad_init(model_dir.c_str(), sample_rate, backend);
    }

    ~Vad() {
        if (handle_) {
            vad_destroy(handle_);
        }
    }

    /**
     * @brief 处理音频数据，检测语音段
     * @param pcm_data 16bit PCM音频数据，单声道，采样率16000
     * @param is_final 是否是最后一段数据
     * @return 检测到的语音段列表，每个段为[start_ms, end_ms]
     */
    std::vector<std::pair<int, int>> Process(const std::vector<short>& pcm_data, bool is_final = false) {
        return Process(pcm_data.data(), pcm_data.size(), is_final);
    }

    /**
     * @brief 处理音频数据，检测语音段
     * @param pcm_data 16bit PCM音频数据指针
     * @param pcm_len 音频数据长度（采样点个数）
     * @param is_final 是否是最后一段数据
     * @return 检测到的语音段列表，每个段为[start_ms, end_ms]
     */
    std::vector<std::pair<int, int>> Process(const short* pcm_data, int pcm_len, bool is_final = false) {
        std::vector<std::pair<int, int>> result;
        if (!handle_) return result;

        int* segments = nullptr;
        int segments_len = 0;
        if (vad_process(handle_, pcm_data, pcm_len, is_final, &segments, &segments_len) == 0) {
            for (int i = 0; i < segments_len; ++i) {
                result.emplace_back(segments[i * 2], segments[i * 2 + 1]);
            }
            vad_free_segments(segments);
        }
        return result;
    }

    /**
     * @brief 重置VAD状态，用于新的音频流处理
     */
    void Reset() {
        if (handle_) {
            vad_reset(handle_);
        }
    }

    /**
     * @brief 获取当前使用的后端类型
     * @return 后端类型
     */
    VadBackendType GetBackendType() const {
        if (!handle_) return VAD_BACKEND_AUTO;
        return vad_get_backend(handle_);
    }

    /**
     * @brief 检查VAD是否初始化成功
     * @return true成功，false失败
     */
    bool IsValid() const { return handle_ != nullptr; }

    // 禁止拷贝
    Vad(const Vad&) = delete;
    Vad& operator=(const Vad&) = delete;

private:
    void* handle_ = nullptr;
};

} // namespace vad

#endif // __cplusplus

#endif // VAD_H