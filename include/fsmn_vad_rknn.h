#ifndef FSMN_VAD_RKNN_H
#define FSMN_VAD_RKNN_H

#include "vad_model.h"
#include "e2e_vad.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include <memory>

// RKNN SDK 头文件，编译时需要指定RKNN SDK路径
#ifdef ENABLE_RKNN
#include <rknn_api.h>
#endif

namespace vad {

class FsmnVadRknn : public VadModel {
public:
    FsmnVadRknn();
    ~FsmnVadRknn() override;

    void Init(const std::string& model_dir, int sample_rate = MODEL_SAMPLE_RATE, int thread_num = 1) override;
    std::vector<std::pair<int, int>> Process(const short* pcm_data, int pcm_len, bool is_final = false) override;
    void Reset() override;
    VadBackendType GetBackendType() const override { return VAD_BACKEND_RKNN; }
    int GetSampleRate() const override { return sample_rate_; }
    void SetConfig(int vad_tail_sil, int vad_max_len, int vad_energy, int vad_snr) override;

private:
    // 模型加载与初始化
    void LoadModel(const std::string& model_path);
    void LoadCmvn(const std::string& cmvn_path);
    void LoadConfig(const std::string& config_path);
    void InitCache();
    void InitOnline();

    // 特征提取与预处理
    void ExtractFbank(const std::vector<float>& pcm_float, std::vector<std::vector<float>>& feats);
    void LfrCmvn(std::vector<std::vector<float>>& feats);
    int OnlineLfrCmvn(std::vector<std::vector<float>>& feats, bool is_final);

    // 模型推理
    void Forward(const std::vector<std::vector<float>>& feats,
                 std::vector<std::vector<float>>& out_probs,
                 std::vector<std::vector<float>>& in_cache,
                 bool is_final);

    // 常量配置
    int sample_rate_ = MODEL_SAMPLE_RATE;
    int vad_silence_duration_ = VAD_SILENCE_DURATION;
    int vad_max_len_ = VAD_MAX_LEN;
    double vad_speech_noise_thres_ = VAD_SPEECH_NOISE_THRES;
    int lfr_m_ = VAD_LFR_M;
    int lfr_n_ = VAD_LFR_N;
    int frame_sample_length_;
    int frame_shift_sample_length_;

    float snr_thres_ = -100.0;
    float decibel_thres_ = -100.0;

#ifdef ENABLE_RKNN
    // RKNN相关
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_tensor_mem> input_mems_;
    std::vector<rknn_tensor_mem> output_mems_;
#endif

    // Fbank特征提取
    knf::FbankOptions fbank_opts_;

    // CMVN参数
    std::vector<float> means_;
    std::vector<float> vars_;

    // 推理缓存
    std::vector<std::vector<float>> in_cache_;
    std::vector<float> reserve_waveforms_;
    std::vector<float> input_cache_;
    std::vector<std::vector<float>> lfr_splice_cache_;

    // VAD端点检测
    E2EVadModel vad_scorer_;
};

} // namespace vad

#endif // FSMN_VAD_RKNN_H