#ifndef FSMN_VAD_ONNX_H
#define FSMN_VAD_ONNX_H

#include "vad_model.h"
#include "e2e_vad.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace vad {

class FsmnVadOnnx : public VadModel {
public:
    FsmnVadOnnx();
    ~FsmnVadOnnx() override;

    void Init(const std::string& model_dir, int sample_rate = MODEL_SAMPLE_RATE, int thread_num = 1) override;
    std::vector<std::pair<int, int>> Process(const short* pcm_data, int pcm_len, bool is_final = false) override;
    void Reset() override;
    VadBackendType GetBackendType() const override { return VAD_BACKEND_ONNX; }
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
    void FbankKaldi(const std::vector<float>& waves, std::vector<std::vector<float>>& vad_feats);
    void ExtractFeats(std::vector<std::vector<float>>& vad_feats, std::vector<float>& waves, bool input_finished);
    int OnlineLfrCmvn(std::vector<std::vector<float>>& vad_feats, bool input_finished);
    static int ComputeFrameNum(int sample_length, int frame_sample_length, int frame_shift_sample_length);
    void ResetCache() {
        reserve_waveforms_.clear();
        input_cache_.clear();
        lfr_splice_cache_.clear();
    }

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

    // ONNX Runtime相关
    std::shared_ptr<Ort::Session> session_ = nullptr;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;

    // Fbank特征提取
    knf::FbankOptions fbank_opts_;

    // CMVN参数
    std::vector<float> means_;
    std::vector<float> vars_;

    // 推理缓存 (4层, 每层[128, 19, 1], 扁平化为128*19=2432)
    std::vector<std::vector<float>> in_cache_;
    std::vector<float> reserve_waveforms_;
    std::vector<float> input_cache_;           // 音频采样缓存 (float, 用于跨块音频拼接)
    std::vector<std::vector<float>> lfr_splice_cache_;  // LFR特征缓存

    // VAD端点检测
    E2EVadModel vad_scorer_;
};

} // namespace vad

#endif // FSMN_VAD_ONNX_H