#include "fsmn_vad_onnx.h"
#include "common.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vad {

FsmnVadOnnx::FsmnVadOnnx() {
    frame_sample_length_ = sample_rate_ / 1000 * 25;
    frame_shift_sample_length_ = sample_rate_ / 1000 * 10;
}

FsmnVadOnnx::~FsmnVadOnnx() {
    // ONNX Runtime会自动释放资源
}

void FsmnVadOnnx::Init(const std::string& model_dir, int sample_rate, int thread_num) {
    sample_rate_ = sample_rate;
    frame_sample_length_ = sample_rate_ / 1000 * 25;
    frame_shift_sample_length_ = sample_rate_ / 1000 * 10;

    // 模型路径
    std::string model_path = model_dir + "/" + MODEL_NAME;
    std::string cmvn_path = model_dir + "/" + VAD_CMVN_NAME;
    std::string config_path = model_dir + "/" + VAD_CONFIG_NAME;

    // 加载模型、CMVN、配置
    LoadModel(model_path);
    LoadCmvn(cmvn_path);
    LoadConfig(config_path);

    // 初始化Fbank
    fbank_opts_.frame_opts.samp_freq = sample_rate_;
    fbank_opts_.frame_opts.window_type = "hamming";
    fbank_opts_.frame_opts.frame_length_ms = 25;
    fbank_opts_.frame_opts.frame_shift_ms = 10;
    fbank_opts_.frame_opts.dither = 0.0;
    fbank_opts_.frame_opts.preemph_coeff = 0.97;
    fbank_opts_.frame_opts.remove_dc_offset = true;
    fbank_opts_.mel_opts.num_bins = 80;
    fbank_opts_.mel_opts.low_freq = 20;
    fbank_opts_.mel_opts.high_freq = sample_rate_ / 2 - 100;
    fbank_opts_.use_energy = false;
    fbank_opts_.use_log_fbank = true;
    fbank_opts_.use_power = true;

    // 初始化缓存
    InitCache();
}

void FsmnVadOnnx::LoadModel(const std::string& model_path) {
    // 初始化ONNX Runtime
    env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "FsmnVad");

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.DisableCpuMemArena();

#ifdef _WIN32
    std::wstring w_model_path(model_path.begin(), model_path.end());
    session_ = std::make_shared<Ort::Session>(env_, w_model_path.c_str(), session_options_);
#else
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_->GetInputCount();
    size_t num_output_nodes = session_->GetOutputCount();

    input_names_.resize(num_input_nodes);
    input_names_ptr_.resize(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name = session_->GetInputName(i, allocator);
        input_names_[i] = name;
        input_names_ptr_[i] = input_names_[i].c_str();
        allocator.Free(name);
    }

    output_names_.resize(num_output_nodes);
    output_names_ptr_.resize(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name = session_->GetOutputName(i, allocator);
        output_names_[i] = name;
        output_names_ptr_[i] = output_names_[i].c_str();
        allocator.Free(name);
    }
}

void FsmnVadOnnx::LoadCmvn(const std::string& cmvn_path) {
    std::ifstream is(cmvn_path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open cmvn file: " + cmvn_path);
    }

    // 读取means和vars
    // 格式：n_means float, n_vars float
    int feat_dim = 80 * lfr_m_;  // LFR后特征维度
    means_.resize(feat_dim);
    vars_.resize(feat_dim);

    is.read(reinterpret_cast<char*>(means_.data()), feat_dim * sizeof(float));
    is.read(reinterpret_cast<char*>(vars_.data()), feat_dim * sizeof(float));

    if (!is) {
        throw std::runtime_error("Failed to read cmvn file: " + cmvn_path);
    }

    // 转换为方差归一化：1/sqrt(var)
    for (int i = 0; i < feat_dim; ++i) {
        if (vars_[i] < 1e-20) {
            vars_[i] = 1.0;
        } else {
            vars_[i] = 1.0 / sqrt(vars_[i]);
        }
    }
}

void FsmnVadOnnx::LoadConfig(const std::string& config_path) {
    // 简单配置，暂时不需要读取yaml，使用默认值即可
    // 如果有特殊配置需求，可以在这里添加yaml解析
}

void FsmnVadOnnx::InitCache() {
    // FSMN缓存：4层，每层19个单元
    in_cache_.resize(4, std::vector<float>(19, 0.0f));

    // 特征缓存
    reserve_waveforms_.clear();
    input_cache_.clear();
    lfr_splice_cache_.clear();

    // 重置VAD检测器
    vad_scorer_.Reset();
}

void FsmnVadOnnx::Reset() {
    InitCache();
}

void FsmnVadOnnx::SetConfig(int vad_tail_sil, int vad_max_len, int vad_energy, int vad_snr) {
    if (vad_tail_sil > 0) {
        vad_silence_duration_ = vad_tail_sil;
    }
    if (vad_max_len > 0) {
        vad_max_len_ = vad_max_len;
    }
    if (vad_energy > -1000) {
        decibel_thres_ = vad_energy;
    }
    if (vad_snr > -1000) {
        snr_thres_ = vad_snr;
    }
}

std::vector<std::pair<int, int>> FsmnVadOnnx::Process(const short* pcm_data, int pcm_len, bool is_final) {
    // 转换为float，范围[-1, 1]
    std::vector<float> pcm_float(pcm_len);
    for (int i = 0; i < pcm_len; ++i) {
        pcm_float[i] = pcm_data[i] / 32768.0f;
    }

    // 特征提取
    std::vector<std::vector<float>> feats;
    ExtractFbank(pcm_float, feats);

    // 在线LFR和CMVN
    std::vector<std::vector<float>> lfr_feats;
    int num_frames = OnlineLfrCmvn(feats, is_final);
    if (num_frames <= 0) {
        return {};
    }

    // 推理
    std::vector<std::vector<float>> logits;
    Forward(feats, logits, in_cache_, is_final);

    // VAD后处理
    std::vector<float> waveform_copy = pcm_float;
    auto segments = vad_scorer_(logits, waveform_copy, is_final, true,
                                vad_silence_duration_, vad_max_len_,
                                vad_speech_noise_thres_, sample_rate_);

    // 转换为pair格式
    std::vector<std::pair<int, int>> result;
    for (auto& seg : segments) {
        if (seg[0] >= 0 && seg[1] > seg[0]) {
            result.emplace_back(seg[0], seg[1]);
        }
    }

    return result;
}

void FsmnVadOnnx::ExtractFbank(const std::vector<float>& pcm_float, std::vector<std::vector<float>>& feats) {
    // 拼接缓存的音频
    std::vector<float> waveform = reserve_waveforms_;
    waveform.insert(waveform.end(), pcm_float.begin(), pcm_float.end());

    if (waveform.size() < frame_sample_length_) {
        reserve_waveforms_ = waveform;
        return;
    }

    // 计算帧数
    int num_frames = (waveform.size() - frame_sample_length_) / frame_shift_sample_length_ + 1;
    int sample_used = (num_frames - 1) * frame_shift_sample_length_ + frame_sample_length_;

    // 保存剩余的音频
    reserve_waveforms_ = std::vector<float>(waveform.begin() + sample_used, waveform.end());

    // 提取Fbank特征
    knf::FbankComputer fbank_computer(fbank_opts_);
    feats.resize(num_frames, std::vector<float>(fbank_computer.Dim()));

    for (int i = 0; i < num_frames; ++i) {
        int start = i * frame_shift_sample_length_;
        std::vector<float> frame(waveform.begin() + start, waveform.begin() + start + frame_sample_length_);

        float energy = 0.0f;
        if (fbank_computer.NeedRawLogEnergy()) {
            for (float sample : frame) {
                energy += sample * sample;
            }
            energy = logf(std::max(energy, 1e-20f));
        }

        fbank_computer.Compute(energy, 1.0f, &frame, feats[i].data());
    }
}

void FsmnVadOnnx::LfrCmvn(std::vector<std::vector<float>>& feats) {
    // LFR：将lfr_m_帧合并为1帧，步长lfr_n_
    int num_frames = feats.size();
    int lfr_frames = (num_frames - lfr_m_) / lfr_n_ + 1;
    if (lfr_frames <= 0) return;

    std::vector<std::vector<float>> lfr_feats;
    lfr_feats.reserve(lfr_frames);

    for (int i = 0; i < lfr_frames; ++i) {
        int start = i * lfr_n_;
        std::vector<float> feat;
        feat.reserve(80 * lfr_m_);
        for (int j = 0; j < lfr_m_; ++j) {
            int idx = start + j;
            if (idx < num_frames) {
                feat.insert(feat.end(), feats[idx].begin(), feats[idx].end());
            } else {
                // 填充最后一帧
                feat.insert(feat.end(), feats.back().begin(), feats.back().end());
            }
        }
        lfr_feats.push_back(feat);
    }

    // CMVN归一化
    for (auto& feat : lfr_feats) {
        for (int i = 0; i < feat.size(); ++i) {
            feat[i] = (feat[i] - means_[i]) * vars_[i];
        }
    }

    feats.swap(lfr_feats);
}

int FsmnVadOnnx::OnlineLfrCmvn(std::vector<std::vector<float>>& feats, bool is_final) {
    // 拼接缓存的特征
    std::vector<std::vector<float>> all_feats = input_cache_;
    all_feats.insert(all_feats.end(), feats.begin(), feats.end());

    int num_frames = all_feats.size();
    int lfr_frames = 0;

    if (is_final) {
        // 最后一段，处理所有帧
        lfr_frames = (num_frames + lfr_n_ - 1) / lfr_n_;
        if (num_frames < lfr_m_) {
            // 不足lfr_m_帧，用最后一帧填充
            while (all_feats.size() < lfr_m_) {
                all_feats.push_back(all_feats.back());
            }
            lfr_frames = 1;
        }
    } else {
        // 非最后一段，只处理完整的LFR块
        lfr_frames = (num_frames - lfr_m_) / lfr_n_ + 1;
        if (lfr_frames <= 0) {
            // 不足lfr_m_帧，缓存起来
            input_cache_ = all_feats;
            return 0;
        }
        // 计算剩余的帧数
        int remaining = num_frames - (lfr_frames - 1) * lfr_n_ - lfr_m_ + lfr_n_;
        if (remaining > 0) {
            input_cache_ = std::vector<std::vector<float>>(
                all_feats.end() - remaining, all_feats.end());
        } else {
            input_cache_.clear();
        }
    }

    // 应用LFR和CMVN
    std::vector<std::vector<float>> lfr_feats;
    lfr_feats.reserve(lfr_frames);

    for (int i = 0; i < lfr_frames; ++i) {
        int start = i * lfr_n_;
        std::vector<float> feat;
        feat.reserve(80 * lfr_m_);
        for (int j = 0; j < lfr_m_; ++j) {
            int idx = start + j;
            if (idx < num_frames) {
                feat.insert(feat.end(), all_feats[idx].begin(), all_feats[idx].end());
            } else {
                // 填充最后一帧
                feat.insert(feat.end(), all_feats.back().begin(), all_feats.back().end());
            }
        }
        // CMVN
        for (int k = 0; k < feat.size(); ++k) {
            feat[k] = (feat[k] - means_[k]) * vars_[k];
        }
        lfr_feats.push_back(feat);
    }

    feats.swap(lfr_feats);
    return feats.size();
}

void FsmnVadOnnx::Forward(const std::vector<std::vector<float>>& feats,
                          std::vector<std::vector<float>>& out_prob,
                          std::vector<std::vector<float>>& in_cache,
                          bool is_final) {
    if (feats.empty()) return;

    int num_frames = feats.size();
    int feat_dim = feats[0].size();

    // 准备输入tensor
    std::vector<int64_t> input_shape = {1, num_frames, feat_dim};
    std::vector<float> input_data;
    input_data.reserve(num_frames * feat_dim);
    for (auto& feat : feats) {
        input_data.insert(input_data.end(), feat.begin(), feat.end());
    }

    // 准备cache输入
    std::vector<int64_t> cache_shape = {1, 4, 19};
    std::vector<float> cache_data;
    cache_data.reserve(4 * 19);
    for (auto& cache : in_cache) {
        cache_data.insert(cache_data.end(), cache.begin(), cache.end());
    }

    // 创建输入tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, cache_data.data(), cache_data.size(),
        cache_shape.data(), cache_shape.size()));

    // 运行推理
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_names_ptr_.data(), input_tensors.data(), input_tensors.size(),
                                        output_names_ptr_.data(), output_names_ptr_.size());

    // 获取输出
    float* logits_data = output_tensors[0].GetTensorMutableData<float>();
    float* cache_out_data = output_tensors[1].GetTensorMutableData<float>();

    // 处理输出概率
    out_prob.resize(num_frames, std::vector<float>(2));
    for (int i = 0; i < num_frames; ++i) {
        float x0 = logits_data[i * 2 + 0];
        float x1 = logits_data[i * 2 + 1];
        // Softmax
        float max_val = std::max(x0, x1);
        float exp0 = expf(x0 - max_val);
        float exp1 = expf(x1 - max_val);
        float sum = exp0 + exp1;
        out_prob[i][0] = exp0 / sum;
        out_prob[i][1] = exp1 / sum;
    }

    // 更新cache
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 19; ++j) {
            in_cache[i][j] = cache_out_data[i * 19 + j];
        }
    }
}

} // namespace vad