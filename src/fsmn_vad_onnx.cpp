#include "fsmn_vad_onnx.h"
#include "common.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iterator>

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

    // 设置vad_scorer参数
    vad_scorer_.vad_opts.decibel_thres = decibel_thres_;
    vad_scorer_.vad_opts.snr_thres = snr_thres_;
    vad_scorer_.vad_opts.speech_noise_thres = vad_speech_noise_thres_;
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

    // 获取输入输出名称和shape信息
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_->GetInputCount();
    size_t num_output_nodes = session_->GetOutputCount();

    input_names_.resize(num_input_nodes);
    input_names_ptr_.resize(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_[i] = name.get();
        input_names_ptr_[i] = input_names_[i].c_str();

        // (debug) auto type_info = session_->GetInputTypeInfo(i);
        // (debug) auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        // (debug) auto shape = shape_info.GetShape();
    }

    output_names_.resize(num_output_nodes);
    output_names_ptr_.resize(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_[i] = name.get();
        output_names_ptr_[i] = output_names_[i].c_str();

        // (debug) auto type_info = session_->GetOutputTypeInfo(i);
        // (debug) auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        // (debug) auto shape = shape_info.GetShape();
    }
}

void FsmnVadOnnx::LoadCmvn(const std::string& cmvn_path) {
    std::ifstream cmvn_stream(cmvn_path);
    if (!cmvn_stream) {
        throw std::runtime_error("Failed to open cmvn file: " + cmvn_path);
    }

    // 解析Kaldi文本格式的CMVN
    // <AddShift> ... <LearnRateCoef> 0 [ values ]
    // <Rescale> ... <LearnRateCoef> 0 [ values ]
    int feat_dim = 80 * lfr_m_;  // LFR后特征维度
    means_.resize(feat_dim);
    vars_.resize(feat_dim);

    std::string line;
    while (getline(cmvn_stream, line)) {
        std::istringstream iss(line);
        std::vector<std::string> line_item{std::istream_iterator<std::string>{iss},
                                           std::istream_iterator<std::string>{}};
        if (line_item.empty()) continue;

        if (line_item[0] == "<AddShift>") {
            // 读取下一行
            getline(cmvn_stream, line);
            std::istringstream means_lines_stream(line);
            std::vector<std::string> means_lines{std::istream_iterator<std::string>{means_lines_stream},
                                                 std::istream_iterator<std::string>{}};
            if (means_lines.size() >= 4 && means_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < means_lines.size() - 1 && (j - 3) < feat_dim; j++) {
                    means_[j - 3] = std::stof(means_lines[j]);
                }
            }
        } else if (line_item[0] == "<Rescale>") {
            // 读取下一行
            getline(cmvn_stream, line);
            std::istringstream vars_lines_stream(line);
            std::vector<std::string> vars_lines{std::istream_iterator<std::string>{vars_lines_stream},
                                                std::istream_iterator<std::string>{}};
            if (vars_lines.size() >= 4 && vars_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < vars_lines.size() - 1 && (j - 3) < feat_dim; j++) {
                    vars_[j - 3] = std::stof(vars_lines[j]);
                }
            }
        }
    }
}

void FsmnVadOnnx::LoadConfig(const std::string& config_path) {
    std::ifstream is(config_path);
    if (!is) {
        fprintf(stderr, "Warning: Failed to open config file: %s, use defaults\n", config_path.c_str());
        return;
    }

    // 简易YAML解析，仅提取VAD需要的配置项
    std::string line;
    std::string section;
    while (std::getline(is, line)) {
        // 去除首尾空格
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r"));
            s.erase(s.find_last_not_of(" \t\r") + 1);
        };
        trim(line);
        if (line.empty() || line[0] == '#') continue;

        // 检测section
        if (line.back() == ':') {
            section = line.substr(0, line.size() - 1);
            continue;
        }

        // 解析 key: value 对
        auto colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = line.substr(0, colon_pos);
        std::string value_str = line.substr(colon_pos + 1);
        trim(key);
        trim(value_str);

        if (value_str.empty()) {
            // 可能是嵌套section
            section = key;
            continue;
        }

        // 根据section和key设置值
        if (section == "model_conf") {
            if (key == "speech_noise_thres") {
                vad_speech_noise_thres_ = std::stod(value_str);
            } else if (key == "max_end_silence_time") {
                vad_silence_duration_ = std::stoi(value_str);
            } else if (key == "max_single_segment_time") {
                vad_max_len_ = std::stoi(value_str);
            } else if (key == "decibel_thres") {
                decibel_thres_ = std::stof(value_str);
            } else if (key == "snr_thres") {
                snr_thres_ = std::stof(value_str);
            }
        }
    }
}

void FsmnVadOnnx::InitCache() {
    // 模型有4层cache，每层shape为[1, 128, 19, 1]，扁平化为128*19=2432
    in_cache_.clear();
    for (int i = 0; i < 4; ++i) {
        in_cache_.emplace_back(128 * 19, 0.0f);
    }

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
        vad_scorer_.vad_opts.decibel_thres = decibel_thres_;
    }
    if (vad_snr > -1000) {
        snr_thres_ = vad_snr;
        vad_scorer_.vad_opts.snr_thres = snr_thres_;
    }
}

int FsmnVadOnnx::ComputeFrameNum(int sample_length, int frame_sample_length, int frame_shift_sample_length) {
    int frame_num = (sample_length - frame_sample_length) / frame_shift_sample_length + 1;
    if (frame_num >= 1 && sample_length >= frame_sample_length)
        return frame_num;
    else
        return 0;
}

void FsmnVadOnnx::FbankKaldi(const std::vector<float>& waves, std::vector<std::vector<float>>& vad_feats) {
    knf::OnlineFbank fbank(fbank_opts_);

    int frame_number = ComputeFrameNum(waves.size(), frame_sample_length_, frame_shift_sample_length_);
    if (frame_number == 0) return;

    // 缩放音频到原始PCM范围
    std::vector<float> buf(waves.size());
    for (size_t i = 0; i != waves.size(); ++i) {
        buf[i] = waves[i] * 32768;
    }

    fbank.AcceptWaveform((float)sample_rate_, buf.data(), buf.size());
    int32_t frames = fbank.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float* frame = fbank.GetFrame(i);
        std::vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
        vad_feats.emplace_back(frame_vector);
    }
}

int FsmnVadOnnx::OnlineLfrCmvn(std::vector<std::vector<float>>& vad_feats, bool input_finished) {
    std::vector<std::vector<float>> out_feats;
    int T = vad_feats.size();
    int T_lrf = ceil((T - (lfr_m_ - 1) / 2) / (float)lfr_n_);
    int lfr_splice_frame_idxs = T_lrf;
    std::vector<float> p;

    for (int i = 0; i < T_lrf; i++) {
        if (lfr_m_ <= T - i * lfr_n_) {
            for (int j = 0; j < lfr_m_; j++) {
                p.insert(p.end(), vad_feats[i * lfr_n_ + j].begin(), vad_feats[i * lfr_n_ + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            if (input_finished) {
                int num_padding = lfr_m_ - (T - i * lfr_n_);
                for (int j = 0; j < (vad_feats.size() - i * lfr_n_); j++) {
                    p.insert(p.end(), vad_feats[i * lfr_n_ + j].begin(), vad_feats[i * lfr_n_ + j].end());
                }
                for (int j = 0; j < num_padding; j++) {
                    p.insert(p.end(), vad_feats[vad_feats.size() - 1].begin(), vad_feats[vad_feats.size() - 1].end());
                }
                out_feats.emplace_back(p);
                p.clear();
            } else {
                lfr_splice_frame_idxs = i;
                break;
            }
        }
    }

    lfr_splice_frame_idxs = std::min(T - 1, lfr_splice_frame_idxs * lfr_n_);
    lfr_splice_cache_.clear();
    lfr_splice_cache_.insert(lfr_splice_cache_.begin(),
                             vad_feats.begin() + lfr_splice_frame_idxs, vad_feats.end());

    // CMVN归一化
    for (auto& out_feat : out_feats) {
        for (int j = 0; j < (int)means_.size(); j++) {
            out_feat[j] = (out_feat[j] + means_[j]) * vars_[j];
        }
    }

    vad_feats = out_feats;
    return lfr_splice_frame_idxs;
}

void FsmnVadOnnx::ExtractFeats(std::vector<std::vector<float>>& vad_feats,
                                std::vector<float>& waves, bool input_finished) {
    // Step 1: FbankKaldi
    // waves is modified and input_cache_ audio is prepended
    waves.insert(waves.begin(), input_cache_.begin(), input_cache_.end());
    int frame_number = ComputeFrameNum(waves.size(), frame_sample_length_, frame_shift_sample_length_);
    input_cache_.clear();
    if (frame_number > 0) {
        input_cache_.insert(input_cache_.begin(),
                            waves.begin() + frame_number * frame_shift_sample_length_, waves.end());
        waves.erase(waves.begin() + (frame_number - 1) * frame_shift_sample_length_ + frame_sample_length_,
                    waves.end());
    }
    FbankKaldi(waves, vad_feats);

    // Step 2: LFR splice cache management
    if (vad_feats.size() > 0) {
        if (!reserve_waveforms_.empty()) {
            waves.insert(waves.begin(), reserve_waveforms_.begin(), reserve_waveforms_.end());
        }
        if (lfr_splice_cache_.empty()) {
            for (int i = 0; i < (lfr_m_ - 1) / 2; i++) {
                lfr_splice_cache_.emplace_back(vad_feats[0]);
            }
        }
        if ((int)(vad_feats.size() + lfr_splice_cache_.size()) >= lfr_m_) {
            vad_feats.insert(vad_feats.begin(), lfr_splice_cache_.begin(), lfr_splice_cache_.end());

            int frame_from_waves = (waves.size() - frame_sample_length_) / frame_shift_sample_length_ + 1;
            int minus_frame = reserve_waveforms_.empty() ? (lfr_m_ - 1) / 2 : 0;
            int lfr_splice_frame_idxs = OnlineLfrCmvn(vad_feats, input_finished);

            int reserve_frame_idx = std::abs(lfr_splice_frame_idxs - minus_frame);
            reserve_waveforms_.clear();
            reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                      waves.begin() + reserve_frame_idx * frame_shift_sample_length_,
                                      waves.begin() + frame_from_waves * frame_shift_sample_length_);
            int sample_length = (frame_from_waves - 1) * frame_shift_sample_length_ + frame_sample_length_;
            waves.erase(waves.begin() + sample_length, waves.end());
        } else {
            reserve_waveforms_.clear();
            reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                      waves.begin() + frame_sample_length_ - frame_shift_sample_length_,
                                      waves.end());
            lfr_splice_cache_.insert(lfr_splice_cache_.end(), vad_feats.begin(), vad_feats.end());
        }
    } else {
        if (input_finished) {
            if (!reserve_waveforms_.empty()) {
                waves = reserve_waveforms_;
            }
            vad_feats = lfr_splice_cache_;
            if (vad_feats.size() > 0) {
                OnlineLfrCmvn(vad_feats, input_finished);
            }
        }
    }

    if (input_finished) {
        ResetCache();
    }
}

std::vector<std::pair<int, int>> FsmnVadOnnx::Process(const short* pcm_data, int pcm_len, bool is_final) {
    // 转换为float，范围[-1, 1]
    std::vector<float> waves(pcm_len);
    for (int i = 0; i < pcm_len; ++i) {
        waves[i] = pcm_data[i] / 32768.0f;
    }

    // 特征提取 (ExtractFeats会修改waves，只保留被处理的部分)
    std::vector<std::vector<float>> vad_feats;
    ExtractFeats(vad_feats, waves, is_final);
    if (vad_feats.empty()) {
        return {};
    }

    // 推理
    std::vector<std::vector<float>> vad_probs;
    Forward(vad_feats, vad_probs, in_cache_, is_final);
    if (vad_probs.empty()) {
        return {};
    }

    // VAD后处理
    auto segments = vad_scorer_(vad_probs, waves, is_final, true,
                                vad_silence_duration_, vad_max_len_,
                                vad_speech_noise_thres_, sample_rate_);
    // 转换为pair格式，保留所有分段（包括在线模式下[start, -1]和[-1, end]部分分段）
    std::vector<std::pair<int, int>> result;
    for (auto& seg : segments) {
        result.emplace_back(seg[0], seg[1]);
    }

    if (is_final) {
        Reset();
    }

    return result;
}

void FsmnVadOnnx::Forward(const std::vector<std::vector<float>>& feats,
                          std::vector<std::vector<float>>& out_prob,
                          std::vector<std::vector<float>>& in_cache,
                          bool is_final) {
    if (feats.empty()) return;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    int num_frames = feats.size();
    const int feature_dim = feats[0].size();

    // 输入0: speech [1, num_frames, feature_dim]
    const int64_t vad_feats_shape[3] = {1, num_frames, feature_dim};
    std::vector<float> vad_feats;
    for (const auto& feat : feats) {
        vad_feats.insert(vad_feats.end(), feat.begin(), feat.end());
    }
    Ort::Value vad_feats_ort = Ort::Value::CreateTensor<float>(
        memory_info, vad_feats.data(), vad_feats.size(), vad_feats_shape, 3);

    // 构建输入tensors: speech + 4 caches
    std::vector<Ort::Value> vad_inputs;
    vad_inputs.emplace_back(std::move(vad_feats_ort));

    // 4层cache, 每层shape [1, 128, 19, 1]
    const int64_t cache_feats_shape[4] = {1, 128, 19, 1};
    for (size_t i = 0; i < in_cache.size(); i++) {
        vad_inputs.emplace_back(std::move(Ort::Value::CreateTensor<float>(
            memory_info, in_cache[i].data(), in_cache[i].size(), cache_feats_shape, 4)));
    }

    // ONNX推理
    std::vector<Ort::Value> vad_ort_outputs;
    try {
        vad_ort_outputs = session_->Run(
            Ort::RunOptions{nullptr}, input_names_ptr_.data(), vad_inputs.data(),
            vad_inputs.size(), output_names_ptr_.data(), output_names_ptr_.size());
    } catch (std::exception const &e) {
        fprintf(stderr, "Error when run vad onnx forward: %s\n", e.what());
        return;
    }

    // 输出0: logits [1, num_frames, output_dim]
    float* logp_data = vad_ort_outputs[0].GetTensorMutableData<float>();
    auto type_info = vad_ort_outputs[0].GetTensorTypeAndShapeInfo();
    int64_t num_outputs = type_info.GetShape()[1];
    int64_t output_dim = type_info.GetShape()[2];

    out_prob.resize(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        out_prob[i].resize(output_dim);
        memcpy(out_prob[i].data(), logp_data + i * output_dim, sizeof(float) * output_dim);
    }

    // 更新4层cache输出 (仅非最终帧时需要)
    if (!is_final) {
        for (int i = 1; i < 5; i++) {
            float* data = vad_ort_outputs[i].GetTensorMutableData<float>();
            memcpy(in_cache[i - 1].data(), data, sizeof(float) * 128 * 19);
        }
    }
}

} // namespace vad