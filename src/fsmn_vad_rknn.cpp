#include "fsmn_vad_rknn.h"
#include "common.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iterator>
#include <cstdlib>

#ifdef ENABLE_RKNN

namespace vad {

FsmnVadRknn::FsmnVadRknn() {
    frame_sample_length_ = sample_rate_ / 1000 * 25;
    frame_shift_sample_length_ = sample_rate_ / 1000 * 10;
}

FsmnVadRknn::~FsmnVadRknn() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

void FsmnVadRknn::Init(const std::string& model_dir, int sample_rate, int thread_num) {
    sample_rate_ = sample_rate;
    frame_sample_length_ = sample_rate_ / 1000 * 25;
    frame_shift_sample_length_ = sample_rate_ / 1000 * 10;

    // 模型路径
    std::string model_path = model_dir + "/" + RKNN_MODEL_NAME;
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

void FsmnVadRknn::LoadModel(const std::string& model_path) {
    // 读取模型文件
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open RKNN model: " + model_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> model_data(size);
    if (!file.read(model_data.data(), size)) {
        throw std::runtime_error("Failed to read RKNN model: " + model_path);
    }

    // 初始化RKNN context
    int ret = rknn_init(&ctx_, model_data.data(), size, 0, NULL);
    if (ret < 0) {
        throw std::runtime_error("Failed to init RKNN context, error: " + std::to_string(ret));
    }

    // 查询模型输入输出信息
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret < 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        throw std::runtime_error("Failed to query RKNN input output num, error: " + std::to_string(ret));
    }

    // 获取输入属性
    input_attrs_.resize(io_num_.n_input);
    for (int i = 0; i < io_num_.n_input; ++i) {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            rknn_destroy(ctx_);
            ctx_ = 0;
            throw std::runtime_error("Failed to query RKNN input attr " + std::to_string(i) + ", error: " + std::to_string(ret));
        }
    }

    // 获取输出属性
    output_attrs_.resize(io_num_.n_output);
    for (int i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            rknn_destroy(ctx_);
            ctx_ = 0;
            throw std::runtime_error("Failed to query RKNN output attr " + std::to_string(i) + ", error: " + std::to_string(ret));
        }
    }

    has_dyn_range_ = true;
    input_dyn_range_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        input_dyn_range_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_DYNAMIC_RANGE, &input_dyn_range_[i],
                         sizeof(rknn_input_range));
        if (ret < 0) {
            has_dyn_range_ = false;
            input_dyn_range_.clear();
            break;
        }
    }
}

bool FsmnVadRknn::SetDynamicShapes(int infer_frames) {
    if (!has_dyn_range_ || input_dyn_range_.empty()) {
        return true;
    }

    int group = 0;
    if (infer_frames <= 1) {
        group = 0;
    } else if (infer_frames <= 6) {
        group = 1;
    } else {
        group = 2;
    }
    if (group >= (int)input_dyn_range_[0].shape_number) {
        group = input_dyn_range_[0].shape_number - 1;
    }

    std::vector<rknn_tensor_attr> attrs(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        attrs[i] = input_attrs_[i];
        if (i < input_dyn_range_.size() && group < (int)input_dyn_range_[i].shape_number) {
            for (uint32_t d = 0; d < input_dyn_range_[i].n_dims; ++d) {
                attrs[i].dims[d] = input_dyn_range_[i].dyn_range[group][d];
            }
        }
    }

    int ret = rknn_set_input_shapes(ctx_, io_num_.n_input, attrs.data());
    if (ret < 0) {
        fprintf(stderr, "rknn_set_input_shapes failed: %d (infer_frames=%d group=%d)\n",
                ret, infer_frames, group);
        return false;
    }

    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        input_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_CURRENT_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_CURRENT_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
    }
    return true;
}

void FsmnVadRknn::LoadCmvn(const std::string& cmvn_path) {
    std::ifstream cmvn_stream(cmvn_path);
    if (!cmvn_stream) {
        throw std::runtime_error("Failed to open cmvn file: " + cmvn_path);
    }

    // 解析Kaldi文本格式的CMVN
    int feat_dim = 80 * lfr_m_;
    means_.resize(feat_dim);
    vars_.resize(feat_dim);

    std::string line;
    while (getline(cmvn_stream, line)) {
        std::istringstream iss(line);
        std::vector<std::string> line_item{std::istream_iterator<std::string>{iss},
                                           std::istream_iterator<std::string>{}};
        if (line_item.empty()) continue;

        if (line_item[0] == "<AddShift>") {
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

void FsmnVadRknn::LoadConfig(const std::string& config_path) {
    std::ifstream is(config_path);
    if (!is) {
        fprintf(stderr, "Warning: Failed to open config file: %s, use defaults\n", config_path.c_str());
        return;
    }

    std::string line;
    std::string section;
    while (std::getline(is, line)) {
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r"));
            s.erase(s.find_last_not_of(" \t\r") + 1);
        };
        trim(line);
        if (line.empty() || line[0] == '#') continue;

        if (line.back() == ':') {
            section = line.substr(0, line.size() - 1);
            continue;
        }

        auto colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = line.substr(0, colon_pos);
        std::string value_str = line.substr(colon_pos + 1);
        trim(key);
        trim(value_str);

        if (value_str.empty()) {
            section = key;
            continue;
        }

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

void FsmnVadRknn::InitCache() {
    // 4层cache，每层shape为[1, 128, 19, 1]，扁平化为128*19=2432
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

void FsmnVadRknn::Reset() {
    InitCache();
}

void FsmnVadRknn::SetConfig(int vad_tail_sil, int vad_max_len, int vad_energy, int vad_snr) {
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

int FsmnVadRknn::ComputeFrameNum(int sample_length, int frame_sample_length, int frame_shift_sample_length) {
    int frame_num = (sample_length - frame_sample_length) / frame_shift_sample_length + 1;
    if (frame_num >= 1 && sample_length >= frame_sample_length)
        return frame_num;
    else
        return 0;
}

void FsmnVadRknn::FbankKaldi(const std::vector<float>& waves, std::vector<std::vector<float>>& vad_feats) {
    knf::OnlineFbank fbank(fbank_opts_);

    int frame_number = ComputeFrameNum(waves.size(), frame_sample_length_, frame_shift_sample_length_);
    if (frame_number == 0) return;

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

int FsmnVadRknn::OnlineLfrCmvn(std::vector<std::vector<float>>& vad_feats, bool input_finished) {
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

void FsmnVadRknn::ExtractFeats(std::vector<std::vector<float>>& vad_feats,
                                std::vector<float>& waves, bool input_finished) {
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

void FsmnVadRknn::ResetCache() {
    reserve_waveforms_.clear();
    input_cache_.clear();
    lfr_splice_cache_.clear();
}

std::vector<std::pair<int, int>> FsmnVadRknn::Process(const short* pcm_data, int pcm_len, bool is_final) {
    static int g_chunk_id = 0;
    const int chunk_id = g_chunk_id++;

    std::vector<float> waves(pcm_len);
    for (int i = 0; i < pcm_len; ++i) {
        waves[i] = pcm_data[i] / 32768.0f;
    }

    std::vector<std::vector<float>> vad_feats;
    ExtractFeats(vad_feats, waves, is_final);
    if (vad_feats.empty()) {
        return {};
    }

    std::vector<std::vector<float>> vad_probs;
    Forward(vad_feats, vad_probs, in_cache_, is_final);
    if (vad_probs.empty()) {
        return {};
    }

    const bool dbg = std::getenv("VAD_DEBUG") != nullptr;
    const bool dbg_this = dbg && (chunk_id < 3 || is_final);
    if (dbg_this) {
        const int num_frames = (int)vad_probs.size();
        const int out_dim = num_frames > 0 ? (int)vad_probs[0].size() : 0;
        int argmax_counts[16] = {0};
        int argmax_other = 0;
        double avg_max = 0.0;
        for (int i = 0; i < num_frames; i++) {
            int best_k = 0;
            float best_v = vad_probs[i].empty() ? 0.0f : vad_probs[i][0];
            for (int k = 1; k < (int)vad_probs[i].size(); k++) {
                if (vad_probs[i][k] > best_v) {
                    best_v = vad_probs[i][k];
                    best_k = k;
                }
            }
            if (best_k >= 0 && best_k < 16) argmax_counts[best_k]++; else argmax_other++;
            avg_max += best_v;
        }
        avg_max = num_frames > 0 ? (avg_max / num_frames) : 0.0;

        fprintf(stderr,
                "[VAD_DEBUG][RKNN] chunk=%d pcm_len=%d is_final=%d feats=%zu probs=%zu dim=%d waves_used=%zu avg_max=%.6f argmax0=%d argmax1=%d argmax2=%d argmax3=%d argmax4=%d other=%d\n",
                chunk_id, pcm_len, (int)is_final,
                vad_feats.size(), vad_probs.size(), out_dim, waves.size(), avg_max,
                argmax_counts[0], argmax_counts[1], argmax_counts[2], argmax_counts[3], argmax_counts[4], argmax_other);
    }

    auto segments = vad_scorer_(vad_probs, waves, is_final, true,
                                vad_silence_duration_, vad_max_len_,
                                vad_speech_noise_thres_, sample_rate_);

    if (dbg_this) {
        std::string segs;
        for (const auto& seg : segments) {
            if (!segs.empty()) segs += " ";
            if (seg.size() >= 2) {
                segs += "[" + std::to_string(seg[0]) + "," + std::to_string(seg[1]) + "]";
            }
        }
        fprintf(stderr, "[VAD_DEBUG][RKNN] chunk=%d segments=%s\n", chunk_id, segs.c_str());
    }

    std::vector<std::pair<int, int>> result;
    for (auto& seg : segments) {
        result.emplace_back(seg[0], seg[1]);
    }

    if (is_final) {
        Reset();
    }

    return result;
}

void FsmnVadRknn::Forward(const std::vector<std::vector<float>>& feats,
                          std::vector<std::vector<float>>& out_prob,
                          std::vector<std::vector<float>>& in_cache,
                          bool /*is_final*/) {
    if (feats.empty()) return;

    const int num_frames = (int)feats.size();
    const int feat_dim = (int)feats[0].size();
    const bool dbg = std::getenv("VAD_DEBUG") != nullptr;

    // RKNN dynamic_input nominally supports {1,6,36}, but on RK3588 we observed that
    // infer_frames=36 may yield only 1 output frame (shape reported as [1,1,248]).
    // That breaks online VAD behavior. To match ONNX behavior, run in 6-frame micro-batches.
    out_prob.clear();
    out_prob.reserve(num_frames);

    int offset = 0;
    while (offset < num_frames) {
        const int sub_frames = std::min(6, num_frames - offset);
        const int infer_frames = (sub_frames <= 1) ? 1 : 6;

        if (dbg && offset == 0) {
            fprintf(stderr, "[VAD_DEBUG][RKNN] forward num_frames=%d (micro-batch infer_frames=%d)\n",
                    num_frames, infer_frames);
        }

        std::vector<float> input_data(infer_frames * feat_dim, 0.0f);
        for (int i = 0; i < sub_frames; i++) {
            memcpy(&input_data[i * feat_dim], feats[offset + i].data(), sizeof(float) * feat_dim);
        }

        // Cache NCHW -> NHWC
        std::vector<std::vector<float>> cache_nhwc(in_cache.size());
        for (size_t i = 0; i < in_cache.size(); i++) {
            cache_nhwc[i].resize(128 * 19, 0.0f);
            for (int c = 0; c < 128; c++) {
                for (int h = 0; h < 19; h++) {
                    cache_nhwc[i][h * 128 + c] = in_cache[i][c * 19 + h];
                }
            }
        }

        if (!SetDynamicShapes(infer_frames)) {
            return;
        }

        std::vector<rknn_input> inputs(io_num_.n_input);
        memset(inputs.data(), 0, sizeof(rknn_input) * inputs.size());

        inputs[0].index = 0;
        inputs[0].type = input_attrs_[0].type;
        inputs[0].size = input_attrs_[0].n_elems * sizeof(float);
        inputs[0].buf = input_data.data();
        inputs[0].pass_through = 0;
        inputs[0].fmt = input_attrs_[0].fmt;

        for (size_t i = 0; i < in_cache.size() && (1 + i) < io_num_.n_input; i++) {
            const uint32_t idx = 1 + static_cast<uint32_t>(i);
            inputs[idx].index = idx;
            inputs[idx].type = input_attrs_[idx].type;
            inputs[idx].size = input_attrs_[idx].n_elems * sizeof(float);
            inputs[idx].buf = cache_nhwc[i].data();
            inputs[idx].pass_through = 0;
            inputs[idx].fmt = input_attrs_[idx].fmt;
        }

        int ret = rknn_inputs_set(ctx_, io_num_.n_input, inputs.data());
        if (ret < 0) {
            fprintf(stderr, "RKNN inputs_set failed: %d (frames=%d infer=%d)\n", ret, num_frames, infer_frames);
            return;
        }

        ret = rknn_run(ctx_, nullptr);
        if (ret < 0) {
            fprintf(stderr, "RKNN run failed: %d\n", ret);
            return;
        }

        std::vector<rknn_output> outputs(io_num_.n_output);
        memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
        for (uint32_t i = 0; i < io_num_.n_output; i++) {
            outputs[i].want_float = 1;
        }

        ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
        if (ret < 0) {
            fprintf(stderr, "RKNN outputs_get failed: %d\n", ret);
            return;
        }

        float* logits_data = static_cast<float*>(outputs[0].buf);
        if (!logits_data) {
            fprintf(stderr, "RKNN output logits buffer is null\n");
            rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
            return;
        }

        const int out0_elems = (outputs[0].size > 0) ? (int)(outputs[0].size / sizeof(float)) : 0;
        if (infer_frames <= 0 || out0_elems <= 0 || (out0_elems % infer_frames) != 0) {
            rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
            fprintf(stderr, "RKNN invalid output size: out0_elems=%d infer_frames=%d\n", out0_elems, infer_frames);
            return;
        }
        const int output_dim = out0_elems / infer_frames;

        // Append per-frame logits (only real frames, not padded)
        for (int i = 0; i < sub_frames; i++) {
            out_prob.emplace_back(output_dim);
            memcpy(out_prob.back().data(), logits_data + i * output_dim, sizeof(float) * output_dim);
        }

        // Update cache outputs 1-4 (NHWC -> NCHW)
        int output_cache_count = std::min({(int)in_cache.size(), (int)io_num_.n_output - 1, 4});
        for (int i = 0; i < output_cache_count; i++) {
            float* cache_out_data = static_cast<float*>(outputs[i + 1].buf);
            for (int c = 0; c < 128; c++) {
                for (int h = 0; h < 19; h++) {
                    in_cache[i][c * 19 + h] = cache_out_data[h * 128 + c];
                }
            }
        }

        rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
        offset += sub_frames;
    }
}

} // namespace vad

#endif // ENABLE_RKNN
