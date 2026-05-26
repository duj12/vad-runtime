#!/usr/bin/env python3
"""
RKNN PC 模拟器 VAD 推理脚本

用实际音频文件跑完整 VAD 流程: 音频 → Fbank → LFR+CMVN → RKNN 推理 → 端点检测 → 分段结果

用法:
    python3 infer_rknn.py <音频文件>                          # 默认用 examples/xmov.raw
    python3 infer_rknn.py /path/to/test.wav
    python3 infer_rknn.py --raw /path/to/test.raw             # 16kHz, 16bit, 单声道 raw PCM
    python3 infer_rknn.py --no-state-machine                  # 只看 logits, 不跑状态机

依赖 (在 WSL conda base 环境中已安装):
    pip install kaldi-native-fbank onnxruntime numpy
"""
import os
import sys
import math
import time
import struct
import argparse
from pathlib import Path


# ============================================================
# 1. 特征提取
# ============================================================

class FbankExtractor:
    """使用 kaldi-native-fbank 提取 Fbank 特征 (与 C++ 后端一致)"""
    def __init__(self, sample_rate=16000):
        try:
            import kaldi_native_fbank as knf
        except ImportError:
            print("Error: kaldi-native-fbank not installed.")
            print("Run: pip install kaldi-native-fbank")
            sys.exit(1)

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = sample_rate
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0.0
        opts.frame_opts.preemph_coeff = 0.97
        opts.frame_opts.remove_dc_offset = True
        opts.mel_opts.num_bins = 80
        self.opts = opts
        self.knf = knf

    def extract(self, waveform):
        """提取 Fbank 特征, 返回 [num_frames, 80] float32"""
        import numpy as np
        fbank = self.knf.OnlineFbank(self.opts)
        fbank.accept_waveform(self.opts.frame_opts.samp_freq, waveform)
        fbank.input_finished()

        num_frames = fbank.num_frames_ready
        feats = np.zeros((num_frames, self.opts.mel_opts.num_bins), dtype=np.float32)
        for i in range(num_frames):
            frame = fbank.get_frame(i)
            feats[i] = np.array(frame, dtype=np.float32)
        return feats


# ============================================================
# 2. LFR M=5 + CMVN
# ============================================================

class LfrCmvn:
    """LFR M=5 拼接 + CMVN 归一化, 输出 [num_frames, 400]"""
    def __init__(self, cmvn_path, lfr_m=5, lfr_n=1):
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.means, self.vars = self._load_cmvn(cmvn_path)

    def _load_cmvn(self, cmvn_path):
        """从 am.mvn (Kaldi文本格式) 解析 means 和 vars"""
        import numpy as np
        with open(cmvn_path, 'r') as f:
            text = f.read()

        # 找 <AddShift> ... ] 之间的所有数值
        def extract_values(text, marker):
            start = text.find(marker)
            if start < 0:
                return []
            bracket = text.find('[', start)
            if bracket < 0:
                return []
            end = text.find(']', bracket)
            if end < 0:
                return []
            nums_str = text[bracket + 1:end]
            vals = []
            for token in nums_str.split():
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
            return vals

        means_vals = extract_values(text, '<AddShift>')
        vars_vals = extract_values(text, '<Rescale>')

        means = np.array(means_vals[:400], dtype=np.float32)
        vars_arr = np.array(vars_vals[:400], dtype=np.float32)

        if len(means) != 400 or len(vars_arr) != 400:
            print(f"Warning: expected 400 CMVN values, got means={len(means)} vars={len(vars_arr)}")

        return means, vars_arr

    def process(self, feats):
        """LFR M=5 + CMVN, feats: [N, 80], 返回 [N, 400]"""
        import numpy as np
        # 在开头 padding lfr_m-1 帧 (用第0帧复制)
        left_padding = np.tile(feats[0:1], (self.lfr_m - 1, 1))
        feats_pad = np.vstack([left_padding, feats])

        # LFR M=5: 每帧拼接前后帧
        T = feats.shape[0]
        lfr_feats = np.zeros((T, 80 * self.lfr_m), dtype=np.float32)
        for t in range(T):
            for m in range(self.lfr_m):
                lfr_feats[t, m * 80:(m + 1) * 80] = feats_pad[t + m]

        # CMVN: feat = (feat + mean) * var
        out = (lfr_feats + self.means) * self.vars
        return out


# ============================================================
# 3. RKNN 推理
# ============================================================

class RknnInference:
    """RKNN 模型推理 (PC 模拟器)"""
    def __init__(self, onnx_path):
        from rknn.api import RKNN
        self.rknn = RKNN(verbose=False)

        # 配置与转换一致
        ret = self.rknn.config(
            mean_values=None, std_values=None,
            target_platform='rk3588',
            optimization_level=3,
            output_optimize=True,
            dynamic_input=[
                [[1, 1, 400], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1]],
                [[1, 6, 400], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1], [1, 128, 19, 1]],
                [[1, 36, 400],[1, 128, 19, 1],[1, 128, 19, 1],[1, 128, 19, 1],[1, 128, 19, 1]],
            ])
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        ret = self.rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        ret = self.rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        ret = self.rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"rknn.init_runtime(simulator) failed: {ret}")

    def inference(self, feats, in_caches):
        """
        推理一帧
        feats: [1, N, 400] float32
        in_caches: list of 4 [1, 128, 19, 1] NCHW caches
        返回: (logits [1,N,248], out_caches list of 4 [1,128,19,1] NCHW)
        """
        import numpy as np
        # Cache NCHW → NHWC
        inputs = [feats]
        for c in in_caches:
            inputs.append(np.transpose(c, (0, 2, 3, 1)))  # NCHW→NHWC

        outputs = self.rknn.inference(inputs=inputs)

        logits = outputs[0]
        out_caches = [outputs[i + 1] for i in range(4)]  # 已经是 NCHW
        return logits, out_caches

    def release(self):
        self.rknn.release()


# ============================================================
# 4. VAD 端点检测状态机 (移植自 e2e_vad.h)
# ============================================================

from enum import Enum

class FrameState(Enum):
    kFrameStateInvalid = -1
    kFrameStateSpeech = 1
    kFrameStateSil = 0

class AudioChangeState(Enum):
    kChangeStateSil2Speech = 0
    kChangeStateSpeech2Sil = 1
    kChangeStateSpeech2Speech = 2
    kChangeStateSil2Sil = 3

class VadStateMachine(Enum):
    kVadInStateStartPointNotDetected = 0
    kVadInStateInSpeechSegment = 1
    kVadInStateEndPointDetected = 2

class VadDetectMode(Enum):
    kVadSingleUtteranceDetectMode = 0
    kVadMutipleUtteranceDetectMode = 1


class WindowDetector:
    """滑窗状态检测"""
    def __init__(self, window_size_ms=200, sil_to_speech_time=150,
                 speech_to_sil_time=150, frame_size_ms=10):
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time = sil_to_speech_time
        self.speech_to_sil_time = speech_to_sil_time
        self.frame_size_ms = frame_size_ms
        self.win_size = window_size_ms // frame_size_ms

        self.sil_to_speech_frmcnt_thres = sil_to_speech_time // frame_size_ms
        self.speech_to_sil_frmcnt_thres = speech_to_sil_time // frame_size_ms
        self.win_state = [0] * self.win_size
        self.pre_frame_state = FrameState.kFrameStateSil
        self.win_idx = 0

    def Reset(self):
        self.win_state = [0] * self.win_size
        self.pre_frame_state = FrameState.kFrameStateSil
        self.win_idx = 0

    def GetWinSize(self):
        return self.win_size

    def DetectOneFrame(self, frame_state, frame_count):
        if frame_state == FrameState.kFrameStateSpeech:
            self.win_state[self.win_idx] = 1
        elif frame_state == FrameState.kFrameStateSil:
            self.win_state[self.win_idx] = 0

        self.win_idx = (self.win_idx + 1) % self.win_size
        win_sum = sum(self.win_state)

        result = AudioChangeState.kChangeStateSil2Sil
        if self.pre_frame_state == FrameState.kFrameStateSil and win_sum >= self.sil_to_speech_frmcnt_thres:
            self.pre_frame_state = FrameState.kFrameStateSpeech
            result = AudioChangeState.kChangeStateSil2Speech
        if self.pre_frame_state == FrameState.kFrameStateSpeech and win_sum <= self.speech_to_sil_frmcnt_thres:
            self.pre_frame_state = FrameState.kFrameStateSil
            result = AudioChangeState.kChangeStateSpeech2Sil

        if result == AudioChangeState.kChangeStateSil2Sil:
            if self.pre_frame_state == FrameState.kFrameStateSil:
                result = AudioChangeState.kChangeStateSil2Sil
            elif self.pre_frame_state == FrameState.kFrameStateSpeech:
                result = AudioChangeState.kChangeStateSpeech2Speech
        return result


class E2EVadModel:
    """VAD 端点检测模型"""
    def __init__(self, config):
        # 从 config 读取参数
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_in_ms = config.get('frame_in_ms', 10)
        self.frame_length_ms = config.get('frame_length_ms', 25)
        self.detect_mode = config.get('detect_mode', 1)
        self.snr_mode = config.get('snr_mode', 0)
        self.max_end_silence_time = config.get('max_end_silence_time', 800)
        self.max_start_silence_time = config.get('max_start_silence_time', 3000)
        self.do_start_point_detection = config.get('do_start_point_detection', True)
        self.do_end_point_detection = config.get('do_end_point_detection', True)
        self.window_size_ms = config.get('window_size_ms', 200)
        self.sil_to_speech_time_thres = config.get('sil_to_speech_time_thres', 150)
        self.speech_to_sil_time_thres = config.get('speech_to_sil_time_thres', 150)
        self.speech_2_noise_ratio = config.get('speech_2_noise_ratio', 1.0)
        self.do_extend = config.get('do_extend', 1)
        self.lookback_time_start_point = config.get('lookback_time_start_point', 200)
        self.lookahead_time_end_point = config.get('lookahead_time_end_point', 100)
        self.max_single_segment_time = config.get('max_single_segment_time', 60000)
        self.snr_thres = config.get('snr_thres', -100.0)
        self.noise_frame_num_used_for_snr = config.get('noise_frame_num_used_for_snr', 100)
        self.decibel_thres = config.get('decibel_thres', -100.0)
        self.speech_noise_thres = config.get('speech_noise_thres', 0.6)
        self.fe_prior_thres = config.get('fe_prior_thres', 0.0001)
        self.silence_pdf_num = config.get('silence_pdf_num', 1)
        self.sil_pdf_ids = config.get('sil_pdf_ids', [0])
        self.speech_noise_thresh_low = config.get('speech_noise_thresh_low', -0.1)
        self.speech_noise_thresh_high = config.get('speech_noise_thresh_high', 0.3)
        self.output_frame_probs = config.get('output_frame_probs', False)

        self.windows_detector = WindowDetector(
            self.window_size_ms, self.sil_to_speech_time_thres,
            self.speech_to_sil_time_thres, self.frame_in_ms
        )

        self.max_end_sil_frame_cnt_thresh = (
            self.max_end_silence_time - self.speech_to_sil_time_thres
        )

        self.scores = []
        self.decibel = []
        self.frm_cnt = 0
        self.idx_pre_chunk = 0
        self.noise_average_decibel = -100.0
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.continous_silence_frame_count = 0
        self.number_end_time_detected = 0
        self.pre_end_silence_detected = False
        self.max_time_out = False
        self.next_seg = True

        self.output_segments = []  # [(start_ms, end_ms), ...]
        self.current_seg_start = -1

    def ComputeDecibel(self, waveform, data_buf_pos=0):
        """从波形计算每帧分贝"""
        frame_sample_length = self.frame_length_ms * self.sample_rate // 1000
        frame_shift_length = self.frame_in_ms * self.sample_rate // 1000
        for offset in range(0, len(waveform) - frame_sample_length + 1, frame_shift_length):
            frame = waveform[offset:offset + frame_sample_length]
            energy = sum(s * s for s in frame)
            decibel = 10 * math.log10(energy + 0.000001)
            self.decibel.append(decibel)

    def ComputeScores(self, scores):
        """接收模型输出的 scores"""
        self.frm_cnt += len(scores)
        self.scores = scores

    def GetFrameState(self, t):
        """判断第 t 帧的语音/静音状态"""
        frame_state = FrameState.kFrameStateInvalid
        if t >= len(self.decibel) or t >= self.idx_pre_chunk + len(self.scores):
            return FrameState.kFrameStateInvalid

        cur_decibel = self.decibel[t]
        cur_snr = cur_decibel - self.noise_average_decibel

        if cur_decibel < self.decibel_thres:
            return FrameState.kFrameStateSil

        sum_score = 0.0
        noise_prob = 0.0
        if self.sil_pdf_ids:
            sil_pdf_scores = [
                self.scores[t - self.idx_pre_chunk][pid]
                for pid in self.sil_pdf_ids
                if pid < len(self.scores[t - self.idx_pre_chunk])
            ]
            sum_score = sum(sil_pdf_scores)
            noise_prob = math.log(sum_score) * self.speech_2_noise_ratio if sum_score > 0 else -float('inf')
            total_score = 1.0
            sum_score = total_score - sum_score

        speech_prob = math.log(sum_score) if sum_score > 0 else -float('inf')

        if math.exp(speech_prob) >= math.exp(noise_prob) + self.speech_noise_thres:
            if cur_snr >= self.snr_thres and cur_decibel >= self.decibel_thres:
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if self.noise_average_decibel < -99.9:
                self.noise_average_decibel = cur_decibel
            else:
                nf = self.noise_frame_num_used_for_snr
                self.noise_average_decibel = (
                    cur_decibel + self.noise_average_decibel * (nf - 1)
                ) / nf

        return frame_state

    def _maybe_voice_end_last_frame(self, cur_frm_idx, is_final_frame):
        if is_final_frame:
            self.OnVoiceEnd(cur_frm_idx, False, False)

    def OnVoiceStart(self, start_frame, fake_result=False):
        if fake_result:
            return
        if self.confirmed_start_frame != -1:
            return
        self.confirmed_start_frame = start_frame
        # 新的一段
        self.current_seg_start = start_frame * self.frame_in_ms

    def OnVoiceEnd(self, end_frame, fake_result=False, is_last_frame=False):
        if fake_result:
            return
        # 记录分段
        if self.current_seg_start >= 0:
            end_ms = end_frame * self.frame_in_ms
            self.output_segments.append((self.current_seg_start, end_ms))
            self.current_seg_start = -1
        self.confirmed_end_frame = end_frame

    def OnVoiceDetected(self, valid_frame):
        self.latest_confirmed_speech_frame = valid_frame

    def OnSilenceDetected(self, valid_frame):
        self.lastest_confirmed_silence_frame = valid_frame
        if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
            pass  # PopDataBufTillFrame

    def LatencyFrmNumAtStartPoint(self):
        vad_latency = self.windows_detector.GetWinSize()
        if self.do_extend:
            vad_latency += self.lookback_time_start_point // self.frame_in_ms
        return vad_latency

    def DetectOneFrame(self, cur_frm_state, cur_frm_idx, is_final_frame):
        tmp_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            if abs(1.0) > self.fe_prior_thres:
                tmp_state = FrameState.kFrameStateSpeech
            else:
                tmp_state = FrameState.kFrameStateSil
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_state = FrameState.kFrameStateSil

        state_change = self.windows_detector.DetectOneFrame(tmp_state, cur_frm_idx)
        frm_shift_ms = self.frame_in_ms

        if state_change == AudioChangeState.kChangeStateSil2Speech:
            self.continous_silence_frame_count = 0
            self.pre_end_silence_detected = False
            if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                # 不精确计算 lookback, 直接用 cur_frm_idx 减延迟
                lookback = self.LatencyFrmNumAtStartPoint()
                start_frame = max(0, cur_frm_idx - lookback)
                self.OnVoiceStart(start_frame)
                self.vad_state_machine = VadStateMachine.kVadInStateInSpeechSegment
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if cur_frm_idx - self.confirmed_start_frame + 1 > self.max_single_segment_time // frm_shift_ms:
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    self._maybe_voice_end_last_frame(cur_frm_idx, is_final_frame)

        elif state_change == AudioChangeState.kChangeStateSpeech2Sil:
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if cur_frm_idx - self.confirmed_start_frame + 1 > self.max_single_segment_time // frm_shift_ms:
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    self._maybe_voice_end_last_frame(cur_frm_idx, is_final_frame)

        elif state_change == AudioChangeState.kChangeStateSpeech2Speech:
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if cur_frm_idx - self.confirmed_start_frame + 1 > self.max_single_segment_time // frm_shift_ms:
                    self.max_time_out = True
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    self._maybe_voice_end_last_frame(cur_frm_idx, is_final_frame)

        elif state_change == AudioChangeState.kChangeStateSil2Sil:
            self.continous_silence_frame_count += 1
            if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                sil_time = self.continous_silence_frame_count * frm_shift_ms
                if (self.detect_mode == VadDetectMode.kVadSingleUtteranceDetectMode.value
                        and sil_time > self.max_start_silence_time) or (is_final_frame and self.number_end_time_detected == 0):
                    self.OnVoiceStart(0, True)
                    self.OnVoiceEnd(0, True, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                sil_time = self.continous_silence_frame_count * frm_shift_ms
                if sil_time >= self.max_end_sil_frame_cnt_thresh:
                    lookback = self.max_end_sil_frame_cnt_thresh // frm_shift_ms
                    if self.do_extend:
                        lookback -= self.lookahead_time_end_point // frm_shift_ms
                        lookback -= 1
                        lookback = max(0, lookback)
                    self.OnVoiceEnd(cur_frm_idx - lookback, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif cur_frm_idx - self.confirmed_start_frame + 1 > self.max_single_segment_time // frm_shift_ms:
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    self._maybe_voice_end_last_frame(cur_frm_idx, is_final_frame)

        if (self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected
                and self.detect_mode == VadDetectMode.kVadMutipleUtteranceDetectMode.value):
            self._reset_detection()

    def _reset_detection(self):
        self.continous_silence_frame_count = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.windows_detector.Reset()
        self.pre_end_silence_detected = False

    def DetectCommonFrames(self):
        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return
        for i in range(self.scores.shape[0] - 1, -1, -1):
            t = self.frm_cnt - len(self.scores) + i
            if t >= len(self.decibel):
                continue
            frame_state = self.GetFrameState(t)
            self.DetectOneFrame(frame_state, t, False)
        self.idx_pre_chunk += len(self.scores)

    def DetectLastFrames(self):
        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return
        for i in range(self.scores.shape[0] - 1, -1, -1):
            t = self.frm_cnt - len(self.scores) + i
            if t >= len(self.decibel):
                continue
            frame_state = self.GetFrameState(t)
            if i != 0:
                self.DetectOneFrame(frame_state, t, False)
            else:
                self.DetectOneFrame(frame_state, t, True)

    def ComputeDecibelFromBuf(self, all_waveform):
        """从累积波形计算分贝 (匹配 C++ data_buf_all_size 语义)"""
        frame_sample_length = self.frame_length_ms * self.sample_rate // 1000
        frame_shift_length = self.frame_in_ms * self.sample_rate // 1000
        new_decibel = []
        for offset in range(0, len(all_waveform) - frame_sample_length + 1, frame_shift_length):
            frame = all_waveform[offset:offset + frame_sample_length]
            energy = sum(s * s for s in frame)
            db = 10 * math.log10(energy + 0.000001)
            new_decibel.append(db)
        self.decibel = new_decibel


def load_config(config_path):
    """加载 config.yaml"""
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('model_conf', cfg)


def load_audio(audio_path, is_raw=False, target_sr=16000):
    """加载音频, 返回 float32 list"""
    import numpy as np
    if is_raw:
        # 16kHz, 16bit, 单声道 raw PCM
        raw = np.fromfile(audio_path, dtype=np.int16).astype(np.float32)
        raw /= 32768.0
        return raw.tolist()
    else:
        import soundfile as sf
        data, sr = sf.read(audio_path)
        if sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        if isinstance(data[0], np.ndarray) or hasattr(data[0], '__len__'):
            data = data[:, 0]  # mono
        return data.astype(np.float32).tolist()


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='RKNN VAD 推理 (PC 模拟器)')
    parser.add_argument('audio', nargs='?',
                        default=os.path.normpath(os.path.join(
                            os.path.dirname(__file__), '..', 'examples', 'xmov.raw')),
                        help='音频文件路径 (WAV 或 raw PCM)')
    parser.add_argument('--raw', action='store_true', help='输入为 raw PCM (16kHz, 16bit, mono)')
    parser.add_argument('--model-dir', default=os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'model')),
        help='模型目录 (包含 model.onnx, am.mvn, config.yaml)')
    parser.add_argument('--no-state-machine', action='store_true',
                        help='不跑状态机, 只输出每帧 logits')
    args = parser.parse_args()

    model_dir = args.model_dir
    onnx_path = os.path.join(model_dir, 'model.onnx')
    cmvn_path = os.path.join(model_dir, 'am.mvn')
    config_path = os.path.join(model_dir, 'config.yaml')

    for p in [args.audio, onnx_path, cmvn_path, config_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found")
            sys.exit(1)

    print(f"Audio: {args.audio}")
    print(f"Model: {onnx_path}")
    print()

    # 1. 加载配置
    config = load_config(config_path)

    # 2. 加载音频 (.raw 文件自动使用 raw 模式)
    is_raw = args.raw or args.audio.endswith('.raw')
    print("Loading audio...")
    waveform = load_audio(args.audio, is_raw=is_raw)
    duration_ms = len(waveform) / 16  # 16kHz
    print(f"  {len(waveform)} samples, {duration_ms:.0f}ms")

    # 3. 特征提取
    print("Extracting Fbank features...")
    extractor = FbankExtractor()
    feats = extractor.extract(waveform)
    print(f"  Fbank: {feats.shape[0]} frames x {feats.shape[1]} dims")

    # 4. LFR + CMVN
    print("Applying LFR M=5 + CMVN...")
    lfr_cmvn = LfrCmvn(cmvn_path)
    lfr_feats = lfr_cmvn.process(feats)
    print(f"  LFR output: {lfr_feats.shape[0]} frames x {lfr_feats.shape[1]} dims")

    # 5. RKNN 推理
    print("Building RKNN and running inference (PC simulator)...")
    t_start = time.time()
    rknn_infer = RknnInference(onnx_path)
    build_time = time.time() - t_start

    import numpy as np
    # 流式推理: 首帧 36f, 后续 6f
    chunk_size_first = 36
    chunk_size = 6
    total_frames = lfr_feats.shape[0]

    all_logits = []
    in_caches = [np.zeros((1, 128, 19, 1), dtype=np.float32) for _ in range(4)]

    pos = 0
    first_chunk = True
    while pos < total_frames:
        if first_chunk:
            n = min(chunk_size_first, total_frames - pos)
            first_chunk = False
        else:
            n = min(chunk_size, total_frames - pos)

        chunk = lfr_feats[pos:pos + n]
        # pad 到有效尺寸
        if n == 1:
            pass  # 直接用 1 帧
        elif n <= 6:
            if n < 6:
                pad = np.tile(chunk[-1:], (6 - n, 1))
                chunk = np.vstack([chunk, pad])
        else:
            if n < 36:
                pad = np.tile(chunk[-1:], (36 - n, 1))
                chunk = np.vstack([chunk, pad])

        feat_input = chunk[np.newaxis, :, :].astype(np.float32)

        logits, in_caches = rknn_infer.inference(feat_input, in_caches)

        # 取实际帧数的 logits
        actual_logits = logits[0, :n, :]
        all_logits.append(actual_logits)
        pos += n

    rknn_infer.release()
    infer_time = time.time() - t_start - build_time

    all_logits = np.vstack(all_logits)
    print(f"  Logits: {all_logits.shape[0]} frames x {all_logits.shape[1]} dims")
    print(f"  Build: {build_time:.1f}s, Infer: {infer_time*1000:.0f}ms")

    if args.no_state_machine:
        # 只输出 logits
        print(f"\n{'='*60}")
        print(f"每帧 logits (前 5 维)")
        print(f"{'='*60}")
        print(f"  {'frame':>5s}  {'time':>7s}  {'dim0':>8s}  {'dim1':>8s}  {'dim2':>8s}  {'dim3':>8s}  {'dim4':>8s}")
        step = max(1, len(all_logits) // 20)
        for t in range(0, len(all_logits), step):
            time_ms = t * 10
            vals = all_logits[t, :5]
            print(f"  {t:5d}  {time_ms:5d}ms  {vals[0]:8.4f}  {vals[1]:8.4f}  "
                  f"{vals[2]:8.4f}  {vals[3]:8.4f}  {vals[4]:8.4f}")
        return

    # 6. VAD 端点检测
    print("Running VAD state machine...")
    vad_model = E2EVadModel(config)

    # 全量计算 decibel
    vad_model.ComputeDecibelFromBuf(waveform)

    # 分块送入 scores
    n = chunk_size_first
    scores_chunk = all_logits[:n]
    vad_model.ComputeScores(scores_chunk)
    vad_model.DetectCommonFrames()

    pos = n
    while pos < total_frames:
        n = min(chunk_size, total_frames - pos)
        scores_chunk = all_logits[pos:pos + n]
        vad_model.ComputeScores(scores_chunk)
        if pos + n >= total_frames:
            vad_model.DetectLastFrames()
        else:
            vad_model.DetectCommonFrames()
        pos += n

    # 7. 输出结果
    print(f"\n{'='*60}")
    print(f"VAD 分段结果 ({len(vad_model.output_segments)} 段)")
    print(f"{'='*60}")
    if not vad_model.output_segments:
        print("  (未检测到语音)")
    for start_ms, end_ms in vad_model.output_segments:
        dur = end_ms - start_ms
        print(f"  [{start_ms:5d}ms - {end_ms:5d}ms]  ({dur}ms)")

    # 打印每 10 帧的语音/静音状态 (用 logits 再算一遍，避免管道结束后的状态偏移)
    print(f"\n{'='*60}")
    print(f"帧级别状态 (每 10 帧采样)")
    print(f"{'='*60}")
    print(f"  {'frame':>5s}  {'time':>7s}  {'decibel':>8s}  {'state':>8s}")
    step = max(1, total_frames // 30)
    for t in range(0, total_frames, step):
        db = vad_model.decibel[t] if t < len(vad_model.decibel) else -100
        # 用 logits 和 decibel 直接判断状态
        cur_decibel = vad_model.decibel[t]
        cur_snr = cur_decibel - vad_model.noise_average_decibel
        if cur_decibel < vad_model.decibel_thres:
            state_str = "Sil"
        else:
            sum_score = sum(all_logits[t, pid] for pid in vad_model.sil_pdf_ids if pid < all_logits.shape[1])
            sum_score = 1.0 - sum_score  # total_score - sum_score
            if sum_score > 0:
                speech_prob = math.log(sum_score)
                noise_prob = math.log(sum(all_logits[t, :vad_model.silence_pdf_num])) * vad_model.speech_2_noise_ratio
                if math.exp(speech_prob) >= math.exp(noise_prob) + vad_model.speech_noise_thres:
                    state_str = "Speech"
                else:
                    state_str = "Sil"
            else:
                state_str = "Sil"
        print(f"  {t:5d}  {t*10:5d}ms  {db:8.2f}  {state_str:>8s}")

    print(f"\nDuration: {duration_ms:.0f}ms")
    print(f"Total inference time: {infer_time*1000:.0f}ms (simulator)")


if __name__ == '__main__':
    main()
