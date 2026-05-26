#!/usr/bin/env python3
"""
ONNX模型转RKNN模型脚本
支持RK3588平台
需要安装RKNN Toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
"""
import os
import argparse
import numpy as np
from rknn.api import RKNN

def convert_onnx_to_rknn(onnx_model_path, rknn_model_path, target_platform="rk3588",
                         do_quantization=False, dataset_path=None):
    """
    将ONNX模型转换为RKNN模型
    :param onnx_model_path: 输入ONNX模型路径
    :param rknn_model_path: 输出RKNN模型路径
    :param target_platform: 目标平台，默认rk3588
    :param do_quantization: 是否量化，默认False
    :param dataset_path: 量化数据集路径，每行一个样本文件路径
    :return: 成功返回True，失败返回False
    """
    # 创建RKNN对象
    rknn = RKNN()

    # 配置模型预处理参数
    print('--> Config model')
    # VAD模型输入是归一化后的音频特征，不需要额外的均值和归一化
    rknn.config(
        mean_values=[[0.0]],
        std_values=[[1.0]],
        target_platform=target_platform,
        optimization_level=3,
        quantize_input_node=do_quantization,
        output_optimize=1
    )
    print('done')

    # 加载ONNX模型
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print('Load ONNX model failed!')
        return False
    print('done')

    # 构建模型
    print('--> Building model')
    if do_quantization and dataset_path:
        ret = rknn.build(do_quantization=do_quantization, dataset=dataset_path)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return False
    print('done')

    # 导出RKNN模型
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export RKNN model failed!')
        return False
    print('done')

    # 释放资源
    rknn.release()
    return True

def generate_calibration_dataset(model_dir, output_dataset_file):
    """
    生成量化校准数据集
    :param model_dir: 模型目录，包含测试音频文件
    :param output_dataset_file: 输出数据集文件路径
    """
    # 这里可以根据实际情况生成校准数据
    # VAD模型的校准数据应该是典型的语音特征
    # 简单起见，生成随机的特征数据作为示例
    np.random.seed(42)
    num_samples = 100
    dataset_dir = os.path.join(os.path.dirname(output_dataset_file), "calibration_data")
    os.makedirs(dataset_dir, exist_ok=True)

    with open(output_dataset_file, 'w') as f:
        for i in range(num_samples):
            # 生成随机特征：[1, 16, 400] 是输入维度示例，实际根据模型调整
            feat = np.random.randn(1, 16, 400).astype(np.float32)
            feat_path = os.path.join(dataset_dir, f"sample_{i}.npy")
            np.save(feat_path, feat)
            f.write(f"{feat_path}\n")

    print(f"Generated calibration dataset: {output_dataset_file}")
    return output_dataset_file

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to RKNN model for RK3588')
    parser.add_argument('onnx_model', help='Input ONNX model path')
    parser.add_argument('rknn_model', help='Output RKNN model path')
    parser.add_argument('--platform', default='rk3588', help='Target platform, default: rk3588')
    parser.add_argument('--quantize', action='store_true', help='Enable quantization')
    parser.add_argument('--dataset', help='Calibration dataset file for quantization')
    parser.add_argument('--generate-dataset', action='store_true', help='Generate random calibration dataset')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.onnx_model):
        print(f"ONNX model not found: {args.onnx_model}")
        return -1

    # 如果需要生成数据集
    dataset_path = args.dataset
    if args.quantize and args.generate_dataset and not dataset_path:
        dataset_path = os.path.join(os.path.dirname(args.rknn_model), "dataset.txt")
        generate_calibration_dataset(os.path.dirname(args.onnx_model), dataset_path)

    # 转换模型
    success = convert_onnx_to_rknn(
        args.onnx_model,
        args.rknn_model,
        target_platform=args.platform,
        do_quantization=args.quantize,
        dataset_path=dataset_path
    )

    if success:
        print(f"Successfully converted {args.onnx_model} to {args.rknn_model}")
        return 0
    else:
        print("Conversion failed!")
        return -1

if __name__ == '__main__':
    main()