import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse

import dill as pickle
import collections
from preprocess.preprocessing_library import FFT, Slice, Magnitude, Log10
from utils.pipeline import Pipeline
import numpy as np
from joblib import Parallel, delayed
import warnings

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):
    """
    将给定的音频数据文件转换为FFT表示形式，并进行频率切片、取模和对数转换。

    参数:
    - window_length: 窗口长度（秒），定义了每个FFT分析的时域样本数。1
    - window_step: 窗口步长（秒），定义了每次窗口移动的时域样本数。0.25
    - fft_min_freq: FFT的最小频率（Hz），用于频率域的切片。
    - fft_max_freq: FFT的最大频率（Hz），用于频率域的切片。
    - sampling_frequency: 采样频率（Hz），定义了时域样本之间的间隔。250
    - file_path: 音频数据文件的路径，使用pickle格式存储。

    返回值:
    - named_data: 包含FFT处理后数据的命名数据对象，具有患者ID和发作类型信息。
    - file_name: 输入文件的基名。
    """

    # 忽略特定警告
    warnings.filterwarnings("ignore")
    # 加载音频数据文件
    type_data = pickle.load(open(file_path, 'rb'))
    # 定义处理管道，包括FFT变换、频率切片、取模和对数转换
    pipeline = Pipeline([FFT(), Slice(fft_min_freq, fft_max_freq), Magnitude(), Log10()])
    # 获取时域音频数据
    time_series_data = type_data.data
    # 计算窗口开始和停止的索引
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    # 初始化FFT数据列表
    fft_data = []

    # 对时域数据应用滑动窗口FFT处理
    # 把(20, 41137)的数组的通道数据点数，通过滑动窗口（250），把这个窗口的数据进行变换，得到12个数据点，作为一条数据（20，12）
    #再存入fft_data中（数据数，通道数20，数据点数12）。
    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:, start:stop]  # 提取窗口信号
        fft_window = pipeline.apply(signal_window)  # 应用处理管道
        fft_data.append(fft_window)  # 将FFT结果添加到列表
        start, stop = start + step, stop + step  # 移动窗口

    # 将FFT数据列表转换为数组
    fft_data = np.array(fft_data)
    # 创建并返回命名数据对象，包含FFT数据、患者ID和发作类型
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=fft_data)

    return named_data,os.path.basename(file_path)


def main():
    """
    主函数用于从TUH EEG Seizure数据集中生成FFT图像。

    使用argparse解析命令行参数，包括：
    - save_data_dir: 保存数据的目录路径
    - preprocess_data_dir: 预处理数据的目录路径
    - tuh_eeg_szr_ver: TUH EEG Seizure数据集的版本

    根据提供的参数，对数据进行预处理，包括转换为FFT表示，并保存处理结果。

    参数:
    - 无

    返回值:
    - 无
    """

    # 初始化命令行参数解析器并设置参数
    parser = argparse.ArgumentParser(description='Generate FFT time&freq coefficients from seizure data')
    parser.add_argument('-l', '--save_data_dir', default='E:\dataSet/tusz_save',
                        help='path from resampled seizure data')
    parser.add_argument('-b', '--preprocess_data_dir', default='E:\dataSet\preprocess_data_images_shanchu',
                        help='path to processed data')
    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.5.2',
                        help='path to output prediction')
    args = parser.parse_args()

    # 解析命令行参数并设置目录路径
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver
    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures')
    preprocess_data_dir = os.path.join(args.preprocess_data_dir, tuh_eeg_szr_ver, 'fft')

    # 收集数据文件名
    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)
    fpaths = [os.path.join(save_data_dir, f) for f in fnames]

    # 定义处理参数
    sampling_frequency = 250  # 采样频率 (Hz)
    fft_min_freq = 1  # FFT最小频率 (Hz)

    # 定义窗口长度和步长进行循环处理
    window_lengths = [1]#[0.25, 0.5, 1]#[1, 2, 4, 8, 16]
    fft_max_freqs = [12]#[12, 24]

    # 开始处理每个窗口长度和步长的组合
    for window_length in window_lengths:
        window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))
        print(window_steps)
        for window_step in window_steps:
            for fft_max_freq_actual in fft_max_freqs:
                # 计算FFT最大频率并创建保存目录
                fft_max_freq = fft_max_freq_actual * window_length
                fft_max_freq = int(np.floor(fft_max_freq))
                print('window length: ', window_length, 'window step: ', window_step, 'fft_max_freq', fft_max_freq)
                save_data_dir = os.path.join(preprocess_data_dir, 'fft_seizures_' + 'wl' + str(window_length) + '_ws_' + str(window_step) \
                                + '_sf_' + str(sampling_frequency) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
                                str(fft_max_freq_actual))
                if not os.path.exists(save_data_dir):
                    os.makedirs(save_data_dir)
                else:
                    exit('Pre-processed data already exists!')

                # 遍历文件并转换为FFT，保存处理结果
                for file_path in sorted(fpaths):
                    converted_data, file_name_base = convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path)
                    if converted_data.data.ndim == 3:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir, file_name_base), 'wb'))





if __name__ == '__main__':
    main()


