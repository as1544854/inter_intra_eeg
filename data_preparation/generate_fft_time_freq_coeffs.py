import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse

import dill as pickle
import collections
from preprocess.preprocessing_library import FFTWithTimeFreqCorrelation
from utils.pipeline import Pipeline
import numpy as np
from joblib import Parallel, delayed
import warnings

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):
    """
    将指定文件中的时间序列数据转换为FFT数据。

    参数:
    - window_length: 窗口长度，表示每次FFT分析的数据段长度（秒）。
    - window_step: 窗口步长，表示每次移动窗口的大小（秒）。
    - fft_min_freq: FFT的最小频率（Hz）。
    - fft_max_freq: FFT的最大频率（Hz）。
    - sampling_frequency: 采样频率（Hz）。
    - file_path: 数据文件的路径。

    返回值:
    - named_data: 转换后的FFT数据，包括患者ID和发作类型等信息。
    - file_name: 输入文件的基名。
    """

    # 忽略所有警告
    warnings.filterwarnings("ignore")
    # 从文件加载数据
    type_data = pickle.load(open(file_path, 'rb'))
    # 创建处理管道，包含FFT和时间-频率相关性分析
    pipeline = Pipeline([FFTWithTimeFreqCorrelation(fft_min_freq, fft_max_freq, sampling_frequency, 'first_axis')])
    # 获取时间序列数据
    time_series_data = type_data.data
    # 计算窗口的开始和结束位置
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    # 初始化FFT数据列表
    fft_data = []

    # 通过窗口滑动进行FFT分析，并收集结果
    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:, start:stop]  # 获取当前窗口的数据
        fft_window = pipeline.apply(signal_window)  # 对窗口数据进行FFT分析
        fft_data.append(fft_window)  # 添加FFT结果到列表
        start, stop = start + step, stop + step  # 移动窗口

    # 将FFT数据列表转换为数组
    fft_data = np.array(fft_data)
    # 创建包含FFT数据的新数据结构
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=fft_data)

    return named_data, os.path.basename(file_path)


def main():
    """
    主函数，用于从癫痫数据中生成FFT时间和频率系数。

    参数:
    - 无命令行参数

    返回值:
    - 无
    """

    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description='Generate FFT time&freq coefficients from seizure data')

    # 设置命令行参数：保存数据的目录
    parser.add_argument('-l','--save_data_dir', default='E:\dataSet/tusz_save',
                        help='path from resampled seizure data')
    # 设置命令行参数：预处理数据的目录
    parser.add_argument('-b','--preprocess_data_dir', default='E:\dataSet\preprocess_data_freq_coffs',
                        help='path to processed data')

    # 设置命令行参数：TUH EEG Seizure数据集的版本
    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.5.2',
                        help='path to output prediction')

    # 解析命令行参数
    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    # 设置保存数据的目录和预处理数据的目录
    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures')
    preprocess_data_dir = os.path.join(args.preprocess_data_dir,tuh_eeg_szr_ver,'fft_with_time_freq_corr')

    # 收集所有文件名
    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)

    # 构建文件路径列表
    fpaths = [os.path.join(save_data_dir,f) for f in fnames]

    # 设置采样频率和FFT的最小频率
    sampling_frequency = 250  # Hz
    fft_min_freq = 1  # Hz

    # 设置窗口长度和对应的FFT最大频率
    # window_lengths = [1, 2, 4, 8, 16]#[0.25, 0.5, 1]#[1, 2, 4, 8, 16]
    # fft_max_freqs = [12, 24, 48, 64, 96]#[12, 24]

    window_lengths = [1]  # [0.25, 0.5, 1]#[1, 2, 4, 8, 16]
    fft_max_freqs = [12]  # [12, 24]

    # 遍历不同窗口长度和步长，以及FFT最大频率，进行处理
    for window_length in window_lengths:
        window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))
        for window_step in window_steps:
            for fft_max_freq_actual in fft_max_freqs:
                # 计算FFT的最大频率
                fft_max_freq = fft_max_freq_actual * window_length
                fft_max_freq = int(np.floor(fft_max_freq))
                # 打印当前的处理参数
                print('window length: ', window_length, 'window step: ', window_step, 'fft_max_freq', fft_max_freq)

                # 构建保存路径
                save_data_dir = os.path.join(preprocess_data_dir, 'fft_seizures_' + 'wl' + str(window_length) + '_ws_' + str(window_step) + '_sf_' + \
                                str(sampling_frequency) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
                                str(fft_max_freq_actual))

                # 检查路径是否存在，如果不存在则创建
                if not os.path.exists(save_data_dir):
                    os.makedirs(save_data_dir)
                else:
                    exit('Pre-processed data already exists!')

                # 对每个文件应用FFT转换并保存结果
                for file_path in sorted(fpaths):
                    converted_data,file_name_base = convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency,file_path)
                    if converted_data.data.ndim == 2:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir, file_name_base), 'wb'))


if __name__ == '__main__':
    main()


