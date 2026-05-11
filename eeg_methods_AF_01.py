import sys
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import threading
from scipy.signal import hilbert
from Data_extractions import extract_block_data_from_subject, extract_data_from_subject
from Data_processing import filter_by_condition, filter_by_class
from Utilitys import ensure_dir, unify_names
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed

plt.switch_backend('Agg')


class TimeoutError(Exception):
    pass


def run_with_timeout(func, args=(), kwargs=None, timeout=600):
    if kwargs is None:
        kwargs = {}
    result = []
    error = []

    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            error.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError("Function call timed out.")
    if error:
        raise error[0]
    return result[0]


# ---------------------- 核心参数配置 ----------------------
root_dir = "G:/InnerSpeech2021/"
save_dir = "G:/InnerSpeech2021-RBEAM/"
save_bool = True
overwrite = True
N_S_list = [4]
datatype = "eeg"
# Conditions_list = [ "Pron", "Inner", "Vis"]
# Classes_list = ["Up", "Down", "Right", "Left"]
Conditions_list =  ["Pron", "Inner", "Vis"]
Classes_list = ["Up", "Down", "Right", "Left"]
random_state = 23
np.random.seed(random_state)

# 频带配置（匹配方法学段落，5个核心功能频带）
bands = [
    (1, 4, 'Delta (1-4 Hz)'),
    (4, 8, 'Theta (4-8 Hz)'),
    (8, 13, 'Alpha (8-13 Hz)'),
    (13, 30, 'Beta (13-30 Hz)'),
    (30, 60, 'Gamma (30-100 Hz)')
]

# 时间参数（20ms窗口，50%重叠）
tmin = 1
tmax = 3.5
win_length = 0.03  # 20ms窗口
win_overlap = 0 # 50%重叠
win_step = win_length * (1 - win_overlap)  # 窗口步长
time_steps = int(np.ceil((tmax - tmin) / win_step))  # 总时间帧数

# 插值网格分辨率
grid_resolution = 64

# ---------------------- 加载电极配置 ----------------------
N_B = 1
N_S = 1
X_S, Y = extract_block_data_from_subject(root_dir, N_S, datatype, N_B=N_B)
Adquisition_eq = "biosemi128"
montage = mne.channels.make_standard_montage(Adquisition_eq)
X_S.set_montage(montage)

# 提取电极二维坐标
electrode_pos = X_S.info['chs']
electrode_x = np.array([ch['loc'][0] for ch in electrode_pos])
electrode_y = np.array([ch['loc'][1] for ch in electrode_pos])
electrode_r = np.sqrt(electrode_x ** 2 + electrode_y ** 2)
max_electrode_r = np.max(electrode_r)

# 创建插值网格（矩形范围）
x_grid = np.linspace(np.min(electrode_x), np.max(electrode_x), grid_resolution)
y_grid = np.linspace(np.min(electrode_y), np.max(electrode_y), grid_resolution)
grid_x, grid_y = np.meshgrid(x_grid, y_grid)
grid_r = np.sqrt(grid_x ** 2 + grid_y ** 2)

# ---------------------- 遍历数据处理 ----------------------
for N_S in N_S_list:
    subject_dir = os.path.join(save_dir, f"Subject_{N_S}")
    ensure_dir(subject_dir)

    for Cond in Conditions_list:
        event_name = {"All": "All", "Pron": "Pron", "Inner": "Inner", "Vis": "Vis"}.get(Cond, Cond)
        event_dir = os.path.join(subject_dir, event_name)
        ensure_dir(event_dir)

        for Classes in Classes_list:
            # 加载并筛选数据
            X_s, Y = extract_data_from_subject(root_dir, N_S, datatype)
            X_cond, Y_cond = filter_by_condition(X_s, Y, condition=Cond)
            X_cond, Y_cond = filter_by_class(X_cond, Y_cond, class_condition=Classes)

            # 简化：取前10个试次（可根据需求调整）
            X_cond = X_cond[:25]
            Y_cond = Y_cond[:25]

            # 遍历每个试次
            for trial_index in range(len(X_cond)):
                trial_X = mne.EpochsArray(X_cond[trial_index:trial_index + 1], X_S.info)
                trial_X.set_montage(montage)
                trial_data = trial_X.get_data()[0]  # 形状：(n_channels, n_times)
                sfreq = trial_X.info['sfreq']  # 采样频率

                # 计算时间点（匹配tmin-tmax）
                n_times_total = trial_data.shape[1]
                time_points = np.linspace(0, n_times_total / sfreq, n_times_total)
                valid_time_mask = (time_points >= tmin) & (time_points <= tmax)
                valid_times = time_points[valid_time_mask]
                trial_data_valid = trial_data[:, valid_time_mask]  # 截取有效时间段数据

                # 试次结果保存目录
                trial_result_dir = os.path.join(event_dir, f"Result_{trial_index + 1}")
                ensure_dir(trial_result_dir)
                stimulus_dir = os.path.join(trial_result_dir, Classes)
                ensure_dir(stimulus_dir)

                # ---------------------- 多频带处理 ----------------------
                for band_idx, (fmin_band, fmax_band, band_name) in enumerate(bands):
                    band_dir = os.path.join(stimulus_dir, band_name.replace(" ", "_"))
                    ensure_dir(band_dir)
                    ensure_dir(os.path.join(band_dir, "Amplitude"))  # 幅值图保存目录
                    ensure_dir(os.path.join(band_dir, "Phase"))  # 相位图保存目录

                    # 1. 带通滤波（FIR滤波器，匹配方法学）
                    filtered_data = mne.filter.filter_data(
                        trial_data_valid,
                        sfreq=sfreq,
                        l_freq=fmin_band,
                        h_freq=fmax_band,
                        method='fir',
                        fir_design='firwin',
                        fir_window='hann',
                        phase='zero-double',
                        filter_length='auto',  # 自动调整滤波器长度
                        verbose=False
                    )

                    # 2. 希尔伯特变换提取相位和幅值
                    analytic_signal = hilbert(filtered_data, axis=-1)  # 复数解析信号
                    amplitude = np.abs(analytic_signal)  # 瞬时幅值：(n_channels, n_valid_times)
                    phase = np.angle(analytic_signal)  # 瞬时相位：(n_channels, n_valid_times)，范围[-π, π]

                    # 3. 滑动时间窗平均（20ms窗口，50%重叠）
                    for step in range(time_steps):
                        # 计算当前窗口的时间范围
                        win_start = int(step * win_step * sfreq)
                        win_end = int(win_start + win_length * sfreq)
                        if win_end > amplitude.shape[1]:
                            win_end = amplitude.shape[1]
                            if win_start >= win_end:
                                break  # 超出有效数据范围，停止当前试次

                        # 窗口内平均（时间维度）
                        avg_amplitude = np.mean(amplitude[:, win_start:win_end], axis=1)  # (n_channels,)
                        avg_phase = np.mean(phase[:, win_start:win_end], axis=1)  # (n_channels,)

                        # 计算当前窗口的实际时间范围（用于保存文件名）
                        current_tmin = valid_times[win_start] if win_start < len(valid_times) else valid_times[-1]
                        current_tmax = valid_times[win_end - 1] if win_end - 1 < len(valid_times) else valid_times[-1]

                        # ---------------------- Kriging插值（幅值+相位） ----------------------
                        # 定义Kriging最优模型选择函数（复用原有逻辑）
                        def calculate_mse(model, x, y, z):
                            loo = LeaveOneOut()
                            mse = 0
                            for train_idx, test_idx in loo.split(x):
                                x_train, x_test = x[train_idx], x[test_idx]
                                y_train, y_test = y[train_idx], y[test_idx]
                                z_train, z_test = z[train_idx], z[test_idx]
                                try:
                                    OK = OrdinaryKriging(x_train, y_train, z_train, variogram_model=model)
                                    z_pred, _ = OK.execute('points', x_test, y_test)
                                    mse += (z_pred - z_test) ** 2
                                except:
                                    mse += np.inf
                            return mse / len(x) if len(x) > 0 else np.inf

                        # 幅值插值
                        models = ['linear', 'spherical', 'exponential', 'gaussian']
                        mse_amp = Parallel(n_jobs=-1)(delayed(calculate_mse)(m, electrode_x, electrode_y, avg_amplitude) for m in models)
                        best_model_amp = models[np.argmin(mse_amp)]
                        OK_amp = OrdinaryKriging(electrode_x, electrode_y, avg_amplitude, variogram_model=best_model_amp)
                        amp_grid, _ = OK_amp.execute('grid', x_grid, y_grid)

                        # 相位插值（注意相位的周期性，插值后保持[-π, π]范围）
                        mse_phase = Parallel(n_jobs=-1)(delayed(calculate_mse)(m, electrode_x, electrode_y, avg_phase) for m in models)
                        best_model_phase = models[np.argmin(mse_phase)]
                        OK_phase = OrdinaryKriging(electrode_x, electrode_y, avg_phase, variogram_model=best_model_phase)
                        phase_grid, _ = OK_phase.execute('grid', x_grid, y_grid)
                        phase_grid = np.mod(phase_grid + np.pi, 2 * np.pi) - np.pi  # 归一化到[-π, π]

                        # ---------------------- 矩形掩码与边缘扩散（调整为矩形范围） ----------------------
                        # 幅值场处理
                        mask = (grid_x >= np.min(electrode_x)) & (grid_x <= np.max(electrode_x)) & \
                               (grid_y >= np.min(electrode_y)) & (grid_y <= np.max(electrode_y))
                        extended_amp = amp_grid.copy()
                        edge_mask_amp = np.logical_and(
                            np.logical_or(grid_x == np.min(electrode_x), grid_x == np.max(electrode_x)),
                            np.logical_and(grid_y >= np.min(electrode_y), grid_y <= np.max(electrode_y))
                        ) | np.logical_and(
                            np.logical_or(grid_y == np.min(electrode_y), grid_y == np.max(electrode_y)),
                            np.logical_and(grid_x > np.min(electrode_x), grid_x < np.max(electrode_x))
                        )
                        edge_values_amp = amp_grid[edge_mask_amp] if edge_mask_amp.any() else np.array([0])

                        # 相位场处理
                        extended_phase = phase_grid.copy()
                        extended_phase[~mask] = 0  # 外部区域设为0

                        # 幅值场边缘扩散
                        outer_mask = ~mask
                        outer_x = grid_x[outer_mask]
                        outer_y = grid_y[outer_mask]
                        if len(outer_x) > 0 and edge_mask_amp.any():
                            outer_distances = np.array([
                                np.min(np.sqrt((x - grid_x[edge_mask_amp])**2 + (y - grid_y[edge_mask_amp])**2))
                                for x, y in zip(outer_x, outer_y)
                            ])
                            max_dist = outer_distances.max() if outer_distances.max() > 0 else 1
                            outer_distances /= max_dist
                            for i, (x, y, dist) in enumerate(zip(outer_x, outer_y, outer_distances)):
                                edge_idx = np.argmin(np.sqrt((x - grid_x[edge_mask_amp])**2 + (y - grid_y[edge_mask_amp])**2))
                                extended_amp[outer_mask][i] = edge_values_amp[edge_idx] * (1 - dist)
                        else:
                            extended_amp[~mask] = 0

                        # ---------------------- 保存矩形相位-幅值地形图 ----------------------
                        # 幅值图
                        fig_amp, ax_amp = plt.subplots(figsize=(20, 15))
                        im_amp = ax_amp.imshow(extended_amp,
                                              extent=[np.min(electrode_x), np.max(electrode_x),
                                                      np.min(electrode_y), np.max(electrode_y)],
                                              origin='lower', cmap='Reds', alpha=0.7)
                        ax_amp.set_xticks([])
                        ax_amp.set_yticks([])
                        for spine in ax_amp.spines.values():
                            spine.set_visible(False)
                        fig_amp.patch.set_facecolor('white')
                        ax_amp.patch.set_facecolor('white')

                        # 相位图（用hsv colormap体现周期性，hue对应相位）
                        fig_phase, ax_phase = plt.subplots(figsize=(20, 15))
                        phase_normalized = (phase_grid + np.pi) / (2 * np.pi)  # 归一化到[0,1]用于colormap
                        im_phase = ax_phase.imshow(phase_normalized,
                                                 extent=[np.min(electrode_x), np.max(electrode_x),
                                                         np.min(electrode_y), np.max(electrode_y)],
                                                 origin='lower', cmap='hsv', alpha=0.7)
                        ax_phase.set_xticks([])
                        ax_phase.set_yticks([])
                        for spine in ax_phase.spines.values():
                            spine.set_visible(False)
                        fig_phase.patch.set_facecolor('white')
                        ax_phase.patch.set_facecolor('white')

                        # 保存文件
                        if save_bool:
                            # 幅值图文件名
                            amp_filename = os.path.join(band_dir, "Amplitude",
                                                      f"Amplitude_{Cond}_{Classes}_{band_name.replace(' ', '_')}_"
                                                      f"t_{current_tmin:.2f}-{current_tmax:.2f}_frame_{step}_rect.png")
                            fig_amp.savefig(amp_filename, bbox_inches='tight', pad_inches=0)

                            # 相位图文件名
                            phase_filename = os.path.join(band_dir, "Phase",
                                                       f"Phase_{Cond}_{Classes}_{band_name.replace(' ', '_')}_"
                                                       f"t_{current_tmin:.2f}-{current_tmax:.2f}_frame_{step}_rect.png")
                            fig_phase.savefig(phase_filename, bbox_inches='tight', pad_inches=0)

                        # 关闭图像，释放内存
                        plt.close(fig_amp)
                        plt.close(fig_phase)

                print(f"Trial {trial_index + 1} - {Cond} - {Classes}: 多频带相位-幅值场构建完成")

print("所有数据处理完成！")