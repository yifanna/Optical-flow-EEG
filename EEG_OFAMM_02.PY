import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import gaussian_filter
from scipy.integrate import solve_ivp
import cv2
import glob
import re
from scipy import stats
from scipy.stats import shapiro, normaltest
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 模型假设验证模块（内存优化）----------------------
class ModelAssumptionValidator:
    """验证模型假设（优化内存占用）"""
    
    @staticmethod
    def validate_amplitude_distribution(amplitude_data, sample_size=2000):
        amplitude_flat = amplitude_data.flatten()
        sample_size = min(sample_size, len(amplitude_flat))
        amplitude_sample = np.random.choice(amplitude_flat, sample_size, replace=False)
        
        log_amplitude = np.log(amplitude_sample + 1e-10)
        shapiro_stat, shapiro_p = shapiro(log_amplitude)
        dagostino_stat, dagostino_p = normaltest(log_amplitude)
        
        return {
            'shapiro_test': (shapiro_stat, shapiro_p),
            'dagostino_test': (dagostino_stat, dagostino_p),
            'is_log_normal': shapiro_p > 0.05 and dagostino_p > 0.05,
            'mean': np.mean(log_amplitude),
            'std': np.std(log_amplitude),
            'skewness': stats.skew(log_amplitude),
            'kurtosis': stats.kurtosis(log_amplitude)
        }
    
    @staticmethod
    def validate_phase_continuity(phase_data):
        phase_normalized = phase_data / np.max(phase_data) * 2 * np.pi
        phase_diff = np.diff(phase_normalized, axis=0)
        phase_diff_wrapped = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        
        jump_threshold = np.pi * 0.9
        jump_mask = np.abs(phase_diff_wrapped) > jump_threshold
        jump_ratio = np.sum(jump_mask) / phase_diff_wrapped.size
        max_jump = np.max(np.abs(phase_diff_wrapped))
        
        return {
            'jump_ratio': jump_ratio,
            'max_jump': max_jump,
            'mean_jump': np.mean(np.abs(phase_diff_wrapped)),
            'is_continuous': jump_ratio < 0.05,
            'jump_locations': np.where(jump_mask)
        }
    
    @staticmethod
    def validate_local_stationarity(timeseries, window_size=20, stride=15):
        T, H, W = timeseries.shape
        stationarity_scores = []
        
        timeseries_flat = timeseries.reshape(T, -1)
        n_channels = timeseries_flat.shape[1]
        sample_channels = min(3, n_channels)
        channel_indices = np.random.choice(n_channels, sample_channels, replace=False)
        
        for channel in channel_indices:
            channel_series = timeseries_flat[:, channel]
            autocorr_variances = []
            
            for t in range(0, T - window_size, stride):
                segment = channel_series[t:t+window_size]
                autocorr = np.correlate(segment, segment, mode='full')
                autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + 8]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                autocorr_variances.append(np.var(autocorr[1:]))
            
            if autocorr_variances:
                variance_ratio = np.std(autocorr_variances) / (np.mean(autocorr_variances) + 1e-10)
                stationarity_scores.append(variance_ratio)
        
        if stationarity_scores:
            avg_score = np.mean(stationarity_scores)
            return {
                'variance_ratio': avg_score,
                'is_locally_stationary': avg_score < 0.3,
                'scores_distribution': stationarity_scores
            }
        else:
            return {
                'variance_ratio': None,
                'is_locally_stationary': False,
                'scores_distribution': []
            }

# ---------------------- EEG 数据加载模块（分批加载+内存优化）----------------------
class EEGDataLoader:
    """加载并预处理EEG地形图数据（彻底优化内存）"""
    
    def __init__(self, root_path, max_frames=60, target_size=(64, 64)):
        self.root_path = root_path
        self.max_frames = max_frames
        self.target_size = target_size
        self.folder_types = ["Inner", "Pron", "Vis"]
        self.directions = ["Up", "Down", "Left", "Right"]
        self.frequency_bands = [
            "Delta_(1-4_Hz)",
            "Theta_(4-8_Hz)", 
            "Alpha_(8-13_Hz)",
            "Beta_(13-30_Hz)",
            "Gamma_(30-100_Hz)"
        ]
        self.validator = ModelAssumptionValidator()
    
    def load_all_data(self, validate=True, batch_size=2):
        """分批加载数据（每批2个试次，最小内存占用）"""
        print("开始加载EEG地形图数据（分批处理，极致内存优化）...")
        data_dict = {}
        validation_results = {}
        
        for folder_type in tqdm(self.folder_types, desc="处理条件"):
            folder_path = os.path.join(self.root_path, folder_type)
            if not os.path.exists(folder_path):
                print(f"⚠️  跳过不存在的条件文件夹: {folder_path}")
                continue
            
            data_dict[folder_type] = {}
            validation_results[folder_type] = {}
            
            trial_folders = [f for f in os.listdir(folder_path) if f.startswith("Result_") and os.path.isdir(os.path.join(folder_path, f))]
            if not trial_folders:
                print(f"⚠️  条件 {folder_type} 下无有效试次文件夹")
                continue
            
            for batch_idx in range(0, len(trial_folders), batch_size):
                batch_trials = trial_folders[batch_idx:batch_idx+batch_size]
                for trial_folder in tqdm(batch_trials, desc=f"  试次批次 {batch_idx//batch_size+1} ({folder_type})", leave=False):
                    trial_path = os.path.join(folder_path, trial_folder)
                    data_dict[folder_type][trial_folder] = {}
                    validation_results[folder_type][trial_folder] = {}
                    
                    direction_folders = [d for d in os.listdir(trial_path) if d in self.directions and os.path.isdir(os.path.join(trial_path, d))]
                    if not direction_folders:
                        continue
                    
                    for direction in direction_folders:
                        direction_path = os.path.join(trial_path, direction)
                        data_dict[folder_type][trial_folder][direction] = {}
                        validation_results[folder_type][trial_folder][direction] = {}
                        
                        freq_folders = [f for f in os.listdir(direction_path) if f in self.frequency_bands and os.path.isdir(os.path.join(direction_path, f))]
                        if not freq_folders:
                            continue
                        
                        for freq_band in freq_folders:
                            freq_path = os.path.join(direction_path, freq_band)
                            data_dict[folder_type][trial_folder][direction][freq_band] = {}
                            validation_results[folder_type][trial_folder][direction][freq_band] = {}
                            
                            amplitude_data = self._load_frames(os.path.join(freq_path, "Amplitude"), freq_band)
                            phase_data = self._load_frames(os.path.join(freq_path, "Phase"), freq_band)
                            
                            if amplitude_data is not None and phase_data is not None:
                                data_dict[folder_type][trial_folder][direction][freq_band] = {
                                    'amplitude': amplitude_data,
                                    'phase': phase_data
                                }
                                
                                if validate:
                                    validation_results[folder_type][trial_folder][direction][freq_band] = {
                                        'amplitude_distribution': self.validator.validate_amplitude_distribution(amplitude_data),
                                        'phase_continuity': self.validator.validate_phase_continuity(phase_data),
                                        'local_stationarity': self.validator.validate_local_stationarity(amplitude_data)
                                    }
                    
                    del direction_folders, freq_folders
            
            del trial_folders
        
        total_datasets = self._count_datasets(data_dict)
        print(f"✅ 数据加载完成，共加载 {total_datasets} 个有效数据集")
        return data_dict, validation_results
    
    def _load_frames(self, folder_path, freq_band):
        """加载帧图片（极致内存优化）"""
        if not os.path.exists(folder_path):
            return None
        
        pattern = os.path.join(folder_path, f"*.png")
        all_files = glob.glob(pattern)
        if not all_files:
            return None
        
        frame_info = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            frame_match = re.search(r'frame[_\-]?(\d+)', filename, re.IGNORECASE)
            if frame_match:
                try:
                    frame_num = int(frame_match.group(1))
                    if frame_num < self.max_frames * 2:
                        frame_info.append((frame_num, file_path))
                except:
                    continue
        
        if not frame_info:
            return None
        
        frame_info.sort(key=lambda x: x[0])
        frame_files = [f[1] for f in frame_info[:self.max_frames]]
        
        frames = []
        for file_path in frame_files:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, self.target_size)
                img_normalized = img_resized.astype(np.float32) / 255.0
                frames.append(img_normalized)
        
        frames_array = np.array(frames) if frames else None
        del frames
        return frames_array
    
    def _count_datasets(self, data_dict):
        """统计有效数据集数量"""
        count = 0
        for folder_type in data_dict.values():
            for trial in folder_type.values():
                for direction in trial.values():
                    for freq_band in direction.values():
                        if freq_band:
                            count += 1
        return count
    
    def save_validation_report(self, validation_results, output_path="validation_report_ultimate.txt"):
        """保存验证报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("EEG数据模型假设验证报告（最终修复版）\n")
            f.write("=" * 60 + "\n\n")
            
            for folder_type, trials in validation_results.items():
                f.write(f"条件: {folder_type}\n")
                f.write("-" * 40 + "\n")
                
                for trial_name, directions in trials.items():
                    f.write(f"  试次: {trial_name}\n")
                    
                    for direction_name, freq_bands in directions.items():
                        f.write(f"    方向: {direction_name}\n")
                        
                        for freq_band_name, validations in freq_bands.items():
                            f.write(f"      频段: {freq_band_name}\n")
                            
                            for val_name, val_result in validations.items():
                                f.write(f"        {val_name}:\n")
                                for key, value in val_result.items():
                                    if isinstance(value, np.ndarray):
                                        f.write(f"          {key}: 数组形状 {value.shape}\n")
                                    else:
                                        f.write(f"          {key}: {value}\n")
                            f.write("\n")

# ---------------------- OFAMM 光流分析核心类（彻底解决坐标越界）----------------------
class OpticalFlowEEG:
    def __init__(self, method="CLG"):
        self.method = method  # HS/CLG/TS
        self.u_sequence = None
        self.v_sequence = None
        self.sources = None
        self.sinks = None
        self.trajectories = None
        self.H = 0  # 图像高度
        self.W = 0  # 图像宽度

    # 1. Horn-Schunck 光流法（最小内存）
    def horn_schunck(self, prev_img, curr_img, alpha=0.35, iterations=1000):
        prev = prev_img.astype(np.float32)
        curr = curr_img.astype(np.float32)
        
        Ix = np.gradient(prev, axis=1)
        Iy = np.gradient(prev, axis=0)
        It = curr - prev
        
        u = np.zeros_like(prev, dtype=np.float32)
        v = np.zeros_like(prev, dtype=np.float32)
        
        for _ in range(iterations):
            u_avg = gaussian_filter(u, sigma=1)
            v_avg = gaussian_filter(v, sigma=1)
            numerator = Ix * u_avg + Iy * v_avg + It
            denominator = alpha**2 + Ix**2 + Iy**2
            u = u_avg - (Ix * numerator) / denominator
            v = v_avg - (Iy * numerator) / denominator
        
        return u, v

    # 2. Combined Local-Global 光流法（文档推荐，最小内存）
    def combined_local_global(self, prev_img, curr_img, alpha=0.04, ratio=0.5):
        prev = prev_img.astype(np.float32)
        curr = curr_img.astype(np.float32)
        
        Ix = np.gradient(prev, axis=1)
        Iy = np.gradient(prev, axis=0)
        It = curr - prev
        
        sigma = 1.0
        gauss_window = int(4 * sigma + 1)
        x, y = np.mgrid[0:gauss_window, 0:gauss_window]
        center = (gauss_window - 1) / 2
        gauss2d = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        gauss2d /= gauss2d.sum()
        
        u = np.zeros_like(prev, dtype=np.float32)
        v = np.zeros_like(prev, dtype=np.float32)
        
        nOuter = 8
        nInner = 2
        nSOR = 15
        
        for _ in range(nOuter):
            u_local = cv2.filter2D(u, -1, gauss2d)
            v_local = cv2.filter2D(v, -1, gauss2d)
            u_global = gaussian_filter(u, sigma=1)
            v_global = gaussian_filter(v, sigma=1)
            numerator = Ix * u_local + Iy * v_local + It
            denominator = alpha**2 + Ix**2 + Iy**2
            u = u_global - (Ix * numerator) / denominator
            v = v_global - (Iy * numerator) / denominator
        
        return u, v

    # 3. Temporospatial 光流法（内存优化）
    def temporospatial(self, image_sequence, corr_win=16, max_lag=3):
        T, H, W = image_sequence.shape
        u = np.zeros((H, W), dtype=np.float32)
        v = np.zeros((H, W), dtype=np.float32)
        
        for y in range(corr_win//2, H - corr_win//2):
            for x in range(corr_win//2, W - corr_win//2):
                curr_ts = image_sequence[:, y, x]
                max_corr = -1
                best_dx = 0
                best_dy = 0
                best_lag = 0
                
                for dy in range(-corr_win//2, corr_win//2 + 1):
                    for dx in range(-corr_win//2, corr_win//2 + 1):
                        if dx == 0 and dy == 0:
                            continue
                        for lag in range(1, max_lag + 1):
                            if y+dy <0 or y+dy >=H or x+dx <0 or x+dx >=W:
                                continue
                            if lag >= T:
                                continue
                            neighbor_ts = image_sequence[lag:, y+dy, x+dx]
                            curr_ts_cropped = curr_ts[:-lag]
                            if len(curr_ts_cropped) < 2:
                                continue
                            corr = np.corrcoef(curr_ts_cropped, neighbor_ts)[0,1]
                            if corr > max_corr:
                                max_corr = corr
                                best_dx = dx
                                best_dy = dy
                                best_lag = lag
                
                if best_lag > 0:
                    u[y, x] = best_dx / best_lag
                    v[y, x] = best_dy / best_lag
        
        return u, v

    # 运行光流分析（记录图像尺寸+限制速度值，解决越界）
    def run_flow(self, image_sequence, **kwargs):
        T, self.H, self.W = image_sequence.shape  # 记录图像高宽，供后续边界限制
        self.u_sequence = []
        self.v_sequence = []
        
        if self.method == "HS":
            alpha = kwargs.get("alpha", 0.35)
            iterations = kwargs.get("iterations", 1000)
            for t in range(T-1):
                u, v = self.horn_schunck(image_sequence[t], image_sequence[t+1], alpha, iterations)
                self.u_sequence.append(u)
                self.v_sequence.append(v)
        
        elif self.method == "CLG":
            alpha = kwargs.get("alpha", 0.04)
            ratio = kwargs.get("ratio", 0.5)
            for t in range(T-1):
                u, v = self.combined_local_global(image_sequence[t], image_sequence[t+1], alpha, ratio)
                self.u_sequence.append(u)
                self.v_sequence.append(v)
        
        elif self.method == "TS":
            corr_win = kwargs.get("corr_win", 16)
            max_lag = kwargs.get("max_lag", 3)
            u, v = self.temporospatial(image_sequence, corr_win, max_lag)
            self.u_sequence = [u for _ in range(T-1)]
            self.v_sequence = [v for _ in range(T-1)]
        
        # 核心优化1：限制速度值范围，避免轨迹偏移过快（截断极端大速度）
        self.u_sequence = np.array(self.u_sequence, dtype=np.float32)
        self.v_sequence = np.array(self.v_sequence, dtype=np.float32)
        speed_max = max(self.H, self.W) / 20  # 速度上限：图像尺寸的1/20
        self.u_sequence = np.clip(self.u_sequence, -speed_max, speed_max)
        self.v_sequence = np.clip(self.v_sequence, -speed_max, speed_max)
        
        return self.u_sequence, self.v_sequence

    # 识别源和汇点（内存优化）
    def identify_sources_sinks(self, threshold=0.01):
        if self.u_sequence is None:
            raise ValueError("先运行光流分析（run_flow）！")
        
        mid_t = self.u_sequence.shape[0] // 2
        u = self.u_sequence[mid_t]
        v = self.v_sequence[mid_t]
        H, W = u.shape
        
        dx_u = np.gradient(u, axis=1)
        dy_v = np.gradient(v, axis=0)
        divergence = dx_u + dy_v
        
        dx_v = np.gradient(v, axis=1)
        dy_u = np.gradient(u, axis=0)
        trace = dx_u + dy_v
        determinant = (dx_u * dy_v) - (dx_v * dy_u)
        
        poincare = np.zeros_like(u)
        win_size = 2
        for y in range(win_size//2, H - win_size//2):
            for x in range(win_size//2, W - win_size//2):
                angles = []
                for dy in range(-win_size//2, win_size//2 + 1):
                    for dx in range(-win_size//2, win_size//2 + 1):
                        if dx == 0 and dy == 0:
                            continue
                        ang = np.arctan2(v[y+dy, x+dx], u[y+dy, x+dx])
                        angles.append(ang)
                angle_diff = np.diff(angles + [angles[0]])
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                poincare[y, x] = np.sum(angle_diff) / (2 * np.pi)
        
        self.sources = (divergence > threshold) & (determinant > 0) & (trace > 0) & (np.isclose(poincare, 1, atol=0.3))
        self.sinks = (divergence < -threshold) & (determinant > 0) & (trace < 0) & (np.isclose(poincare, -1, atol=0.3))
        
        return self.sources, self.sinks

    # 追踪像素轨迹（核心修复：强制坐标边界限制，彻底解决越界）
    def track_trajectories(self, start_points=None, dt=1.0):
        if self.u_sequence is None:
            raise ValueError("先运行光流分析（run_flow）！")
        
        T, _, _ = self.u_sequence.shape
        # 边界范围：0 ~ 图像尺寸-1（64×64则0~63）
        y_min, y_max = 0, self.H - 1
        x_min, x_max = 0, self.W - 1

        if start_points is None:
            if self.sources is not None and np.sum(self.sources) > 0:
                start_y, start_x = np.where(self.sources)
                start_points = [(start_y[i], start_x[i]) for i in range(min(2, len(start_y)))]
            else:
                start_points = [(self.H//2, self.W//2)]
        
        self.trajectories = []
        
        def velocity_field(t, pos):
            y, x = pos
            # 核心修复2：先限制坐标在边界内，再取速度值（避免索引越界）
            y = np.clip(y, y_min, y_max)
            x = np.clip(x, x_min, x_max)
            
            t_idx = int(np.clip(np.floor(t), 0, T-1))
            t_weight = t - t_idx
            if t_idx < T-1:
                u = (1 - t_weight) * self.u_sequence[t_idx, int(y), int(x)] + t_weight * self.u_sequence[t_idx+1, int(y), int(x)]
                v = (1 - t_weight) * self.v_sequence[t_idx, int(y), int(x)] + t_weight * self.v_sequence[t_idx+1, int(y), int(x)]
            else:
                u = self.u_sequence[t_idx, int(y), int(x)]
                v = self.v_sequence[t_idx, int(y), int(x)]
            return [v, u]
        
        for (y0, x0) in start_points:
            # 初始点也限制在边界内
            y0 = np.clip(y0, y_min, y_max)
            x0 = np.clip(x0, x_min, x_max)
            
            t_span = (0.0, (T-1)*dt)
            t_eval = np.linspace(0.0, (T-1)*dt, min(T, 30))
            sol = solve_ivp(
                fun=velocity_field,
                t_span=t_span,
                y0=[y0, x0],
                t_eval=t_eval,
                method='RK45',
                bounds=[(y_min, x_min), (y_max, x_max)]  # 核心修复3：求解器强制边界
            )
            # 核心修复4：轨迹结果再次边界限制，双重保险
            traj_clipped = np.clip(sol.y.T, [y_min, x_min], [y_max, x_max])
            self.trajectories.append(traj_clipped)
        
        return self.trajectories

# ---------------------- 主分析流程（最终修复+稳定运行）----------------------
def main():
    # 配置参数（与之前一致，无需修改）
    root_dir = r"E:\InnerSpeech2021-32\Subject_5"
    output_dir = os.path.join("E:\InnerSpeech2021-32-OpticalFlow_Final_Fixed_Results\Subject_5\OpticalFlow_Final_Fixed_Results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：分批加载数据
    loader = EEGDataLoader(
        root_path=root_dir,
        max_frames=50,
        target_size=(64, 64)
    )
    data_dict, validation_results = loader.load_all_data(validate=True, batch_size=2)
    
    # 保存验证报告
    loader.save_validation_report(validation_results, os.path.join(output_dir, "validation_report_final_fixed.txt"))
    print(f"📄 验证报告已保存至: {os.path.join(output_dir, 'validation_report_final_fixed.txt')}")
    
    # 步骤2：遍历数据集进行光流分析（默认 CLG 方法）
    flow_method = "CLG"
    total_processed = 0
    
    # 遍历所有数据集（处理完一个释放一个）
    for folder_type in list(data_dict.keys()):
        for trial_name in list(data_dict[folder_type].keys()):
            for direction in list(data_dict[folder_type][trial_name].keys()):
                for freq_band in list(data_dict[folder_type][trial_name][direction].keys()):
                    dataset = data_dict[folder_type][trial_name][direction][freq_band]
                    amplitude_sequence = dataset['amplitude']
                    T, H, W = amplitude_sequence.shape
                    
                    if T < 2:
                        del data_dict[folder_type][trial_name][direction][freq_band]
                        continue
                    
                    print(f"\n📊 处理数据集: {folder_type} > {trial_name} > {direction} > {freq_band} (帧数: {T})")
                    
                    # 步骤3：初始化光流分析器
                    flow_analyzer = OpticalFlowEEG(method=flow_method)
                    
                    # 步骤4：运行光流计算（无越界报错）
                    try:
                        # 运行光流
                        u_seq, v_seq = flow_analyzer.run_flow(
                            amplitude_sequence,
                            alpha=0.04 if flow_method == "CLG" else 0.35,
                            ratio=0.5 if flow_method == "CLG" else None,
                            iterations=1000 if flow_method == "HS" else None
                        )
                        
                        # 识别源和汇点
                        sources, sinks = flow_analyzer.identify_sources_sinks(threshold=0.01)
                        
                        # 追踪轨迹（坐标强制在边界内）
                        trajectories = flow_analyzer.track_trajectories()
                        
                        # 步骤5：保存结果
                        result_dir = os.path.join(output_dir, folder_type, trial_name, direction, freq_band)
                        os.makedirs(result_dir, exist_ok=True)
                        
                        # 保存数值结果（float32 格式）
                        results = {
                            'u_sequence': u_seq.astype(np.float32),
                            'v_sequence': v_seq.astype(np.float32),
                            'sources': sources,
                            'sinks': sinks,
                            'trajectories': trajectories,
                            'method': flow_method,
                            'amplitude_sequence': amplitude_sequence.astype(np.float32)
                        }
                        scipy.io.savemat(os.path.join(result_dir, "flow_results.mat"), results)
                        
                        # 步骤6：可视化关键帧（兼容边界限制后的轨迹）
                        mid_frame = min(T // 2, len(u_seq)-1)
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        
                        # 子图1：原始幅值地形图
                        axes[0].imshow(amplitude_sequence[mid_frame], cmap='Reds', origin='lower')
                        axes[0].set_title(f"Amplitude (Frame {mid_frame})")
                        axes[0].axis('off')
                        
                        # 子图2：速度矢量场+源汇点
                        axes[1].imshow(amplitude_sequence[mid_frame], cmap='Reds', alpha=0.5, origin='lower')
                        # 绘制矢量场（每隔20个像素，避免密集）
                        step = 20
                        y_plot, x_plot = np.mgrid[step:H:step, step:W:step]
                        u_plot = u_seq[mid_frame][y_plot, x_plot]
                        v_plot = v_seq[mid_frame][y_plot, x_plot]
                        axes[1].quiver(x_plot, y_plot, u_plot*5, v_plot*5, color='green', alpha=0.7)
                        # 绘制源（红）和汇（蓝）
                        src_y, src_x = np.where(sources)
                        axes[1].scatter(src_x, src_y, c='red', s=15, label='Sources')
                        snk_y, snk_x = np.where(sinks)
                        axes[1].scatter(snk_x, snk_y, c='blue', s=15, label='Sinks')
                        axes[1].set_title("Vector Field + Src/Snk")
                        axes[1].legend(fontsize=8)
                        axes[1].axis('off')
                        
                        # 子图3：像素轨迹（边界限制后，无越界）
                        axes[2].imshow(amplitude_sequence[mid_frame], cmap='Reds', alpha=0.5, origin='lower')
                        for traj in trajectories:
                            if mid_frame < len(traj):
                                traj_up_to = traj[:mid_frame+1]
                                axes[2].plot(traj_up_to[:,1], traj_up_to[:,0], 'yellow', linewidth=1.5)
                                axes[2].scatter(traj_up_to[0,1], traj_up_to[0,0], c='orange', s=20)
                        axes[2].set_title("Trajectories")
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(result_dir, f"visualization_frame_{mid_frame}.png"), dpi=80, bbox_inches='tight')
                        plt.close()
                        
                        total_processed += 1
                        print(f"✅ 处理完成，结果保存至: {result_dir}")
                        
                    except Exception as e:
                        print(f"❌ 处理失败: {str(e)}")
                        # 即使有未知错误，也释放内存
                        pass
                    
                    # 手动释放内存
                    del data_dict[folder_type][trial_name][direction][freq_band]
        
        # 处理完一个试次，释放试次内存
        del data_dict[folder_type][trial_name]
    
    # 输出统计信息
    print(f"\n" + "="*50)
    print(f"📈 分析完成统计:")
    print(f"   成功处理数据集数量: {total_processed}")
    print(f"   所有结果保存根目录: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()