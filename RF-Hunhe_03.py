import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # 用于保存CSV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 全局配置（核心：多盘+多受试者选择）----------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

# ====================== 可自定义配置区（已整合G+M盘）======================
# 1. 多个数据根路径（G盘 + M盘）
data_roots = [
    r"M:\InnerSpeech2021-32-OpticalFlow_Final_Fixed_Results",
    r"G:\InnerSpeech2021-32-OpticalFlow_Final_Fixed_Results"  # 新增M盘路径
    # 如需添加更多盘，直接在下面加即可，例如：
    # r"D:\visSpeech2021-32-OpticalFlow_Final_Fixed_Results"
]

# 2. 选择要分析的受试者组合（可跨盘混合选）
# selected_subjects = ['Subject_1','Subject_2','Subject_3','Subject_4','Subject_5','Subject_6','Subject_7','Subject_8','Subject_9','Subject_10']  # 按需修改，比如加Subject_3、Subject_4等
selected_subjects = ['Subject_1']
# 3. 固定配置（无需修改）
directions = ["Up", "Down", "Left", "Right"]  # 4类标签
freq_bands = ["Beta_(13-30_Hz)", "Gamma_(30-100_Hz)", "Alpha_(8-13_Hz)"]  # 多频带融合
# ===========================================================================

# 结果文件命名（包含所选受试者信息，避免覆盖）
subject_suffix = "_".join([s.replace("Subject_", "S") for s in selected_subjects])
result_fig = f"vis_Direction_Result_{subject_suffix}.png"
feature_mat = f"vis_Direction_Features_{subject_suffix}.mat"
final_model = f"vis_Direction_RF_Model_{subject_suffix}.pkl"
final_scaler = f"vis_Direction_Scaler_{subject_suffix}.pkl"
# CSV文件命名
metrics_csv = f"vis_Direction_Metrics_{subject_suffix}.csv"  # 分类指标
cm_num_csv = f"vis_Direction_CM_Numeric_{subject_suffix}.csv"  # 混淆矩阵数值版
cm_percent_csv = f"vis_Direction_CM_Percent_{subject_suffix}.csv"  # 混淆矩阵百分比版
summary_csv = f"vis_Direction_Summary_{subject_suffix}.csv"  # 实验汇总
# 结果保存路径（统一保存到第一个盘的根目录，方便查看）
result_save_dir = data_roots[0]

# ---------------------- 工具函数 ----------------------
def numpy_mode(arr):
    """numpy原生众数计算，无版本问题"""
    if arr.size == 0:
        return 0.0
    unique_vals, counts = np.unique(arr, return_counts=True)
    return float(unique_vals[np.argmax(counts)])

def balance_samples(X, y):
    """小样本平衡：上采样至各类样本数一致"""
    X_balanced, y_balanced = [], []
    # 适配动态类别数
    class_labels = np.unique(y)
    if len(class_labels) == 0:
        return np.array([]), np.array([])
    max_samples = max([np.sum(y==i) for i in class_labels])
    for i in class_labels:
        X_i, y_i = X[y==i], y[y==i]
        if len(X_i) == 0:
            continue
        X_i_res, y_i_res = resample(X_i, y_i, n_samples=max_samples, random_state=23)
        X_balanced.append(X_i_res)
        y_balanced.append(y_i_res)
    if not X_balanced:
        return np.array([]), np.array([])
    return np.vstack(X_balanced), np.hstack(y_balanced)

def print_standard_metrics(report, test_acc, y_test):
    """标准化输出：测试集分类性能指标（每类+整体）"""
    print("="*80)
    print(f"测试集分类性能指标（受试者组合：{', '.join(selected_subjects)}）")
    print("="*80)
    # 打印表头
    header = f"{'方向':<8} {'精确率(Precision)':<18} {'召回率(Recall/每类准确率)':<22} {'F1值(F1-Score)':<15} {'测试集支持样本数':<10}"
    print(header)
    print("-"*80)
    # 打印每类指标
    total_support = 0
    for dir_name in directions:
        metrics = report[dir_name]
        pre = f"{metrics['precision']*100:.2f}%"
        rec = f"{metrics['recall']*100:.2f}%"
        f1 = f"{metrics['f1-score']:.3f}"
        sup = f"{int(metrics['support'])}"
        total_support += int(metrics['support'])
        line = f"{dir_name:<8} {pre:<18} {rec:<22} {f1:<15} {sup:<10}"
        print(line)
    # 打印整体/宏平均/加权平均
    print("-"*80)
    print(f"{'整体':<8} {'-':<18} {f'总体准确率：{test_acc*100:.2f}%':<22} {'-':<15} {total_support:<10}")
    macro_pre = f"{report['macro avg']['precision']*100:.2f}%"
    macro_rec = f"{report['macro avg']['recall']*100:.2f}%"
    macro_f1 = f"{report['macro avg']['f1-score']:.3f}"
    print(f"{'宏平均':<8} {macro_pre:<18} {macro_rec:<22} {macro_f1:<15} {'-':<10}")
    wei_pre = f"{report['weighted avg']['precision']*100:.2f}%"
    wei_rec = f"{report['weighted avg']['recall']*100:.2f}%"
    wei_f1 = f"{report['weighted avg']['f1-score']:.3f}"
    print(f"{'加权平均':<8} {wei_pre:<18} {wei_rec:<22} {wei_f1:<15} {'-':<10}")
    print("="*80)

def save_metrics_to_csv(report, test_acc, cv_acc, target_names, save_path):
    """将分类指标（每类+整体）保存到CSV文件"""
    data = []
    # 每类指标
    for name in target_names:
        metrics = report[name] if name in report else {'precision':0, 'recall':0, 'f1-score':0, 'support':0}
        data.append({
            '类别': name,
            '精确率(%)': round(metrics['precision']*100, 2),
            '召回率(%)': round(metrics['recall']*100, 2),
            'F1值': round(metrics['f1-score'], 3),
            '测试集支持样本数': int(metrics['support']),
            '类型': '单类'
        })
    # 宏平均
    data.append({
        '类别': '宏平均',
        '精确率(%)': round(report['macro avg']['precision']*100, 2),
        '召回率(%)': round(report['macro avg']['recall']*100, 2),
        'F1值': round(report['macro avg']['f1-score'], 3),
        '测试集支持样本数': '-',
        '类型': '汇总'
    })
    # 加权平均
    data.append({
        '类别': '加权平均',
        '精确率(%)': round(report['weighted avg']['precision']*100, 2),
        '召回率(%)': round(report['weighted avg']['recall']*100, 2),
        'F1值': round(report['weighted avg']['f1-score'], 3),
        '测试集支持样本数': '-',
        '类型': '汇总'
    })
    # 整体
    data.append({
        '类别': '整体',
        '精确率(%)': '-',
        '召回率(%)': round(test_acc*100, 2),
        'F1值': '-',
        '测试集支持样本数': sum([d['测试集支持样本数'] for d in data if d['类型']=='单类']),
        '类型': '汇总'
    })
    # 交叉验证准确率
    data.append({
        '类别': '5折交叉验证',
        '精确率(%)': '-',
        '召回率(%)': round(cv_acc*100, 2),
        'F1值': '-',
        '测试集支持样本数': '-',
        '类型': '汇总'
    })
    # 保存为CSV（utf-8-sig支持中文）
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 分类指标已保存至CSV：{save_path}")

def save_cm_to_csv(y_test, y_pred, target_names, num_save_path, percent_save_path):
    """保存混淆矩阵（数值版+百分比版）到CSV"""
    cm = confusion_matrix(y_test, y_pred)
    row_sums = cm.sum(axis=1)
    total_sum = cm.sum()
    col_sums = cm.sum(axis=0)
    
    # 数值版混淆矩阵
    df_cm_num = pd.DataFrame(cm, index=target_names, columns=target_names)
    df_cm_num['行和(真实样本数)'] = row_sums
    # 构建列和行
    col_sum_row = list(col_sums) + [total_sum]
    df_cm_num.loc['列和(预测样本数)'] = col_sum_row
    # 保存
    df_cm_num.to_csv(num_save_path, encoding='utf-8-sig')
    
    # 百分比版混淆矩阵
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_percent = (cm / row_sums_safe.reshape(-1,1) * 100).round(1)
    df_cm_percent = pd.DataFrame(cm_percent, index=target_names, columns=target_names)
    df_cm_percent = df_cm_percent.astype(str) + '%'
    df_cm_percent['行和(真实样本数)'] = '100.0%'
    # 构建列和行（用'-'填充）
    col_sum_percent_row = ['-']*len(target_names) + ['-']
    df_cm_percent.loc['列和(预测样本数)'] = col_sum_percent_row
    # 保存
    df_cm_percent.to_csv(percent_save_path, encoding='utf-8-sig')
    
    print(f"✅ 混淆矩阵数值版已保存至：{num_save_path}")
    print(f"✅ 混淆矩阵百分比版已保存至：{percent_save_path}")

def save_summary_to_csv(cv_acc, test_acc, best_params, target_names, save_path):
    """保存实验汇总信息（准确率、参数、配置）"""
    summary_data = {
        '实验配置': [
            '所选受试者', '分类类别数', '特征维度', 
            '5折交叉验证最优准确率(%)', '测试集准确率(%)',
            '随机森林_n_estimators', '随机森林_max_depth', '随机森林_min_samples_split'
        ],
        '数值/内容': [
            ','.join(selected_subjects),
            len(target_names),
            f'{len(freq_bands)}频带×18维/频带={len(freq_bands)*18}维',
            round(cv_acc*100, 2),
            round(test_acc*100, 2),
            best_params.get('rf__n_estimators', '-'),
            best_params.get('rf__max_depth', '-'),
            best_params.get('rf__min_samples_split', '-')
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 实验汇总信息已保存至：{save_path}")

def print_standard_cm(y_test, y_pred):
    """标准化输出：混淆矩阵（数值版+百分比版）"""
    cm = confusion_matrix(y_test, y_pred)
    # 输出数值版混淆矩阵
    print("\n" + "="*70)
    print(f"标准化混淆矩阵（数值版）| 受试者：{', '.join(selected_subjects)} | 行=真实标签，列=预测标签")
    print("="*70)
    # 表头
    col_header = "真实\\预测 | " + " | ".join([f"{d:<6}" for d in directions]) + " | 行和(真实样本数)"
    print(col_header)
    print("-"*70)
    # 每行数据
    row_sums = cm.sum(axis=1)
    total_sum = cm.sum()
    for i, dir_name in enumerate(directions):
        row_vals = [f"{cm[i,j]:<6}" for j in range(4)]
        row_line = f"{dir_name:<8} | " + " | ".join(row_vals) + f" | {row_sums[i]}"
        print(row_line)
    # 列和
    col_sums = cm.sum(axis=0)
    col_vals = [f"{col_sums[j]:<6}" for j in range(4)]
    col_line = f"列和(预测数) | " + " | ".join(col_vals) + f" | {total_sum}"
    print("-"*70)
    print(col_line)
    print("="*70)

    # 输出百分比版混淆矩阵（按真实样本数归一化，每行100%）
    print("\n" + "="*70)
    print(f"标准化混淆矩阵（百分比版）| 受试者：{', '.join(selected_subjects)} | 行=真实标签，列=预测标签（每行归一化100%）")
    print("="*70)
    print(col_header.replace("数值版", "百分比版"))
    print("-"*70)
    # 避免除以0
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_percent = cm / row_sums_safe.reshape(-1,1) * 100
    for i, dir_name in enumerate(directions):
        row_vals = [f"{cm_percent[i,j]:.1f}%".ljust(6) for j in range(4)]
        row_line = f"{dir_name:<8} | " + " | ".join(row_vals) + f" | 100.0%"
        print(row_line)
    print("="*70)

# ---------------------- 18维/频带 高维特征提取 ----------------------
def extract_flow_features(mat_path):
    """提取单频带18维特征"""
    try:
        data = scipy.io.loadmat(mat_path, simplify_cells=True)
    except:
        return np.zeros(18)
    # 数据兜底，避免键缺失/维度错误
    u_seq = data.get("u_sequence", np.zeros((1,64,64)))
    v_seq = data.get("v_sequence", np.zeros((1,64,64)))
    trajectories = data.get("trajectories", [])
    sources = data.get("sources", np.zeros((64,64)))
    sinks = data.get("sinks", np.zeros((64,64)))
    H, W = sources.shape if sources.ndim == 2 else (64, 64)

    # 光流基础计算+异常值处理
    speed_seq = np.sqrt(np.square(u_seq) + np.square(v_seq))
    speed_seq = np.nan_to_num(speed_seq, nan=0, posinf=0, neginf=0)
    dir_angles = np.arctan2(v_seq, u_seq) * 180 / np.pi
    dir_angles = np.nan_to_num(dir_angles, nan=0, posinf=0, neginf=0)
    speed_flat, angle_flat = speed_seq.flatten(), dir_angles.flatten()

    # 1. 速度统计特征（4维）：平均/峰值/标准差/中位数
    avg_speed, peak_speed = np.mean(speed_flat), np.max(speed_flat)
    std_speed, med_speed = np.std(speed_flat), np.median(speed_flat)
    # 2. 方向统计特征（4维）：众数/标准差/绝对值均值/主方向占比
    dir_mode = numpy_mode(angle_flat)
    std_dir, avg_abs_dir = np.std(angle_flat), np.mean(np.abs(angle_flat))
    main_dir_ratio = np.sum(np.abs(angle_flat - dir_mode) < 30) / len(angle_flat) if len(angle_flat) > 0 else 0.0
    # 3. 轨迹特征（3维）：平均长度/最大长度/平均偏移量
    traj_lengths, traj_offsets = [], []
    if isinstance(trajectories, list) and len(trajectories) > 0:
        for traj in trajectories:
            traj = np.nan_to_num(traj, nan=0, posinf=0, neginf=0)
            if traj.shape[0] >= 2:
                dist = np.sum(np.sqrt(np.square(np.diff(traj[:,0])) + np.square(np.diff(traj[:,1]))))
                traj_lengths.append(dist)
                offset = np.sqrt(np.square(traj[-1,0]-traj[0,0]) + np.square(traj[-1,1]-traj[0,1]))
                traj_offsets.append(offset)
    avg_traj_len = np.mean(traj_lengths) if traj_lengths else 0.0
    max_traj_len = np.max(traj_lengths) if traj_lengths else 0.0
    avg_traj_offset = np.mean(traj_offsets) if traj_offsets else 0.0
    # 4. 源汇特征（4维）：源点中心x/y/占比 + 汇点占比
    src_coords = np.where(sources > 0)
    src_ratio = len(src_coords[0])/(H*W) if (H*W)>0 else 0.0
    src_x = np.mean(src_coords[1])/W if len(src_coords[0])>0 else 0.5
    src_y = np.mean(src_coords[0])/H if len(src_coords[0])>0 else 0.5
    snk_coords = np.where(sinks > 0)
    snk_ratio = len(snk_coords[0])/(H*W) if (H*W)>0 else 0.0
    # 5. 分布特征（3维）：速度时空平均/方向时空平均/光流活性
    speed_spatial_avg = np.mean(speed_seq, axis=(1,2)).mean()
    dir_spatial_avg = np.mean(dir_angles, axis=(1,2)).mean()
    flow_act = np.sum(speed_flat > 0.01)/len(speed_flat) if len(speed_flat)>0 else 0.0

    # 拼接18维特征并返回
    features = np.array([
        avg_speed, peak_speed, std_speed, med_speed,
        dir_mode, std_dir, avg_abs_dir, main_dir_ratio,
        avg_traj_len, max_traj_len, avg_traj_offset,
        src_x, src_y, src_ratio, snk_ratio,
        speed_spatial_avg, dir_spatial_avg, flow_act
    ])
    return np.nan_to_num(features, nan=0, posinf=0, neginf=0)

# ---------------------- 多盘 + 多受试者特征融合 ----------------------
def load_all_features():
    """加载 多个盘 + 所选受试者 的所有特征"""
    feature_matrix, labels, trial_list, subject_records = [], [], [], []
    print(f"\n📌 开始从多盘提取特征 | 所选受试者：{', '.join(selected_subjects)}")

    # 遍历每个盘的根目录
    for root_parent_dir in data_roots:
        print(f"\n🗂️  正在扫描路径：{root_parent_dir}")

        # 遍历选中的每个受试者
        for subject in selected_subjects:
            # 构建当前受试者路径
            subject_path = os.path.join(root_parent_dir, subject, "OpticalFlow_Final_Fixed_Results", "vis")
            if not os.path.exists(subject_path):
                print(f"  ⚠️  未找到受试者 {subject} 在路径 {root_parent_dir}，跳过")
                continue  # 这个受试者不在这个盘，跳过

            print(f"  ✅ 找到受试者：{subject}（路径：{subject_path}）")

            # 遍历方向
            for dir_idx, direction in enumerate(directions):
                print(f"    正在提取 {subject} - {direction} 方向的特征...")
                for trial_name in os.listdir(subject_path):
                    trial_path = os.path.join(subject_path, trial_name)
                    if not os.path.isdir(trial_path) or not trial_name.startswith("Result_"):
                        continue
                    dir_path = os.path.join(trial_path, direction)
                    if not os.path.exists(dir_path):
                        continue

                    # 提取多频带特征
                    freq_feats = []
                    for freq in freq_bands:
                        mat_path = os.path.join(dir_path, freq, "flow_results.mat")
                        freq_feats.append(extract_flow_features(mat_path) if os.path.exists(mat_path) else np.zeros(18))
                    total_feat = np.hstack(freq_feats)

                    if np.sum(total_feat) > 1e-5:
                        feature_matrix.append(total_feat)
                        labels.append(dir_idx)
                        trial_list.append(f"{subject}_{trial_name}")
                        subject_records.append(subject)

    # 转数组
    X = np.array(feature_matrix, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    trial_arr = np.array(trial_list)
    subject_arr = np.array(subject_records)

    if len(X) == 0:
        print("❌ 未提取到任何有效样本！请检查路径或受试者名称是否正确")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    X_bal, y_bal = balance_samples(X, y)
    print(f"\n✅ 多盘融合完成！")
    print(f"   原始样本：{len(X)} → 平衡后：{len(X_bal)}")
    print(f"   特征维度：{X_bal.shape[1]} 维")
    return X_bal, y_bal, trial_arr, X, y, subject_arr

# ---------------------- 随机森林模型训练+网格搜索调参 ----------------------
def train_model(X, y):
    """RF模型训练，自动网格搜索最优参数"""
    # 分层划分训练集/测试集（7:3）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=23, stratify=y
    )
    # 构建Pipeline（标准化+RF）
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=23, n_jobs=-1))  # 多核加速
    ])
    # 网格搜索参数
    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [5, 10, None],
        "rf__min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"\n🔍 网格搜索随机森林最优参数（5折交叉验证）...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    cv_best_acc = grid_search.best_score_
    print(f"✅ 最优参数：{grid_search.best_params_}")
    print(f"✅ 交叉验证最优准确率：{cv_best_acc*100:.2f}%")

    # 测试集预测
    y_pred = best_model.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=directions, digits=3, output_dict=True)
    # 标准化输出指标和混淆矩阵
    print_standard_metrics(report, test_acc, y_test)
    print_standard_cm(y_test, y_pred)

    # 保存模型和标准化器
    model_path = os.path.join(result_save_dir, final_model)
    scaler_path = os.path.join(result_save_dir, final_scaler)
    joblib.dump(best_model.named_steps["rf"], model_path)
    joblib.dump(best_model.named_steps["scaler"], scaler_path)
    print(f"\n✅ 模型保存路径：{model_path}")
    print(f"✅ 标准化器保存路径：{scaler_path}")

    # 保存所有CSV结果
    # 1. 分类指标
    metrics_path = os.path.join(result_save_dir, metrics_csv)
    save_metrics_to_csv(report, test_acc, cv_best_acc, directions, metrics_path)
    # 2. 混淆矩阵
    cm_num_path = os.path.join(result_save_dir, cm_num_csv)
    cm_percent_path = os.path.join(result_save_dir, cm_percent_csv)
    save_cm_to_csv(y_test, y_pred, directions, cm_num_path, cm_percent_path)
    # 3. 实验汇总
    summary_path = os.path.join(result_save_dir, summary_csv)
    save_summary_to_csv(cv_best_acc, test_acc, grid_search.best_params_, directions, summary_path)

    return best_model, X_test, y_test, y_pred, report

# ---------------------- 结果可视化 ----------------------
def visualize(best_model, X, y, X_test, y_test, y_pred, report):
    """4子图可视化：特征热力图+混淆矩阵+每类召回率+精确率/F1对比"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    test_acc = np.mean(y_pred == y_test)

    # 子图1：前12维特征均值热力图
    X_vis = X[:, :12]
    feat_names = [f"特征{i+1}" for i in range(12)]
    dir_means = np.array([X_vis[y==i].mean(axis=0) for i in range(4)])
    sns.heatmap(dir_means, annot=True, fmt=".3f", cmap="RdBu_r",
                xticklabels=feat_names, yticklabels=directions, ax=ax1)
    ax1.set_title(f"vis各方向特征均值热力图（受试者：{subject_suffix}）", 
                  fontsize=14, fontweight="bold", pad=20)
    ax1.set_xlabel("特征维度", fontsize=12)
    ax1.set_ylabel("运动方向", fontsize=12)

    # 子图2：混淆矩阵（标注数值）
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                xticklabels=directions, yticklabels=directions, ax=ax2)
    ax2.set_title(f"混淆矩阵（测试集总体准确率：{test_acc*100:.2f}%）",
                  fontsize=14, fontweight="bold", pad=20)
    ax2.set_xlabel("预测方向", fontsize=12)
    ax2.set_ylabel("真实方向", fontsize=12)

    # 子图3：每类召回率（每类准确率）柱状图
    recalls = [report[dir]["recall"]*100 for dir in directions]
    sns.barplot(x=directions, y=recalls, palette="viridis", ax=ax3)
    ax3.set_title("每类方向召回率（每类准确率）", fontsize=14, fontweight="bold", pad=20)
    ax3.set_ylabel("召回率（%）", fontsize=12)
    ax3.set_ylim(0, 100)
    # 柱子标注数值
    for i, v in enumerate(recalls):
        ax3.text(i, v+2, f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 子图4：每类精确率+F1值双柱状图
    precisions = [report[dir]["precision"]*100 for dir in directions]
    f1s = [report[dir]["f1-score"]*100 for dir in directions]
    x = np.arange(4)
    width = 0.35
    ax4.bar(x-width/2, precisions, width, label="精确率", color="skyblue", edgecolor="black")
    ax4.bar(x+width/2, f1s, width, label="F1值", color="lightcoral", edgecolor="black")
    ax4.set_title("每类方向精确率 & F1值对比", fontsize=14, fontweight="bold", pad=20)
    ax4.set_ylabel("百分比（%）", fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(directions)
    ax4.legend(fontsize=11)
    ax4.set_ylim(0, 100)
    # 标注数值
    for i, (p, f) in enumerate(zip(precisions, f1s)):
        ax4.text(i-width/2, p+2, f"{p:.1f}%", ha="center", va="bottom", fontsize=10)
        ax4.text(i+width/2, f+2, f"{f:.1f}%", ha="center", va="bottom", fontsize=10)

    # 总标题
    fig.suptitle(f"vis方向分类结果可视化（受试者：{subject_suffix}）", 
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(result_save_dir, result_fig)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 可视化结果保存路径：{fig_path}")

# ---------------------- 主函数：全流程执行 ----------------------
if __name__ == "__main__":
    try:
        # 1. 提取多频带融合特征+样本平衡
        X_balanced, y_balanced, trial_ids, X_original, y_original, subject_arr = load_all_features()
        if len(X_balanced) == 0:
            print("❌ 程序终止：未提取到有效样本")
            exit()
        # 2. 保存高维特征矩阵
        feature_path = os.path.join(result_save_dir, feature_mat)
        scipy.io.savemat(feature_path, {
            "X_balanced": X_balanced, "y_balanced": y_balanced,
            "X_original": X_original, "y_original": y_original,
            "trial_ids": trial_ids, "directions": directions,
            "freq_bands": freq_bands, "selected_subjects": selected_subjects,
            "subject_records": subject_arr
        })
        print(f"\n✅ 高维特征矩阵保存路径：{feature_path}")
        # 3. 训练模型+输出标准化结果+保存CSV
        best_model, X_test, y_test, y_pred, report = train_model(X_balanced, y_balanced)
        # 4. 结果可视化
        visualize(best_model, X_balanced, y_balanced, X_test, y_test, y_pred, report)
        # 5. 最终提示
        print("\n" + "="*90)
        print(f"✅ 全流程执行完成！（受试者组合：{', '.join(selected_subjects)}）")
        print(f"✅ 所有结果已保存至：{result_save_dir}")
        print(f"✅ 生成的文件列表：")
        print(f"   - 可视化图：{result_fig}")
        print(f"   - 特征矩阵：{feature_mat}")
        print(f"   - 分类模型：{final_model}")
        print(f"   - 标准化器：{final_scaler}")
        print(f"   - 分类指标CSV：{metrics_csv}")
        print(f"   - 混淆矩阵数值版CSV：{cm_num_csv}")
        print(f"   - 混淆矩阵百分比版CSV：{cm_percent_csv}")
        print(f"   - 实验汇总CSV：{summary_csv}")
        print("="*90)
    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()