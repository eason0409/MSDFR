import numpy as np
from scipy.stats import wilcoxon

# ========================================
# 从图片中提取的 AUC 数据
# ========================================

datasets = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 
            'A1', 'A2', 'C1', 'C2', 'C3']

# MSDFR 的 AUC 值
msdfr = np.array([0.95, 0.97, 0.95, 0.92, 0.93, 0.97, 0.92, 
                  0.96, 0.94, 0.97, 0.93, 1.0, 1.0, 0.98])

# 各基线方法的 AUC 值
baselines = {
    'MiPo':    np.array([0.94, 0.97, 0.95, 0.87, 0.91, 0.99, 0.87, 
                         0.93, 0.88, 0.91, 0.95, 0.98, 0.95, 0.97]),
    'DAC':     np.array([0.85, 0.83, 0.80, 0.76, 0.82, 0.91, 0.88, 
                         0.90, 0.87, 0.81, 0.83, 0.86, 0.85, 0.90]),
    'TPBA':    np.array([0.87, 0.81, 0.80, 0.60, 0.55, 0.88, 0.85, 
                         0.91, 0.88, 0.74, 0.83, 0.88, 0.81, 0.86]),
    'RapidLOF': np.array([0.80, 0.80, 0.70, 0.80, 0.63, 0.86, 0.83, 
                          0.92, 0.83, 0.72, 0.81, 0.84, 0.76, 0.82]),
    'ATDC':    np.array([0.80, 0.91, 0.81, 0.61, 0.57, 0.51, 0.35, 
                         0.86, 0.69, 0.78, 0.81, 0.85, 0.77, 0.90]),
    'ELSTMAE': np.array([0.79, 0.73, 0.76, 0.63, 0.61, 0.74, 0.66, 
                         0.82, 0.84, 0.77, 0.75, 0.82, 0.78, 0.85]),
}

# ========================================
# Wilcoxon 符号秩检验
# ========================================

print("=" * 60)
print("Wilcoxon Signed-Rank Test: MSDFR vs. Each Baseline")
print("=" * 60)
print(f"{'Comparison':<25} {'p-value':>12} {'Significant (α=0.05)':>20}")
print("-" * 60)

results = {}

for name, baseline_auc in baselines.items():
    # 计算差值
    diff = msdfr - baseline_auc
    
    # Wilcoxon 检验
    # alternative='greater' 表示检验 MSDFR > baseline 的单侧假设
    statistic, p_value = wilcoxon(diff, alternative='greater', zero_method='zsplit')
    
    results[name] = {
        'statistic': statistic,
        'p_value': p_value,
        'mean_diff': np.mean(diff),
        'median_diff': np.median(diff),
        'wins': np.sum(diff > 0),
        'ties': np.sum(diff == 0),
        'losses': np.sum(diff < 0),
    }
    
    significant = "Yes" if p_value < 0.05 else "No"
    print(f"MSDFR vs. {name:<15} {p_value:>12.10f} {significant:>20}")
    
    # 打印详细对比（可选）
    print(f"  Mean AUC diff: {results[name]['mean_diff']:+.4f}")
    print(f"  Wins/Ties/Losses: {results[name]['wins']}/{results[name]['ties']}/{results[name]['losses']}")
    print()

print("=" * 60)
print("Summary: MSDFR vs. All Baselines")
print("=" * 60)

# 汇总统计
for name, res in results.items():
    sig_mark = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else "ns"
    print(f"{name}: p={res['p_value']:.6f} {sig_mark}, mean_diff={res['mean_diff']:+.4f}")

# ========================================
# 生成论文表格（LaTeX/Markdown 格式）
# ========================================

print("\n" + "=" * 60)
print("Table for Paper (Markdown Format)")
print("=" * 60)

print("| Comparison | p-value | Significant (α = 0.05) |")
print("|-----------|---------|----------------------|")
for name, res in results.items():
    sig = "Yes" if res['p_value'] < 0.05 else "No"
    print(f"| MSDFR vs. {name} | {res['p_value']:.4f} | {sig} |")

print("\n" + "=" * 60)
print("Table for Paper (LaTeX Format)")
print("=" * 60)

print("\\begin{table}[htbp]")
print("\\caption{Wilcoxon Signed-Rank Test: MSDFR vs. Baselines}")
print("\\begin{tabular}{lcc}")
print("\\hline")
print("Comparison & p-value & Significant ($\\\\alpha$ = 0.05) \\\\")
print("\\hline")
for name, res in results.items():
    sig = "Yes" if res['p_value'] < 0.05 else "No"
    print(f"MSDFR vs. {name} & {res['p_value']:.4f} & {sig} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")
