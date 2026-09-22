"""Plot five attack methods on the completed, matched CataPro test cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_no_esm_lm"
RUNS = [HERE / "catapro_test_2697_residual_a3_distinct",
        HERE / "catapro_test_2697_residual_remaining_distinct"]
METHODS = ["A0_pamp", "A1_hotflip", "A2_fw_avg", "A3_fw_avg_pamp", "B2_random"]
LABELS = ["A0\nPAMP", "A1\nHotFlip", "A2\nFW-avg", "A3\nFW-avg\n+ PAMP", "B2\nRandom"]
SHORT_LABELS = ["SPM", "HF", "FWA", "FWA-P", "RND"]
DISPLAY_NAMES = ["Single-point mutation", "HotFlip", "Three-point gradient average",
                 "Three-point gradient average\n+ distance penalty", "Random mutation"]
COLORS = ["#DFA869", "#7DACC4", "#96BBA5", "#397B69", "#B2B5BE"]
ENDPOINTS = ["round1_top1", "round1_best_of_top5", "round2_top1"]
TITLES = ["Round 1: Top1", "Round 1: best of Top5", "Round 2: Top1"]


def main(central_only=False):
    OUT.mkdir(exist_ok=True)
    for run in RUNS:
        assert json.loads((run / "status.json").read_text())["stage"] == "complete"
        assert json.loads((run / "verification.json").read_text())["status"] == "PASS"
    cohorts = [pd.read_csv(run / "cohort.csv") for run in RUNS]
    pd.testing.assert_frame_equal(cohorts[0], cohorts[1])
    paths = [run / "endpoint_results.csv" for run in RUNS]
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    frame = frame[frame.method.isin(METHODS)].copy()
    assert frame.source.eq("residual").all() and frame.evaluator.eq("residual").all()
    assert not frame.duplicated(["row_id", "method", "endpoint"]).any()
    assert set(frame.endpoint) == set(ENDPOINTS)
    assert np.isfinite(frame.delta_log2).all()
    assert len(frame) == 2697 * 5 * 3
    for _, group in frame.groupby(["method", "endpoint"]):
        assert set(group.row_id) == set(cohorts[0].row_id)
    assert frame.loc[frame.endpoint == "round2_top1", "net_substitutions"].eq(2).all()
    columns = ["row_id", "method", "endpoint", "wt_log2", "mutant_log2", "delta_log2", "edits", "net_substitutions"]
    frame[columns].to_csv(OUT / "plot_data.csv", index=False, float_format="%.17g")

    summary = []
    values = {}
    for endpoint in ENDPOINTS:
        values[endpoint] = []
        for method in METHODS:
            x = frame.loc[(frame.endpoint == endpoint) & (frame.method == method), "delta_log2"].to_numpy()
            values[endpoint].append(x)
            q1, median, q3 = np.quantile(x, [.25, .5, .75])
            iqr = q3 - q1
            inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
            summary.append(dict(method=method, endpoint=endpoint, N=len(x), mean=x.mean(),
                median=median, q1=q1, q3=q3, whisker_low=inside.min(), whisker_high=inside.max(),
                min=x.min(), max=x.max(), positive_fraction=(x > 1e-6).mean()))
    stats = pd.DataFrame(summary)
    stats.to_csv(OUT / "boxplot_statistics.csv", index=False, float_format="%.17g")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#24343D", "text.color": "#24343D",
        "xtick.color": "#24343D", "ytick.color": "#52616B",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    if central_only:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), squeeze=False)
        fig.subplots_adjust(left=.055, right=.785, top=.875, bottom=.22, wspace=.15)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2),
            gridspec_kw={"height_ratios": [1, 1.55]}, sharex="col")
        fig.subplots_adjust(left=.075, right=.98, top=.845, bottom=.205, hspace=.28, wspace=.15)
    for plot_row, row in enumerate([1] if central_only else range(2)):
        for col, endpoint in enumerate(ENDPOINTS):
            ax = axes[plot_row, col]
            boxes = ax.boxplot(values[endpoint], positions=np.arange(1, 6), widths=.54,
                patch_artist=True, whis=1.5, showfliers=True,
                medianprops={"color": "#172A34", "linewidth": 1.8},
                whiskerprops={"color": "#62717B", "linewidth": 1},
                capprops={"color": "#62717B", "linewidth": 1},
                flierprops={"marker": "o", "markersize": 2.1,
                    "markeredgewidth": 0, "alpha": .22})
            for i, (patch, flier) in enumerate(zip(boxes["boxes"], boxes["fliers"])):
                patch.set_facecolor(COLORS[i])
                patch.set_edgecolor("#465861")
                patch.set_linewidth(1)
                flier.set_markerfacecolor(COLORS[i] if i != 3 else "#286957")
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="#E6EBEF", linewidth=.75)
            ax.axhline(0, color="#596B78", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.set_xlim(.45, 5.55)
            ax.tick_params(axis="both", length=3)
            ax.spines["left"].set_color("#BAC4CB")
            ax.spines["bottom"].set_color("#BAC4CB")
            if row == 0:
                ax.set_ylim(-8, 12)
                ax.set_yticks([-8, -4, 0, 4, 8, 12])
                ax.set_title(TITLES[col], fontsize=13, fontweight="bold", pad=14)
                ax.tick_params(axis="x", bottom=False)
            else:
                if central_only:
                    ax.set_title(TITLES[col], fontsize=13, fontweight="bold", pad=16)
                ax.set_ylim(-.55, .95)
                ax.yaxis.set_major_locator(MultipleLocator(.25))
                ax.set_xticks(range(1, 6), SHORT_LABELS if central_only else LABELS, fontsize=10)
                ax.tick_params(axis="x", pad=8)
                for i, x in enumerate(values[endpoint], 1):
                    ax.text(i, .90, f"{np.median(x):+.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold" if i == 4 else "normal",
                        color="#286957" if i == 4 else "#52616B",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.5})
            if col:
                ax.tick_params(axis="y", labelleft=False)
    if central_only:
        axes[0, 0].set_ylabel(r"Predicted $\Delta\log_2(k_{cat})$", labelpad=10)
        legend = fig.add_axes([.815, .22, .175, .655])
        legend.set_axis_off()
        legend.text(0, 1, "Methods", fontsize=13, va="top")
        for i, (short, method, name, color) in enumerate(zip(SHORT_LABELS, METHODS, DISPLAY_NAMES, COLORS)):
            y = .825 - .177 * i
            legend.add_patch(Rectangle((0, y-.012), .066, .045, facecolor=color,
                edgecolor="#465861", linewidth=.7, transform=legend.transAxes))
            legend.text(.10, y+.012, f"{short}  ({method.split('_')[0]})", va="center",
                fontsize=11, fontweight="bold" if i == 3 else "normal")
            legend.text(.10, y-.040, name, fontsize=10.5, va="top", linespacing=1.25)
        fig.text(.055, .072, "N = 2,697 per method. All data used; y-axis zoomed. Numbers: medians. Boxes: Q1–Q3; whiskers: 1.5 × IQR.", fontsize=10, color="#52616B")
        fig.text(.055, .032, "Model prediction changes relative to the original sequence. Observations outside the displayed range are not shown.", fontsize=10, color="#52616B")
    else:
        axes[0, 0].set_ylabel("Full range\n" + r"$\Delta\log_2(k_{cat})$", labelpad=10)
        axes[1, 0].set_ylabel("Central view\n" + r"$\Delta\log_2(k_{cat})$", labelpad=10)
        fig.text(.075, .965, "Attack-induced prediction changes", fontsize=23, fontweight="bold")
        fig.text(.075, .925, "Residual Predictor  |  2,697 matched enzyme–substrate records per method  |  ESM-LM excluded",
                 fontsize=12, color="#52616B")
        fig.text(.075, .105, "Boxes: 25th–75th percentiles; line: median; whiskers: 1.5 × IQR; dots: observations beyond whiskers.", fontsize=10)
        fig.text(.075, .078, "Both rows use all observations. Lower panels zoom the y-axis; displayed numbers are medians. No values were removed.", fontsize=10)
        fig.text(.075, .051, "Top5 selects the highest predicted candidate. Round 2 continues from fixed Top1 and changes a second, distinct site.", fontsize=10)
        fig.text(.075, .024, "Changes are model predictions relative to the original sequence, not experimentally measured activity changes.", fontsize=10, color="#52616B")
    basename = "attack_boxplots_no_esm_lm" + ("_central" if central_only else "")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{basename}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    manifest = {"cohort_size": 2697, "methods": METHODS, "endpoints": ENDPOINTS,
        "trimmed": False, "whiskers": "1.5 IQR", "central_view_ylim": [-.55, .95],
        "display_names": {m: {"abbreviation": s, "name": n} for m, s, n in zip(METHODS, SHORT_LABELS, DISPLAY_NAMES)},
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "README.md").write_text("# 五种攻击方法箱形图\n\n"
        "排除 B0 ESM-LM，展示 A0、A1、A2、A3 和 B2。全部 2697 条测试记录，未截尾。\n\n"
        "三列分别为首轮 Top1、首轮 Top5 最优、两轮 Top1。纵轴为相对原序列的预测 Δlog₂(kcat)。"
        "上排显示全范围，下排仅放大中央区域，二者使用相同完整数据。下排顶部数字为中位数。"
        "箱体为 Q1–Q3，中线为中位数，须为 1.5 IQR 范围内的最远观测，点为须外观测。\n\n"
        "Top5 最优由预测器评价后选择；两轮 Top1 从固定首轮排名第一继续，第二轮为另一位点。"
        "结果为模型预测变化，不等同于实测活性；同一酶不同底物保留为不同记录。\n\n"
        "PNG 为预览，PDF/SVG 为矢量图，plot_data.csv 为完整绘图数据，boxplot_statistics.csv 为统计值。\n\n"
        "文件名含 _central 的版本仅保留三个中央放大面板，统计仍使用全部数据。运行绘图脚本时加 --central-only 即可生成。\n\n"
        "中央面板横轴简写：SPM=单点位突变（原 A0）；HF=HotFlip（A1）；FWA=三点梯度平均（A2）；"
        "FWA-P=三点梯度平均＋距离惩罚（A3）；RND=随机突变（B2）。绘图名称调整不改变底层方法标识。\n\n"
        "右侧说明已改为英文，全部图形由 Python/Matplotlib 生成。\n")
    print(stats.to_string(index=False))
    print("Saved figures to", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--central-only", action="store_true")
    main(parser.parse_args().central_only)
