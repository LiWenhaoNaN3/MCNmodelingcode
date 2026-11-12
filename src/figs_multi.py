# -*- coding: utf-8 -*-
"""多船/多 AUV 协同搜索图件 (一套 5 张图):

  fig_multi              总览: 热力图(P72/期望时间) + 成本-效能 + 代表性曲线
  fig_multi_curves       15 种组合的 P(t) 曲线 (按船数分组)
  fig_multi_marginal     边际收益 / 重复扫描 / Pareto 前沿
  fig_multi_beta         重复扫描惩罚 β 的调优 (成功率 vs 重复率)
  fig_multi_plan         示例案例: 多资产航迹 + 每格被搜索次数热图
"""
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from . import plotmpl as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
C = P.C


def _load():
    d = np.load(os.path.join(CACHE, "multi_results.npz"), allow_pickle=False)
    return dict(curves=d["curves"], ships=d["n_ships"], auvs=d["n_auv"], costs=d["costs"])


def _single_results():
    """单船 7 方案结果 (data/搜索方案对比.csv)。"""
    fn = os.path.join(DATA, "搜索方案对比.csv")
    if not os.path.exists(fn):
        return [], []
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    ids = [r["方案"] for r in rows]
    return [float(r["成本估计(万美元)"]) for r in rows], [100 * float(r["P(72h)"]) for r in rows], ids


# ============================================================================
# 1. 总览 (已有)
# ============================================================================
def fig_multi():
    P.setup()
    d = _load()
    curves, ships, auvs, costs = d["curves"], d["ships"], d["auvs"], d["costs"]
    t = np.arange(curves.shape[1])
    sh, au = np.unique(ships), np.unique(auvs)
    P72 = np.zeros((len(sh), len(au)))
    ET = np.zeros((len(sh), len(au)))
    for i in range(len(curves)):
        a = int(np.where(sh == ships[i])[0][0])
        b = int(np.where(au == auvs[i])[0][0])
        P72[a, b] = 100 * curves[i, 72]
        ET[a, b] = float(np.sum(1.0 - curves[i][:-1]))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.34)
    for k, (mat, title, fmt, cmap) in enumerate((
            (P72, "(a) 72 h 内找到比例 (%)", "%.1f", "YlGnBu"),
            (ET, "(b) 期望发现时间 (h, 截断 96)", "%.1f", "YlOrRd_r"))):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(au))); ax.set_xticklabels(au)
        ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh)
        ax.set_xlabel("AUV 数量"); ax.set_ylabel("船数 (含母船)")
        for i in range(len(sh)):
            for j in range(len(au)):
                ax.text(j, i, fmt % mat[i, j], ha="center", va="center", fontsize=7.5)
        ax.set_title(title, fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(costs, 100 * curves[:, 72], c=ships, cmap="viridis", s=42,
               edgecolor="k", linewidth=0.3)
    for i in range(len(costs)):
        ax.annotate("%d/%d" % (ships[i], auvs[i]), (costs[i], 100 * curves[i, 72]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.2)
    ax.set_xlabel("成本估计 (万美元)"); ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(90, 100.8)
    ax.set_title("(c) 成本-效能 (船数/AUV数)", fontsize=8.5)
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(gs[1, :])
    picks = [(2, 0), (2, 2), (3, 0), (3, 2), (4, 0), (4, 4)]
    palette = [C["gray"], C["sky"], C["green"], C["blue"], C["orange"], C["red"]]
    for (ns, na), col in zip(picks, palette):
        m = (ships == ns) & (auvs == na)
        if not m.any():
            continue
        i = int(np.where(m)[0][0])
        ax.plot(t, 100 * curves[i], "-", lw=1.5, color=col,
                label="船 %d + AUV %d (%.0f 万)" % (ns, na, costs[i]))
    ax.axvline(3, color="0.5", lw=0.8, ls=":")
    ax.text(3.4, 5, "增援到位 3 h", fontsize=7.5, color="0.4")
    ax.axvline(72, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel("已找到比例 (%)")
    ax.set_xlim(0, 96); ax.set_ylim(0, 102)
    ax.legend(fontsize=7.2, ncol=3, framealpha=0.9, loc="lower right")
    ax.set_title("(d) 代表性组合的发现概率曲线 (t=0 仅母船, 3 h 后增援到位)", fontsize=9)
    ax.grid(alpha=0.25)
    return P.save(fig, "fig_multi")


# ============================================================================
# 2. 15 条曲线 (按船数分组)
# ============================================================================
def fig_curves():
    P.setup()
    d = _load()
    curves, ships, auvs, costs = (d["curves"], d["ships"], d["auvs"], d["costs"])
    t = np.arange(curves.shape[1])
    au = np.unique(auvs)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(au)))
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, ns in zip(axs, (2, 3, 4)):
        m0 = ships == ns
        for j, na in enumerate(au):
            m = m0 & (auvs == na)
            if not m.any():
                continue
            i = int(np.where(m)[0][0])
            ax.plot(t, 100 * curves[i], lw=1.4, color=cols[j],
                    label="AUV %d" % na)
            ax.text(97, 100 * curves[i, 72], "%.1f" % (100 * curves[i, 72]),
                    fontsize=6.5, color=cols[j], va="center")
        ax.axvline(3, color="0.6", lw=0.8, ls=":")
        ax.set_xlim(0, 100); ax.set_ylim(40, 101)
        ax.set_xlabel(P.L("t_h"))
        ax.set_title("船数 = %d (含母船)" % ns, fontsize=9)
        ax.grid(alpha=0.25)
        if ns == 2:
            ax.set_ylabel("已找到比例 (%)")
    axs[0].legend(fontsize=6.8, loc="lower right", framealpha=0.9)
    fig.suptitle("15 种组合的发现概率曲线 (右侧数字 = 72 h 成功率; 成本见图 fig_multi_marginal)", fontsize=9.5)
    fig.tight_layout()
    return P.save(fig, "fig_multi_curves")


# ============================================================================
# 3. 边际收益 / 重复扫描 / Pareto
# ============================================================================
def fig_marginal():
    P.setup()
    d = _load()
    curves, ships, auvs, costs = (d["curves"], d["ships"], d["auvs"], d["costs"])
    p72 = 100 * curves[:, 72]
    p96 = 100 * curves[:, 96]
    sh, au = np.unique(ships), np.unique(auvs)

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6))
    # (a) P72 vs AUV 数量 (按船数分组)
    ax = axs[0, 0]
    for k, ns in enumerate(sh):
        m = ships == ns
        ys = [p72[m & (auvs == na)][0] for na in au]
        ax.plot(au, ys, "o-", lw=1.4, color=[C["sky"], C["green"], C["orange"]][k],
                label="船 %d" % ns)
    ax.set_xlabel("AUV 数量"); ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_xticks(au); ax.set_ylim(95, 100.2)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("(a) AUV 的边际作用", fontsize=9)

    # (b) 每 100 万美元带来的 P72 增量
    ax = axs[0, 1]
    single_costs, single_p72, single_ids = _single_results()
    base = 89.6
    base_cost = 182.0
    steps = [
        ("单船M4→2船0AUV", 252 - base_cost, p72[(ships == 2) & (auvs == 0)][0] - base),
        ("2船: +4 AUV", 480 - 252, p72[(ships == 2) & (auvs == 4)][0]
         - p72[(ships == 2) & (auvs == 0)][0]),
        ("2船→3船(0AUV)", 322 - 252, p72[(ships == 3) & (auvs == 0)][0]
         - p72[(ships == 2) & (auvs == 0)][0]),
        ("3船→4船(0AUV)", 392 - 322, p72[(ships == 4) & (auvs == 0)][0]
         - p72[(ships == 3) & (auvs == 0)][0]),
        ("3船2AUV→4船4AUV", 620 - 436, p72[(ships == 4) & (auvs == 4)][0]
         - p72[(ships == 3) & (auvs == 2)][0]),
    ]
    names = [s[0] for s in steps][::-1]
    eff = [100.0 * s[2] / s[1] for s in steps][::-1]
    ax.barh(np.arange(len(eff)), eff, color=C["blue"], alpha=0.85)
    for i, e in enumerate(eff):
        ax.text(e + 0.1, i, "%.1f" % e, va="center", fontsize=7.5)
    ax.set_yticks(np.arange(len(eff))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("每增加 100 万美元的 P72 增量 (百分点)")
    ax.set_title("(b) 边际效益: 先买船, 后买 AUV", fontsize=9)
    ax.grid(alpha=0.25, axis="x")

    # (c) 重复扫描率
    ax = axs[1, 0]
    width = 0.26
    for k, ns in enumerate(sh):
        m = ships == ns
        dups = [100 * np.nan for _ in au]
        for j, na in enumerate(au):
            mm = m & (auvs == na)
            if mm.any():
                dups[j] = 100 * (1 - 0)  # 占位
        # 从 npz 没有 dup, 从 CSV 读
    fn = os.path.join(DATA, "多船多AUV组合.csv")
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    for k, ns in enumerate(sh):
        vals = [100 * float(r["重复扫描率"]) for r in rows if int(r["船数"]) == ns]
        ax.bar(np.arange(len(au)) + (k - 1) * width, vals, width=width,
               color=[C["sky"], C["green"], C["orange"]][k], label="船 %d" % ns)
        for j, v in enumerate(vals):
            ax.text(j + (k - 1) * width, v + 0.4, "%.0f" % v, ha="center", fontsize=6)
    ax.set_xticks(np.arange(len(au))); ax.set_xticklabels(au)
    ax.set_xlabel("AUV 数量"); ax.set_ylabel("跨小时重复扫描率 (%)")
    ax.set_title("(c) 协调负担: 资产越多重复越多", fontsize=9)
    ax.legend(fontsize=7.5, ncol=3); ax.grid(alpha=0.25, axis="y")

    # (d) Pareto 前沿 (单船 7 + 多船 15)
    ax = axs[1, 1]
    ax.scatter(single_costs, single_p72, s=38, marker="s", color=C["gray"],
               label="单船 7 方案", zorder=4)
    ax.scatter(costs, p72, s=38, color=C["red"], label="多船/AUV 15 组合", zorder=5)
    allc = np.array(single_costs + list(costs))
    allp = np.array(single_p72 + list(p72))
    order = np.argsort(allc)
    pareto, best = [], -1
    for i in order:
        if allp[i] > best + 1e-9:
            pareto.append(i); best = allp[i]
    ax.plot(allc[pareto], allp[pareto], "k--", lw=1.0, alpha=0.7, label="Pareto 前沿")
    for i in pareto:
        ax.annotate("%.0f" % allc[i], (allc[i], allp[i]), textcoords="offset points",
                    xytext=(4, -9), fontsize=6.5)
    ax.set_xlabel("成本估计 (万美元)"); ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(85, 100.6)
    ax.set_title("(d) 成本-效能 Pareto 前沿", fontsize=9)
    ax.legend(fontsize=7.2, loc="lower right"); ax.grid(alpha=0.25)

    fig.tight_layout()
    return P.save(fig, "fig_multi_marginal")


# ============================================================================
# 4. β 调优
# ============================================================================
def fig_beta():
    P.setup()
    fn = os.path.join(DATA, "协同调优_beta.csv")
    if not os.path.exists(fn):
        print("  [跳过] 没有 协同调优_beta.csv")
        return None
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    b = [float(r["beta"]) for r in rows]
    p72 = [100 * float(r["P72"]) for r in rows]
    dup = [100 * float(r["跨小时重复率"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(b, p72, "o-", color=C["blue"], lw=1.5, label="72 h 成功率 (%)")
    ax.set_xlabel("重复扫描惩罚系数 β")
    ax.set_ylabel("72 h 内找到比例 (%)", color=C["blue"])
    ax.tick_params(axis="y", labelcolor=C["blue"])
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(b, dup, "s--", color=C["red"], lw=1.4, label="跨小时重复扫描率 (%)")
    ax2.set_ylabel("跨小时重复扫描率 (%)", color=C["red"])
    ax2.tick_params(axis="y", labelcolor=C["red"])
    ax.axvline(1.0, color="0.4", lw=1.0, ls=":")
    ax.text(1.05, min(p72) + 0.3, "采用 β=1.0", fontsize=8, color="0.3")
    lines = [l for l in ax.get_lines() + ax2.get_lines()
             if not l.get_label().startswith("_")]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7.8, loc="center right")
    ax.set_title("协调参数调优: 维持成功率前提下降低重复扫描 (船 2 + AUV 0)", fontsize=9)
    fig.tight_layout()
    return P.save(fig, "fig_multi_beta")


# ============================================================================
# 5. 示例案例: 多资产航迹 + 每格搜索次数
# ============================================================================
def fig_plan():
    P.setup()
    fn = os.path.join(CACHE, "multi_plan_example.json")
    if not os.path.exists(fn):
        print("  [跳过] 没有 multi_plan_example.json")
        return None
    with open(fn, encoding="utf-8") as f:
        d = json.load(f)
    names = d["names"]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.8))

    # (a) 航迹 (只画前 30 h, 避免过密)
    ax = axs[0]
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(names), 1)))
    for k, nm in enumerate(names):
        tr = np.array(d["tracks"][nm], float)
        m = tr[:, 2] <= 12
        if not m.any():
            continue
        ax.plot(tr[m, 0], tr[m, 1], "-", lw=0.8, alpha=0.75, color=cmap[k],
                label=nm if k % 2 == 0 else None)
    ax.plot(0, 0, "*", ms=14, mfc=C["red"], mec="white", mew=0.8, zorder=10)
    ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
    ax.set_aspect("equal"); ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
    ax.legend(fontsize=6.2, ncol=2, framealpha=0.85, loc="lower left")
    ax.set_title("(a) 各资产搜索航迹 (前 12 h; 红★为失事点)", fontsize=9)
    ax.grid(alpha=0.2)

    # (b) 每格被搜索次数
    ax = axs[1]
    pts = []
    for nm in names:
        tr = np.array(d["tracks"][nm], float)
        pts.append(tr[:, :2])
    pts = np.vstack(pts)
    H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=30, range=[[-15, 15], [-15, 15]])
    from matplotlib.colors import LogNorm
    im = ax.imshow(H.T, origin="lower", extent=[-15, 15, -15, 15], cmap="magma",
                   aspect="auto", norm=LogNorm(vmin=1, vmax=max(H.max(), 2)))
    ax.plot(0, 0, "*", ms=12, mfc=C["sky"], mec="white", mew=0.6)
    ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
    ax.set_title("(b) 1 km 网格被搜索次数 (协调后)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("搜索次数", fontsize=8)
    fig.suptitle("示例案例: 船 %d + AUV %d, P72 = %.1f%%"
                 % (d["n_ships"], d["n_auv"], 100 * d["P72"]), fontsize=10)
    fig.tight_layout()
    return P.save(fig, "fig_multi_plan")




# ============================================================================
# 6. 收益阶梯图 (基线 → 增船 → 增 AUV)
# ============================================================================
def fig_waterfall():
    P.setup()
    import csv as _csv
    fn = os.path.join(DATA, "搜索方案对比.csv")
    rows = list(_csv.DictReader(open(fn, encoding="utf-8-sig")))
    m4 = next(r for r in rows if r["方案"] == "M4")
    base_p, base_c = 100 * float(m4["P(72h)"]), float(m4["成本估计(万美元)"])
    d = _load()
    curves, ships, auvs, costs = d["curves"], d["ships"], d["auvs"], d["costs"]

    def p(cost_s, cost_a):
        i = int(np.where((ships == cost_s) & (auvs == cost_a))[0][0])
        return 100 * curves[i, 72], costs[i]

    steps = [("单船 M4", base_c, base_p, None)]
    for (ns, na, label, col) in ((2, 0, "+1 艘船", C["sky"]),
                                 (2, 2, "+2 台 AUV", C["green"]),
                                 (4, 4, "再加船/AUV", C["orange"])):
        pp, cc = p(ns, na)
        steps.append((label, cc, pp, col))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(0, base_p, width=0.6, color=C["gray"], label="单船 M4 (%.0f 万)" % base_c)
    prev = base_p
    for k, (label, cc, pp, col) in enumerate(steps[1:], start=1):
        ax.bar(k, pp - prev, bottom=prev, width=0.6, color=col,
               label="%s → %.1f%% (总 %.0f 万)" % (label, pp, cc))
        ax.text(k, pp + 0.25, "%.1f%%" % pp, ha="center", fontsize=8)
        prev = pp
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(85, 101)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("增援带来的收益阶梯 (基线 89.6%)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return P.save(fig, "fig_multi_waterfall")


# ============================================================================
# 7. 增援到位时间敏感性
# ============================================================================
def fig_delay():
    P.setup()
    fn = os.path.join(DATA, "多船增援延迟.csv")
    if not os.path.exists(fn):
        print("  [跳过] 没有 多船增援延迟.csv")
        return None
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    combos = sorted(set((int(r["船数"]), int(r["AUV数"])) for r in rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    palette = [C["sky"], C["green"], C["orange"], C["purple"]]
    for k, (ns, na) in enumerate(combos):
        sub = [r for r in rows if int(r["船数"]) == ns and int(r["AUV数"]) == na]
        d = [float(r["增援到位(h)"]) for r in sub]
        p = [100 * float(r["P72"]) for r in sub]
        ax.plot(d, p, "o-", color=palette[k % len(palette)], lw=1.4,
                label="船 %d + AUV %d" % (ns, na))
    base = 89.6
    ax.axhline(base, color="0.45", lw=1.0, ls="--")
    ax.text(0.2, base + 0.25, "单船 M4 基线 %.1f%%" % base, fontsize=7.5, color="0.35")
    ax.axvline(3, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("增援到位时间 (h)"); ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(85, 101)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("增援越快越好: 到位时间对 72 h 成功率的影响", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return P.save(fig, "fig_multi_delay")


def main():
    print("多船/多AUV 图件:")
    fig_multi()
    fig_curves()
    fig_marginal()
    fig_beta()
    fig_plan()
    fig_waterfall()
    fig_delay()


if __name__ == "__main__":
    main()
