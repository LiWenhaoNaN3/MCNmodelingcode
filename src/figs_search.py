# -*- coding: utf-8 -*-
"""搜索评估图件:
  S1 fig_search_curves  7 种方案的 P(找到 | t) 曲线
  S2 fig_search_cost    成功率-成本对比 (72 h) + 期望时间
  S3 fig_search_plan    一个示例案例的搜索航迹 (叠加概率分布)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from . import config, data, equipment as EQ, plotmpl as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
C = P.C


def _load():
    d = np.load(os.path.join(CACHE, "search_results.npz"), allow_pickle=False)
    return (d["curves"], [str(x) for x in d["ids"]], [str(x) for x in d["names"]],
            d["costs"])


# ============================================================================
# S1 曲线
# ============================================================================
def fig_curves():
    P.setup()
    curves, ids, names, costs = _load()
    t = np.arange(curves.shape[1])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = [C["gray"], C["sky"], C["green"], C["blue"], C["purple"], C["orange"], C["red"]]
    for i in range(curves.shape[0]):
        ax.plot(t, 100 * curves[i], "-", lw=1.6, color=colors[i],
                label="%s %s (%.0f 万)" % (ids[i], names[i].replace("+", "+"), costs[i]))
    ax.axvline(72, color="0.5", lw=0.8, ls="--")
    ax.text(72.8, 5, "72 h", fontsize=8, color="0.4")
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel("已找到比例 (%)")
    ax.set_xlim(0, 96); ax.set_ylim(0, 105)
    ax.legend(fontsize=6.6, framealpha=0.9, loc="lower right")
    ax.set_title("7 种装备组合的发现概率曲线 (30 000 粒子)", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return P.save(fig, "fig_search_curves")


# ============================================================================
# S2 成本-效能
# ============================================================================
def fig_cost():
    P.setup()
    curves, ids, names, costs = _load()
    p72 = 100 * curves[:, 72]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5))
    ax = axs[0]
    order = np.argsort(costs)
    ax.barh(np.arange(len(ids)), p72[order], color=C["blue"], alpha=0.85)
    for i, k in enumerate(order):
        ax.text(p72[k] + 1.5, i, "%.0f%% / %.0f万" % (p72[k], costs[k]),
                va="center", fontsize=7)
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels([ids[k] for k in order], fontsize=8)
    ax.set_xlabel("72 h 内找到比例 (%)")
    ax.set_xlim(0, 118)
    ax.set_title("(a) 按成本排序的 72 h 成功率", fontsize=9)

    ax = axs[1]
    ax.scatter(costs, p72, s=45, c=C["red"], zorder=5)
    for i in range(len(ids)):
        ax.annotate(ids[i], (costs[i], p72[i]), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5)
    ax.set_xlabel("单次事故成本估计 (万美元)")
    ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) 成本–效能", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return P.save(fig, "fig_search_cost")


# ============================================================================
# S3 示例搜索航迹
# ============================================================================
def fig_plan():
    P.setup()
    import json
    path = os.path.join(CACHE, "search_plan_example.json")
    if not os.path.exists(path):
        print("  [跳过] 没有 search_plan_example.json")
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for nm in d["names"]:
        tr = np.array(d["tracks"][nm], float)
        ax.plot(tr[:, 0], tr[:, 1], "-", lw=1.0, alpha=0.9, label=nm)
    ax.plot(0, 0, "*", ms=14, mfc=C["red"], mec="white", mew=0.8, zorder=10)
    ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.set_title("示例案例的搜索航迹 (各装备)", fontsize=9)
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
    fig.tight_layout()
    return P.save(fig, "fig_search_plan")


def main():
    print("搜索评估图件:")
    fig_curves()
    fig_cost()
    fig_plan()
    fig_delay()


if __name__ == "__main__":
    main()


# ============================================================================
# S4 搜救启动延迟敏感性
# ============================================================================
def fig_delay():
    import csv
    fn = os.path.join(ROOT, "data", "搜救延迟敏感性.csv")
    if not os.path.exists(fn):
        print("  [跳过] 没有 搜救延迟敏感性.csv")
        return None
    rows = list(csv.DictReader(open(fn, encoding="utf-8-sig")))
    if not rows:
        return None
    schemes = sorted(set(r["方案"] for r in rows))
    P.setup()
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    palette = [C["gray"], C["sky"], C["red"], C["green"], C["purple"]]
    colors = {sc: palette[i % len(palette)] for i, sc in enumerate(schemes)}
    ax = axs[0]
    for sc in schemes:
        sub = [r for r in rows if r["方案"] == sc]
        d = [float(r["延迟(h)"]) for r in sub]
        p72 = [100 * float(r["P(72h)"]) for r in sub]
        ax.plot(d, p72, "o-", color=colors.get(sc), lw=1.3, label=sc)
    ax.set_xlabel("搜救启动延迟 (h)"); ax.set_ylabel("72 h 内找到比例 (%)")
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.set_title("(a) 延迟对 72 h 成功率的影响", fontsize=9)
    ax.grid(alpha=0.25)
    ax = axs[1]
    for sc in schemes:
        sub = [r for r in rows if r["方案"] == sc]
        d = [float(r["延迟(h)"]) for r in sub]
        et = [min(float(r["期望时间(h)"]), 96) for r in sub]
        ax.plot(d, et, "s-", color=colors.get(sc), lw=1.3, label=sc)
    ax.set_xlabel("搜救启动延迟 (h)"); ax.set_ylabel("期望发现时间 (h, 截断 96)")
    ax.set_title("(b) 延迟对期望时间的影响", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return P.save(fig, "fig_search_delay")
