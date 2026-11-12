# -*- coding: utf-8 -*-
"""定位模型的统计图件与包含域 (基于当前 30 000 粒子真值集, 与搜索模型同一套粒子)。

与旧 posterior.py 的区别:
  * 旧版由 P2 集合 (4×4 掩膜先验 + 三角深度) 计算, 与当前"上层给定初始位置"不一致;
  * 本模块直接读 cache/truth_search[_tag].npz, 用**相对自身初始位置**的漂移量做包含域,
    对应"搜救方从 100 m 圆柱出发, t 时刻需要搜索多大面积"的问题。
"""
import csv
import os

import numpy as np

import matplotlib.pyplot as plt

from . import config, plotmpl as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

HALF_KM = 300.0
DXY_KM = 2.0
NG = int(round(2 * HALF_KM / DXY_KM))
T_HOURS = [1, 2, 4, 8, 12, 24, 36, 48, 72, 96]


def _tag_suffix(tag):
    return ("_" + tag) if tag else ""


def _blur(H, n=2):
    for _ in range(n):
        H = 0.5 * H + 0.125 * (np.roll(H, 1, 0) + np.roll(H, -1, 0)
                               + np.roll(H, 1, 1) + np.roll(H, -1, 1))
    return H


def _area_of_containment(grid, dxy, frac):
    """按概率从高到低累加, 达到 frac 的总概率所需面积 (km²)。"""
    p = grid.ravel()
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    if cum[-1] <= 0:
        return float("nan")
    cum /= cum[-1]
    k = int(np.searchsorted(cum, frac)) + 1
    return float(k * dxy * dxy)


def _grid_xy(xkm, ykm):
    """散点 → 网格密度 (cells/km², 已平滑归一)。"""
    j = np.clip(((xkm + HALF_KM) / DXY_KM).astype(int), 0, NG - 1)
    i = np.clip(((ykm + HALF_KM) / DXY_KM).astype(int), 0, NG - 1)
    H = np.zeros((NG, NG))
    np.add.at(H, (i, j), 1.0)
    n = H.sum()
    H = _blur(H) if n > 0 else H
    s = H.sum()
    return H / s if s > 0 else H


def load_truth(tag=None):
    d = np.load(os.path.join(CACHE, "truth_search%s.npz" % _tag_suffix(tag)),
                allow_pickle=False)
    return {k: d[k] for k in d.files}


def compute(tag=None, verbose=True):
    d = load_truth(tag)
    pos, cls = d["pos"], d["cls"]
    times = d["times"]
    x0, y0 = pos[0, :, 0], pos[0, :, 1]
    rows = []
    grids = {}
    for th in T_HOURS:
        k = int(np.argmin(np.abs(times - th)))
        xr = (pos[k, :, 0] - x0) / 1000.0
        yr = (pos[k, :, 1] - y0) / 1000.0
        c = cls[k]
        g_all = _grid_xy(xr, yr)
        g_w = _grid_xy(xr[c == 1], yr[c == 1])
        g_b = _grid_xy(xr[c == 2], yr[c == 2])
        r = np.hypot(xr, yr)
        rows.append(dict(
            t_h=float(times[k]),
            A50=_area_of_containment(g_all, DXY_KM, 0.50),
            A95=_area_of_containment(g_all, DXY_KM, 0.95),
            A50w=_area_of_containment(g_w, DXY_KM, 0.50),
            A95w=_area_of_containment(g_w, DXY_KM, 0.95),
            A50b=_area_of_containment(g_b, DXY_KM, 0.50),
            A95b=_area_of_containment(g_b, DXY_KM, 0.95),
            r50=float(np.percentile(r, 50)),
            r95=float(np.percentile(r, 95)),
            dr50=float(np.percentile(r, 50)),
            surf=float(np.mean(c == 0)), water=float(np.mean(c == 1)),
            bot=float(np.mean(c == 2)),
        ))
        grids[float(times[k])] = dict(all=g_all, water=g_w, bottom=g_b)
    if verbose:
        print("%s: t(h)  A50   A95  | 水柱A95 沉底A95 | 水面/水柱/海底 | r50 r95"
              % ("carib" if tag else "med"))
        for r_ in rows:
            print("  %4.0f %6.0f %7.0f | %6.0f %8.0f | %.2f %.2f %.2f | %.1f %.1f"
                  % (r_["t_h"], r_["A50"], r_["A95"], r_["A95w"], r_["A95b"],
                     r_["surf"], r_["water"], r_["bot"], r_["r50"], r_["r95"]))
    return rows, grids


def save(rows, grids, tag=None):
    suf = _tag_suffix(tag)
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "定位不确定性%s.csv" % suf)
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t(h)", "A50(km2)", "A95(km2)", "水柱A50", "水柱A95",
                    "沉底A50", "沉底A95", "r50(km)", "r95(km)",
                    "水面占比", "水柱占比", "海底占比"])
        for r in rows:
            w.writerow(["%.0f" % r["t_h"], "%.0f" % r["A50"], "%.0f" % r["A95"],
                        "%.0f" % r["A50w"], "%.0f" % r["A95w"],
                        "%.0f" % r["A50b"], "%.0f" % r["A95b"],
                        "%.1f" % r["r50"], "%.1f" % r["r95"],
                        "%.3f" % r["surf"], "%.3f" % r["water"], "%.3f" % r["bot"]])
    np.savez_compressed(os.path.join(CACHE, "locate_stats%s.npz" % suf),
                        rows=np.array([[*r.values()] for r in rows], dtype=float),
                        keys=np.array(list(rows[0].keys())))
    print("已保存 -> %s" % fn)


def fig_posterior(tag=None):
    """p(x,y) 相对初始位置的漂移分布 (t = 1, 12, 24, 96 h), 水柱+沉底合并。"""
    P.setup()
    _, grids = compute(tag, verbose=False)
    show = [1.0, 12.0, 24.0, 96.0]
    VIEW = 60.0
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 7.0))
    for ax, th in zip(axs.ravel(), show):
        H = grids[th]["all"] / (DXY_KM * DXY_KM)     # 密度 (1/km²)
        im = ax.imshow(np.log10(np.maximum(H, 1e-8)), origin="lower",
                       extent=[-HALF_KM, HALF_KM, -HALF_KM, HALF_KM],
                       cmap="viridis", vmin=-4.0, vmax=0.5)
        ax.plot(0, 0, "o", ms=5, mfc="white", mec="k", mew=0.6)
        ax.add_patch(plt.Rectangle((-0.1, -0.1), 0.2, 0.2, fill=False,
                                   ec="r", lw=1.2))
        ax.set_title("t = %.0f h" % th, fontsize=9)
        ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
        ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW)
        ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label(r"$\log_{10}$ 概率密度 (1/km$^2$)", fontsize=8)
    fig.suptitle("位置概率分布 (相对失事点, 红框 = 初始 100 m 圆柱), %s"
                 % ("加勒比海" if tag else "地中海"), fontsize=10)
    return P.save(fig, "fig_L1_posterior%s" % _tag_suffix(tag))


def fig_uncertainty(tag=None):
    """包含域增长 / 分支占比 / 深度分布。"""
    P.setup()
    d = load_truth(tag)
    rows, _ = compute(tag, verbose=False)
    t = [r["t_h"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6))
    ax = axs[0]
    ax.plot(t, [r["A95"] for r in rows], "o-", color=P.C["blue"], label="全部")
    ax.plot(t, [r["A95w"] for r in rows], "s--", color=P.C["green"], label="水柱分支")
    ax.plot(t, [r["A95b"] for r in rows], "^:", color=P.C["red"], label="沉底分支")
    ax.set_yscale("log")
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel("95% 包含域面积 (km$^2$)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
    ax.set_title("(a) 搜救面积 A95(t)", fontsize=9)

    ax = axs[1]
    ax.stackplot(t, [r["surf"] for r in rows], [r["water"] for r in rows],
                 [r["bot"] for r in rows],
                 colors=[P.C["sky"], P.C["green"], P.C["red"]],
                 labels=["水面", "水柱", "海底"], alpha=0.85)
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel("占比")
    ax.set_ylim(0, 1); ax.set_xlim(0, 96)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(b) 目标类别构成随时间", fontsize=9)

    ax = axs[2]
    pos, cls = d["pos"], d["cls"]
    k = int(np.argmin(np.abs(d["times"] - 96)))
    zw = pos[k, cls[k] == 1, 2]
    zb = pos[k, cls[k] == 2, 2]
    bins = np.arange(0, 1200, 50)
    ax.hist(zw, bins=bins, density=True, alpha=0.6, color=P.C["green"], label="水柱")
    ax.hist(zb, bins=bins, density=True, alpha=0.6, color=P.C["red"], label="海底")
    ax.set_xlabel("深度 (m)"); ax.set_ylabel("概率密度")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    ax.set_title("(c) 96 h 深度分布", fontsize=9)
    fig.suptitle("定位不确定性 (%s)" % ("加勒比海" if tag else "地中海"), fontsize=10)
    fig.tight_layout()
    return P.save(fig, "fig_L2_uncertainty%s" % _tag_suffix(tag))


def main():
    print("== 定位统计 (两个海域) ==")
    rows_m, g_m = compute(None)
    save(rows_m, g_m, None)
    rows_c, g_c = compute("carib")
    save(rows_c, g_c, "carib")
    fig_posterior(None)
    fig_uncertainty(None)
    fig_posterior("carib")
    fig_uncertainty("carib")


if __name__ == "__main__":
    main()
