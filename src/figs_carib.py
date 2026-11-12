# -*- coding: utf-8 -*-
"""加勒比海外推图件:
  C1 fig_carib_map      区域地形 + 出发点/失事点 + 流场
  C2 fig_carib_drift    两个海域的漂移距离对比 (由真值集成统计)
  C3 fig_carib_compare  两个海域的 7 方案成功率对比 + P(t) 曲线
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from . import config, data, plotmpl as P, search as SE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
C = P.C
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]


def _load_site(tag):
    suf = ("_" + tag) if tag else ""
    ter = np.load(os.path.join(CACHE, "etopo_local%s.npz" % suf))
    cur = np.load(os.path.join(CACHE, "hycom4d%s.npz" % suf))
    return ter, cur


# ============================================================================
# C1 地形与流场
# ============================================================================
def fig_map():
    P.setup()
    config.set_site("carib")
    ter, cur = _load_site("carib")
    LA, LO, Z = ter["lat"], ter["lon"], ter["z"]
    dep = -Z
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axs[0]
    levels = [0, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000]
    cf = ax.contourf(LO, LA, dep, levels=levels, cmap="YlGnBu_r", extend="max")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("水深 (m)", fontsize=8)
    la0, lo0 = config.SITE["lat"], config.SITE["lon"]
    dla, dlo = config.SITE["depart_lat"], config.SITE["depart_lon"]
    ax.plot(lo0, la0, "^", ms=11, mfc=C["red"], mec="white", mew=0.8,
            label="失事海域中心", zorder=10)
    ax.plot(dlo, dla, "*", ms=14, mfc=C["orange"], mec="white", mew=0.8,
            label="出发点 %s" % config.SITE["depart_name"], zorder=10)
    ax.plot([dlo, lo0], [dla, la0], "k--", lw=1.0, alpha=0.7,
            label="距离 %.1f km" % (np.hypot((la0 - dla) * 110.57,
                                           (lo0 - dlo) * 111.32
                                           * np.cos(np.deg2rad(la0)))))
    # 4x4 km 框
    half = 2.0
    cosl = np.cos(np.deg2rad(la0))
    xs = [lo0 - half / (111.32 * cosl), lo0 + half / (111.32 * cosl)]
    ys = [la0 - half / 110.57, la0 + half / 110.57]
    ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]],
            [ys[0], ys[0], ys[1], ys[1], ys[0]], "r-", lw=1.6)
    ax.set_xlim(LO[0], LO[-1]); ax.set_ylim(LA[0], LA[-1])
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.legend(fontsize=7.2, loc="upper left")
    ax.set_title("(a) 小开曼岛周边地形 (ETOPO 15″)", fontsize=9)
    ax.tick_params(labelsize=7)

    # (b) 表层 100 m 流场
    ax = axs[1]
    lv = int(np.argmin(np.abs(cur["depth"] - 100.0)))
    u = cur["water_u"][0, lv]
    v = cur["water_v"][0, lv]
    spd = np.hypot(u, v)
    cf = ax.pcolormesh(cur["lon"], cur["lat"], spd * 100, cmap="viridis", shading="auto")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("流速 (cm/s, 100 m)", fontsize=8)
    step = 2
    q = ax.quiver(cur["lon"][::step], cur["lat"][::step],
                  u[::step, ::step], v[::step, ::step], color="white", scale=8,
                  width=0.003)
    ax.plot(lo0, la0, "^", ms=10, mfc=C["red"], mec="white", mew=0.8, zorder=10)
    ax.plot(dlo, dla, "*", ms=12, mfc=C["orange"], mec="white", mew=0.8, zorder=10)
    ax.set_xlim(cur["lon"][0], cur["lon"][-1]); ax.set_ylim(cur["lat"][0], cur["lat"][-1])
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.set_title("(b) 100 m 流场 (HYCOM, t=0)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.suptitle("加勒比海外推算例: 小开曼岛西南、开曼海槽边缘", fontsize=10)
    fig.tight_layout()
    return P.save(fig, "fig_carib_map")


# ============================================================================
# C2 漂移对比
# ============================================================================
def _drift(tag):
    suf = ("_" + tag) if tag else ""
    d = np.load(os.path.join(CACHE, "truth_search%s.npz" % suf))
    pos = d["pos"]
    init = pos[0, :, :2] / 1000.0
    final = pos[-1, :, :2] / 1000.0
    return np.hypot(*(final - init).T)


def fig_drift():
    P.setup()
    dm = _drift(None)
    dc = _drift("carib")
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axs[0]
    bins = np.linspace(0, 80, 60)
    ax.hist(dm, bins=bins, density=True, alpha=0.55, color=C["sky"], label="地中海 (主算例)")
    ax.hist(dc, bins=bins, density=True, alpha=0.55, color=C["orange"], label="加勒比海 (外推)")
    ax.set_xlabel("96 h 水平漂移距离 (km)")
    ax.set_ylabel("概率密度")
    ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("(a) 96 h 漂移距离分布", fontsize=9)
    ax = axs[1]
    qs = [10, 25, 50, 75, 90, 99]
    x = np.arange(len(qs))
    w = 0.38
    ax.bar(x - w / 2, np.percentile(dm, qs), w, color=C["sky"], label="地中海")
    ax.bar(x + w / 2, np.percentile(dc, qs), w, color=C["orange"], label="加勒比海")
    ax.set_xticks(x); ax.set_xticklabels(["%d%%" % q for q in qs])
    ax.set_ylabel("漂移距离 (km)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    ax.set_title("(b) 漂移距离分位对比", fontsize=9)
    fig.tight_layout()
    return P.save(fig, "fig_carib_drift")


# ============================================================================
# C3 搜索效果对比
# ============================================================================
def _curves(tag):
    suf = ("_" + tag) if tag else ""
    d = np.load(os.path.join(CACHE, "search_results%s.npz" % suf))
    return d["curves"], [str(x) for x in d["ids"]], [str(x) for x in d["names"]], d["costs"]


def _csv(tag):
    suf = ("_" + tag) if tag else ""
    fn = os.path.join(DATA, "搜索方案对比%s.csv" % suf)
    return list(csv.DictReader(open(fn, encoding="utf-8-sig")))


def fig_compare():
    P.setup()
    rows_m, rows_c = _csv(None), _csv("carib")
    cm, ids, names, costs = _curves(None)
    cc, _, _, _ = _curves("carib")
    fig = plt.figure(figsize=(10.6, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    # (a) P72 柱状对比
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(ids))
    w = 0.38
    pm = [100 * float(r["P(72h)"]) for r in rows_m]
    pc = [100 * float(r["P(72h)"]) for r in rows_c]
    ax.bar(x - w / 2, pm, w, color=C["sky"], label="地中海 (主算例)")
    ax.bar(x + w / 2, pc, w, color=C["orange"], label="加勒比海 (外推)")
    for i in range(len(ids)):
        ax.text(i - w / 2, pm[i] + 0.7, "%.0f" % pm[i], ha="center", fontsize=7)
        ax.text(i + w / 2, pc[i] + 0.7, "%.0f" % pc[i], ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s %s" % (ids[i], names[i].replace("+", "+")) for i in range(len(ids))],
                       fontsize=7)
    ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    ax.set_title("(a) 7 种装备组合: 两个海域的 72 h 成功率", fontsize=9)

    # (b) 全部曲线 (地中海 实线, 加勒比 虚线)
    ax = fig.add_subplot(gs[1, 0])
    t = np.arange(cm.shape[1])
    for i in range(cm.shape[0]):
        ax.plot(t, 100 * cm[i], "-", lw=1.1, color=COLORS[i % len(COLORS)],
                label="%s 地中海" % ids[i])
        ax.plot(t, 100 * cc[i], "--", lw=1.1, color=COLORS[i % len(COLORS)])
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel("已找到比例 (%)")
    ax.set_xlim(0, 96); ax.set_ylim(0, 102)
    ax.legend(fontsize=6.2, ncol=2)
    ax.grid(alpha=0.25)
    ax.set_title("(b) P(t) 曲线 (实线=地中海, 虚线=加勒比)", fontsize=9)

    # (c) 期望时间/成功率散点
    ax = fig.add_subplot(gs[1, 1])
    etm = [float(r["期望发现时间(h)"]) for r in rows_m]
    etc = [float(r["期望发现时间(h)"]) for r in rows_c]
    ax.scatter(etm, pm, s=42, color=C["sky"], label="地中海", zorder=5)
    ax.scatter(etc, pc, s=42, color=C["orange"], label="加勒比海", zorder=5)
    for i in range(len(ids)):
        ax.annotate(ids[i], (etm[i], pm[i]), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5)
        ax.annotate(ids[i], (etc[i], pc[i]), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5)
    ax.set_xlabel("期望发现时间 (h, 截断 96)"); ax.set_ylabel("72 h 内找到比例 (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("(c) 时间-成功率", fontsize=9)
    return P.save(fig, "fig_carib_compare")


def main():
    print("加勒比海外推图件:")
    fig_map()
    fig_drift()
    fig_compare()
    fig_current()
    fig_driftmap()


if __name__ == "__main__":
    main()


# ============================================================================
# C4 流场对比: 剖面流速 + 四维场分深度统计
# ============================================================================
def fig_current():
    P.setup()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axs[0]
    for tag, name, col in ((None, "地中海", C["sky"]), ("carib", "加勒比海", C["orange"])):
        suf = ("_" + tag) if tag else ""
        p = np.load(os.path.join(CACHE, "profile%s.npz" % suf))
        m = ~np.isnan(p["water_temp"])
        z = p["depth"][m]; spd = np.hypot(p["water_u"][m], p["water_v"][m])
        ax.plot(spd, z, "o-", ms=3, lw=1.3, color=col, label=name)
    ax.set_ylim(1000, 0)
    ax.set_xlabel("流速 (m/s)"); ax.set_ylabel("深度 (m)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_title("(a) 站点剖面流速", fontsize=9)

    ax = axs[1]
    depths = [100, 300, 500, 1000]
    width = 0.38
    for k, (tag, name, col) in enumerate(((None, "地中海", C["sky"]),
                                          ("carib", "加勒比海", C["orange"]))):
        suf = ("_" + tag) if tag else ""
        cur = np.load(os.path.join(CACHE, "hycom4d%s.npz" % suf))
        vals = []
        for d in depths:
            lv = int(np.argmin(np.abs(cur["depth"] - d)))
            spd = np.hypot(cur["water_u"][:, lv], cur["water_v"][:, lv])
            vals.append(np.percentile(spd[~np.isnan(spd)], 90))
        xs = np.arange(len(depths)) + (k - 0.5) * width
        ax.bar(xs, vals, width, color=col, label=name)
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels(["%d m" % d for d in depths])
    ax.set_ylabel("流速 90% 分位 (m/s)")
    ax.set_xlabel("深度层")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    ax.set_title("(b) 四维流场各深度层流速 (0–96 h, 90% 分位)", fontsize=9)
    fig.tight_layout()
    return P.save(fig, "fig_carib_current")


# ============================================================================
# C5 终点位置散布 (漂移方向与离散度)
# ============================================================================
def fig_driftmap():
    P.setup()
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.4))
    names = {0: "水面", 1: "水柱", 2: "海底"}
    cols = {0: C["sky"], 1: C["green"], 2: C["red"]}
    for ax, (tag, name) in zip(axs, ((None, "地中海 (主算例)"), ("carib", "加勒比海 (外推)"))):
        suf = ("_" + tag) if tag else ""
        d = np.load(os.path.join(CACHE, "truth_search%s.npz" % suf))
        pos, cls = d["pos"], d["cls"]
        fin = pos[-1, :, :2] / 1000.0
        for c in (0, 1, 2):
            m = cls[-1] == c
            ax.scatter(fin[m, 0], fin[m, 1], s=1.2, alpha=0.25, color=cols[c],
                       label="%s (%.0f%%)" % (names[c], 100 * m.mean()))
        ax.plot(0, 0, "*", ms=12, mfc=C["orange"], mec="white", mew=0.7)
        ax.set_xlim(-70, 70); ax.set_ylim(-70, 70)
        ax.set_aspect("equal")
        ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
        ax.set_title("%s: 96 h 终点位置" % name, fontsize=9)
        ax.legend(fontsize=7, markerscale=5, loc="upper left")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return P.save(fig, "fig_carib_driftmap")
