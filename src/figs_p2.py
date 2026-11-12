# -*- coding: utf-8 -*-
"""P2 图件 (matplotlib):
  F3 fig_p2_framework    模型总体框架与失效模式树
  F4 fig_p2_trajectories 典型轨迹集 (水柱漂流 / 沉底滑移 / 垂向沉降)
  F5 fig_p2_posterior    p(x,y) 后验热图 t = 1 / 6 / 24 / 96 h (悬浮分支)
  F6 fig_p2_uncertainty  A50/A95、深度分布、沉底终点水深、漂移分位数
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from . import config, data, locate, plotmpl as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
C = P.C


# ============================================================================
# F3 框架图
# ============================================================================
def _box(ax, x, y, w, h, text, fc, fs=7.0, ec="0.35", tc="black"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.004,rounding_size=0.015",
                       fc=fc, ec=ec, lw=0.6, zorder=2, transform=ax.transAxes)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            transform=ax.transAxes, color=tc, linespacing=1.35)


def _arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35",
                                shrinkA=0, shrinkB=2), zorder=1)


def fig_framework():
    P.setup()
    fig = plt.figure(figsize=(7.5, 4.3))
    a1 = fig.add_axes([0.01, 0.03, 0.55, 0.92])
    a2 = fig.add_axes([0.60, 0.03, 0.38, 0.92])
    for a in (a1, a2):
        a.set_axis_off()
        a.set_xlim(0, 1); a.set_ylim(0, 1)

    # ---------------- 左: 失效模式树 ----------------
    _box(a1, 0.5, 0.95, 0.62, 0.09, "失联 + 可能失去推进", "#DCE9F5", fs=8)
    modes = [
        ("S0", "失联但正常\n自动上浮", 0.15, "#E8F3E4"),
        ("S1", "失去推进\n近中性", 0.55, "#FDE9C8"),
        ("S2", "断电\n抛载成/败", 0.15, "#E9E4F5"),
        ("S3", "耐压壳进水\n快速下沉", 0.10, "#F8DED6"),
        ("S4", "压载舱进水\n中速下沉", 0.05, "#E8F3E4"),
    ]
    xm = np.linspace(0.105, 0.895, 5)
    for (code, txt, pr, col), x in zip(modes, xm):
        _box(a1, x, 0.72, 0.168, 0.16, "%s\n%s\nP=%.2f" % (code, txt, pr), col, fs=6.0)
        _arrow(a1, (0.5, 0.905), (x, 0.805))

    outs = [
        ("水面\n雷达/目视", 0.16, "#DCE9F5"),
        ("水柱\n定深漂流", 0.5, "#FDE9C8"),
        ("海底\n滑移/碎片", 0.84, "#F8DED6"),
    ]
    for txt, x, col in outs:
        _box(a1, x, 0.36, 0.26, 0.10, txt, col, fs=7)
    _arrow(a1, (0.105, 0.63), (0.16, 0.42))
    _arrow(a1, (0.32, 0.63), (0.44, 0.42))
    _arrow(a1, (0.52, 0.63), (0.30, 0.42))
    _arrow(a1, (0.50, 0.63), (0.50, 0.42))
    _arrow(a1, (0.70, 0.63), (0.78, 0.42))
    _arrow(a1, (0.895, 0.63), (0.86, 0.42))

    _box(a1, 0.5, 0.175, 0.86, 0.125,
         "混合先验: 模式概率 × 参数分布\nΔρ, z0, μ_s, C_D, 初始定位误差, 流场误差",
         "#F2F2F2", fs=6.6)
    _arrow(a1, (0.5, 0.305), (0.5, 0.238))
    _box(a1, 0.5, 0.045, 0.86, 0.07, "蒙特卡洛集合预报  N = 20 000", "#EAF2FA", fs=7.2)
    _arrow(a1, (0.5, 0.115), (0.5, 0.081))
    a1.set_title("(a) 失效模式树", fontsize=9, pad=2)

    # ---------------- 右: 计算管线 ----------------
    steps = [
        ("① 先验采样 (拉丁超立方)", "#DCE9F5"),
        ("② 惯性段: 垂向解析动力学\n+ 海底库仑滑移   dt = 10 s", "#FDE9C8"),
        ("③ 准稳态漂流: 流场平流\n+ OU 误差 + 系统偏差   dt = 900 s", "#FDE9C8"),
        ("④ 后验: p(x,y,z; t)\nA50(t), A95(t), 分支概率", "#E8F3E4"),
        ("⑤ 搜索模型: 发射率→扫宽\n→贝叶斯努力分配", "#E9E4F5"),
        ("⑥ 信息价值: 遥测通道\n降低 A95 / 缩短搜寻时间", "#F2F2F2"),
    ]
    ys = np.linspace(0.93, 0.31, len(steps))
    for (txt, col), y in zip(steps, ys):
        _box(a2, 0.5, y, 0.94, 0.115, txt, col, fs=6.3)
    for i in range(len(steps) - 1):
        _arrow(a2, (0.5, ys[i] - 0.058), (0.5, ys[i + 1] + 0.058))
    a2.set_title("(b) 计算管线", fontsize=9, pad=2)
    return P.save(fig, "fig_p2_framework")


# ============================================================================
# F4 轨迹集
# ============================================================================
def fig_trajectories():
    P.setup()
    d = np.load(os.path.join(CACHE, "p2_ensemble.npz"), allow_pickle=False)
    times = d["snap_times"].astype(float)
    sub = d["sub"].astype(str)
    drho = d["drho"]
    X, Y = d["snap_x"] / 1000.0, d["snap_y"] / 1000.0
    ter = data.Terrain()
    rng = np.random.default_rng(0)

    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.6),
                            gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]))

    # (a) 悬浮分支漂流轨迹
    ax = axs[0]
    P.terrain_ax(ax, ter)
    wat = np.where(np.isin(sub, ("S1", "S2b")))[0]
    idx = rng.choice(wat, size=min(240, wat.size), replace=False)
    for i in idx:
        ax.plot(X[:, i], Y[:, i], "-", color=C["orange"], lw=0.5, alpha=0.16, zorder=3)
    ax.plot(X[-1, idx], Y[-1, idx], ".", color=C["red"], ms=1.8, alpha=0.75, zorder=4)
    P.site_marker(ax)
    ax.set_xlim(-27, 27); ax.set_ylim(-31, 33)
    ax.set_title("(a) 水柱分支: 中性浮力漂流 (至 96 h)", fontsize=8)

    # (b) 沉底分支: 终点 (距离很小, 单独放大)
    ax = axs[1]
    ax.set_aspect("equal")
    bot = np.isin(sub, ("S3", "S4"))
    impl = d["imploded"]
    ax.plot(X[-1, bot & ~impl], Y[-1, bot & ~impl], "o", mfc=C["blue"], mec="none",
            ms=2.2, alpha=0.5, label="沉底 (未压溃)")
    ax.plot(X[-1, bot & impl], Y[-1, bot & impl], "s", mfc=C["red"], mec="none",
            ms=2.2, alpha=0.5, label="压溃 (碎片场)")
    ax.plot(0, 0, "*", ms=11, mfc=C["red"], mec="white", mew=0.8, zorder=10)
    ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
    ax.set_xlim(-4, 4); ax.set_ylim(-3, 3)
    ax.legend(loc="upper right", framealpha=0.9, handletextpad=0.3)
    ax.set_title("(b) 沉底分支终点 (局部放大)", fontsize=8)

    # (c) 垂向沉降
    ax = axs[2]
    for tr, col in zip(locate.trace_vertical(), (C["blue"], C["green"], C["orange"], C["red"])):
        ts, zs, dr = tr
        ax.plot(ts / 60.0, zs, "-", lw=1.2, color=col, label="Δρ = %+.0f kg/m³" % dr)
    ax.axhline(876, color=C["black"], lw=0.8, ls=":", label="海底 876 m")
    ax.invert_yaxis()
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel(P.L("depth_m"))
    ax.legend(framealpha=0.9, handletextpad=0.3)
    ax.set_title("(c) 不同 Δρ 的沉降过程", fontsize=8)

    fig.tight_layout()
    return P.save(fig, "fig_p2_trajectories")


# ============================================================================
# F5 后验热图 (悬浮分支)
# ============================================================================
def fig_posterior(show_times=(1.0, 6.0, 24.0, 96.0)):
    P.setup()
    post = np.load(os.path.join(CACHE, "p2_posterior.npz"), allow_pickle=False)
    times = post["times"]
    grids_w = post["grids_w"]
    edges = post["edges"]
    keys = [str(k) for k in post["keys"]]
    vals = post["values"]
    ter = data.Terrain()
    Xc, Yc = np.meshgrid(edges[:-1], edges[:-1])
    levels = np.array([0.03, 0.07, 0.13, 0.22, 0.35, 0.5, 0.7, 0.85, 1.0])
    cmap = plt.get_cmap("YlOrRd")

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6), sharex=True, sharey=True)
    for k, (ax, th) in enumerate(zip(axs.ravel(), show_times)):
        i = int(np.argmin(np.abs(times - th)))
        H = grids_w[i]
        H = H / H.max() if H.max() > 0 else H
        P.terrain_ax(ax, ter)
        ax.contourf(Xc, Yc, H, levels=levels, cmap=cmap, alpha=0.85, zorder=2, extend="max")
        # 95% 包含域边界
        p = H.ravel()
        order = np.argsort(-p)
        cum = np.cumsum(p[order] / max(p.sum(), 1e-12))
        m = np.zeros(p.size)
        m[order[cum <= 0.95]] = 1.0
        ax.contour(Xc, Yc, m.reshape(H.shape), levels=[0.5], colors="k",
                   linestyles="--", linewidths=0.8, zorder=5)
        P.site_marker(ax, label=(k == 0))
        ax.set_xlim(-27, 27); ax.set_ylim(-27, 35)
        if k < 2:
            ax.set_xlabel("")
        if k % 2 == 1:
            ax.set_ylabel("")
        ax.set_title("t = %g h    A50 = %.0f km²   A95 = %.0f km²"
                     % (th, vals[i, keys.index("A50w_km2")], vals[i, keys.index("A95w_km2")]),
                     fontsize=8)
    fig.subplots_adjust(hspace=0.28, wspace=0.10, right=0.86, top=0.90)
    cax = fig.add_axes([0.88, 0.15, 0.022, 0.68])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("相对概率密度 (各面板独立归一化)", fontsize=8)
    fig.suptitle("水柱分支后验 p(x, y; t)  (条件于 S1+S2b, 占总概率 %.0f%%)"
                 % (100 * (vals[0, keys.index("P_S1")] + vals[0, keys.index("P_S2b")])),
                 fontsize=9)
    return P.save(fig, "fig_p2_posterior")


# ============================================================================
# F6 不确定性
# ============================================================================
def fig_uncertainty():
    P.setup()
    post = np.load(os.path.join(CACHE, "p2_posterior.npz"), allow_pickle=False)
    ens = np.load(os.path.join(CACHE, "p2_ensemble.npz"), allow_pickle=False)
    sub = ens["sub"].astype(str)
    times = post["times"]
    keys = [str(k) for k in post["keys"]]
    vals = post["values"]
    drift = post["drift_dist_km"]
    g = {k: vals[:, j] for j, k in enumerate(keys)}

    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.2))

    ax = axs[0, 0]
    ax.plot(times, g["A95_km2"], "o-", color=C["black"], ms=3, label="混合 (全水下分支)")
    ax.plot(times, g["A95w_km2"], "s-", color=C["orange"], ms=3, label="悬浮分支 A95")
    ax.plot(times, g["A95b_km2"], "^-", color=C["blue"], ms=3, label="沉底分支 A95")
    ax.set_yscale("log")
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel(P.L("area"))
    ax.legend(framealpha=0.9)
    ax.set_title("(a) 包含域面积随时间", fontsize=8)

    ax = axs[0, 1]
    # p(z | t=96h), 按子分支分开画: 深度不确定性是声呐垂直覆盖设计的关键
    zz = ens["snap_z"][-1]
    bins = np.arange(0, 1200, 10)
    for name, mask, col in (("S1 近中性", sub == "S1", C["orange"]),
                            ("S2b 无动力", sub == "S2b", C["blue"]),
                            ("合计", np.isin(sub, ("S1", "S2b")), C["black"])):
        h, _ = np.histogram(zz[mask], bins=bins, density=False)
        ax.plot(h / max(h.sum(), 1), 0.5 * (bins[:-1] + bins[1:]), "-", lw=1.2,
                color=col, label=name)
    ax.invert_yaxis()
    ax.set_xlabel("深度概率密度 p(z | t = 96 h)")
    ax.set_ylabel(P.L("depth_m"))
    ax.set_xlim(0, None)
    ax.legend(framealpha=0.9, handletextpad=0.3)
    ax.set_title("(b) 悬浮分支的深度分布 (t = 96 h)", fontsize=8)

    ax = axs[1, 0]
    bot = np.isin(sub, ("S3", "S4"))
    bot_depth = ens["bot_depth"]
    bins = np.arange(800, 1800, 40)
    ax.hist(bot_depth[bot], bins=bins, color=C["blue"], alpha=0.55, label="沉底分支终点水深")
    ax.axvline(1000, color=C["orange"], lw=0.9, ls="--", label="额定工作深度 1000 m")
    ax.set_xlabel(P.L("depth_m")); ax.set_ylabel("样本数")
    ax2 = ax.twinx()
    th = ens["crush_thresholds"]
    cv = ens["crush_curve_bottom"]
    ax2.plot(th, 100 * cv, "-", color=C["red"], lw=1.4, label="若压溃深度 = 该值")
    ax2.set_ylabel("占沉底分支的概率 (%)", color=C["red"])
    ax2.tick_params(axis="y", colors=C["red"])
    ax2.set_ylim(0, 105)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, framealpha=0.9, handletextpad=0.3, loc="upper left")
    ax.set_title("(c) 沉底终点水深 + 事后压溃概率曲线", fontsize=8)

    ax = axs[1, 1]
    ax.plot(times, drift[:, 1], "o-", color=C["black"], ms=3, label="中位数")
    ax.fill_between(times, drift[:, 0], drift[:, 2], color=C["gray"], alpha=0.3,
                    label="10–90 分位")
    ax.set_xlabel(P.L("t_h")); ax.set_ylabel(P.L("drift"))
    ax.legend(framealpha=0.9)
    ax.set_title("(d) 悬浮分支漂移距离 (距失事点)", fontsize=8)

    fig.tight_layout()
    return P.save(fig, "fig_p2_uncertainty")


# ============================================================================
# G3 失事海域 (失事位置是海域随机点, 不是固定点)
# ============================================================================
def fig_region():
    P.setup()
    ens = np.load(os.path.join(CACHE, "p2_ensemble.npz"), allow_pickle=False)
    x0 = ens["x0"] / 1000.0
    y0 = ens["y0"] / 1000.0
    depth0 = ens["depth0"]
    slope0 = ens["slope0"]
    slide = ens["slide_dist"] > 0
    sub = ens["sub"].astype(str)
    bot = np.isin(sub, ("S3", "S4"))

    ter = data.Terrain()
    reg = config.PRIOR.get("site_region")
    hx = float(reg[1]) if reg else 0.1
    hy = float(reg[2]) if reg else 0.1

    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.5),
                            gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

    # (a) 失事海域: 底图 = 水深, 方框 = 先验海域, 散点 = 抽到的失事点
    ax = axs[0]
    x = (ter.lon - config.SITE["lon"]) * 111320.0 * np.cos(np.deg2rad(config.SITE["lat"])) / 1000.0
    y = (ter.lat - config.SITE["lat"]) * 111320.0 / 1000.0
    X, Y = np.meshgrid(x, y)
    sm = ax.pcolormesh(X, Y, ter.depth, shading="auto", cmap="Blues", vmin=300, vmax=1800)
    ax.contour(X, Y, ter.depth, levels=(400, 600, 800, 1000, 1200), colors="0.5",
               linewidths=0.35)
    ax.add_patch(plt.Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False,
                               ec=C["red"], lw=1.4, ls="--"))
    rng = np.random.default_rng(0)
    idx = rng.choice(x0.size, size=min(1500, x0.size), replace=False)
    ax.plot(x0[idx], y0[idx], ".", color=C["orange"], ms=1.2, alpha=0.6)
    ax.plot(0, 0, "*", ms=11, mfc=C["red"], mec="white", mew=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("东向 (km)"); ax.set_ylabel("北向 (km)")
    ax.set_title("(a) 失事海域: 底图=水深, 方框=先验范围", fontsize=8.5)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("水深 (m)", fontsize=7); cb.ax.tick_params(labelsize=6.5)

    # (b) 失事点的当地海底深度分布
    ax = axs[1]
    ax.hist(depth0[bot], bins=np.arange(300, 1100, 25), color=C["blue"], alpha=0.85,
            label="沉底分支的失事点")
    ax.hist(depth0[~bot], bins=np.arange(300, 1100, 25), color=C["gray"], alpha=0.35,
            label="其余")
    ax.axvline(1000, color=C["orange"], lw=0.9, ls="--", label="额定 1000 m")
    ax.set_xlabel("失事点当地海底深度 (m)"); ax.set_ylabel("样本数")
    ax.legend(framealpha=0.9, handletextpad=0.3)
    ax.set_title("(b) 失事点深度分布 (中位 %.0f m)" % np.median(depth0), fontsize=8.5)

    # (c) 失事点坡度 vs 实际是否滑动
    ax = axs[2]
    edges = np.arange(0, 30, 2.5)
    bc = 0.5 * (edges[:-1] + edges[1:])
    frac = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (slope0 >= a) & (slope0 < b) & bot
        frac.append(100.0 * np.mean(slide[m]) if m.sum() > 20 else np.nan)
    ax.bar(bc, frac, width=2.1, color=C["green"], alpha=0.8)
    ax.set_xlabel("失事点坡度 (°)"); ax.set_ylabel("其中滑动的比例 (%)")
    ax.set_title("(c) 失事点坡度 → 是否滑动", fontsize=8.5)

    fig.tight_layout()
    return P.save(fig, "fig_p2_region")


def main():
    print("P2 图件:")
    fig_framework()
    fig_trajectories()
    fig_posterior()
    fig_uncertainty()
    fig_region()


if __name__ == "__main__":
    main()
