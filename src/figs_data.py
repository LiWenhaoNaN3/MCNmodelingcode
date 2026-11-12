# -*- coding: utf-8 -*-
"""数据章节图: 失事海域地形/深度/坡度 (与当前 30 000 粒子设定一致)。"""
import os

import numpy as np
import matplotlib.pyplot as plt

from . import config, data, plotmpl as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
INPUT = os.path.join(ROOT, "input")
C = P.C


def main():
    P.setup()
    ter = data.Terrain()
    d = np.load(os.path.join(CACHE, "etopo_local.npz"))
    LA, LO, Z = d["lat"], d["lon"], d["z"]
    dep = -Z
    lat0, lon0 = config.SITE["lat"], config.SITE["lon"]
    cosl = np.cos(np.deg2rad(lat0))
    xkm = (np.meshgrid(LA, LO, indexing="ij")[1] - lon0) * 111.32 * cosl
    ykm = (np.meshgrid(LA, LO, indexing="ij")[0] - lat0) * 110.57

    fig = plt.figure(figsize=(11.8, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.52)

    # (a) 地形 + 4x4 km + 出发点
    ax = fig.add_subplot(gs[0])
    m = (np.abs(xkm) <= 12) & (np.abs(ykm) <= 12)
    cf = ax.contourf(xkm[m].reshape(-1, 1) if False else xkm, ykm, dep,
                     levels=[0, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1800],
                     cmap="YlGnBu_r", extend="both")
    ax.contour(xkm, ykm, dep, levels=[400, 800, 1000, 1200], colors="k",
               linewidths=0.4, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, fraction=0.038, pad=0.015)
    cb.set_label("水深 (m)", fontsize=7.5)
    ax.plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], "r--", lw=1.4, label="4 km × 4 km 失事海域")
    ax.plot(0, 0, "r*", ms=11, mec="white", mew=0.7, label="海域中心 (0,0)")
    dep_km = np.hypot((config.SITE["depart_lat"] - lat0) * 110.57,
                      (config.SITE["depart_lon"] - lon0) * 111.32 * cosl)
    ax.plot((config.SITE["depart_lon"] - lon0) * 111.32 * cosl,
            (config.SITE["depart_lat"] - lat0) * 110.57, "o", ms=7,
            mfc=C["orange"], mec="k", mew=0.4, label="出发点 (%.1f km)" % dep_km)
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    ax.set_xlabel(P.L("x_km")); ax.set_ylabel(P.L("y_km"))
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(a) 失事海域地形 (ETOPO 15″)", fontsize=8.5)
    ax.tick_params(labelsize=7)

    # (b) 4x4 km 内海底深度分布 + 失事点当地海底深度
    ax = fig.add_subplot(gs[1])
    inb = (np.abs(xkm) <= 2) & (np.abs(ykm) <= 2)
    ax.hist(dep[inb], bins=np.arange(300, 1801, 50), density=True,
            alpha=0.55, color=C["sky"], label="4×4 km 内全部格点")
    pos = np.load(os.path.join(INPUT, "initial_positions.npz"))
    la = lat0 + pos["y0"] / 111320.0
    lo = lon0 + pos["x0"] / (111320.0 * cosl)
    bot = ter.depth_at(la, lo)
    ax.hist(bot, bins=np.arange(300, 1801, 50), density=True,
            histtype="step", lw=1.4, color=C["red"], label="30 000 个失事点")
    ax.axvline(1000, color="0.4", ls="--", lw=1.0)
    ax.text(1015, ax.get_ylim()[1] * 0.75, "额定潜深 1000 m", fontsize=6.5, color="0.3")
    ax.set_xlabel("水深 (m)"); ax.set_ylabel("概率密度")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)
    ax.set_title("(b) 海底深度分布", fontsize=8.5)
    ax.tick_params(labelsize=7)

    # (c) 失事点坡度分布
    ax = fig.add_subplot(gs[2])
    w = xkm[inb].ravel(); h = ykm[inb].ravel()
    sl_box = ter.slope_at(lat0 + h / 110.57, lon0 + w / (111.32 * cosl))
    ax.hist(sl_box, bins=np.arange(0, 26, 1), density=True, alpha=0.55,
            color=C["gray"], label="4×4 km 内格点")
    ax.hist(ter.slope_at(la, lo), bins=np.arange(0, 26, 1), density=True,
            histtype="step", lw=1.4, color=C["red"], label="30 000 个失事点")
    ax.set_xlabel("海底坡度 (°)"); ax.set_ylabel("概率密度")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25)
    ax.set_title("(c) 海底坡度分布", fontsize=8.5)
    ax.tick_params(labelsize=7)

    fig.suptitle("主算例数据: 伊奥尼亚海失事海域 (%.4f°N, %.4f°E)" % (lat0, lon0), fontsize=9.5)
    return P.save(fig, "fig_D1_region")


if __name__ == "__main__":
    main()
