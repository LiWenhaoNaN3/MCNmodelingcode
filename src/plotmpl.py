# -*- coding: utf-8 -*-
"""P2 绘图基础: matplotlib 配置、配色、中文/英文标签表、地形底图。

注意: MPLCONFIGDIR 必须在 import matplotlib 之前设置, 否则字体缓存会写进
     %USERPROFILE%\\.matplotlib (C 盘)。这里统一指到 G:\\meisai\\test1\\.cache\\mpl。
"""
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", ".cache", "mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: E402

from . import config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(os.path.join(ROOT, "..", ".cache", "mpl"), exist_ok=True)

# Okabe-Ito 色盲友好配色
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "gray": "#999999", "black": "#222222",
}

# ---- 中英文标签 ----
_L = {
    "x_km": ("东向距离 (km)", "East offset (km)"),
    "y_km": ("北向距离 (km)", "North offset (km)"),
    "depth_m": ("水深 (m)", "Depth (m)"),
    "t_h": ("时间 (h)", "Time (h)"),
    "prob": ("概率", "Probability"),
    "area": ("包含域面积 (km²)", "Containment area (km²)"),
    "drift": ("漂移距离 (km)", "Drift distance (km)"),
    "site": ("失事点 (最后已知位置)", "Last known position"),
    "terrain": ("ETOPO 2022 15″", "ETOPO 2022 15″"),
}


def L(key):
    """按 config.FIG['lang'] 返回标签。"""
    zh, en = _L[key]
    return zh if config.FIG.get("lang", "zh") == "zh" else en


def _find_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if cand in names:
            return cand
    return None


def setup():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
    })
    if config.FIG.get("lang", "zh") == "zh" and _find_font() is None:
        print("  [警告] 未找到中文字体, 中文标签会显示为方块")


def save(fig, stem):
    """同时输出 PNG(评审看) 与 PDF(以后插 LaTeX)。"""
    os.makedirs(FIGS, exist_ok=True)
    png = os.path.join(FIGS, stem + ".png")
    pdf = os.path.join(FIGS, stem + ".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("   图 -> %s" % png)
    return png


# ---------------- 地形底图 ----------------
def terrain_ax(ax, ter, cmap="Greys", alpha=0.55, vmax=2600.0, contours=(200, 500, 1000, 1500)):
    """在 ax 上画地形底图 (坐标单位 km, 以东/北相对失事点)。"""
    import numpy as np
    from . import data as _data
    xx = (ter.lon - config.SITE["lon"]) * _data.DEG_LAT_M * np.cos(np.deg2rad(config.SITE["lat"])) / 1000.0
    yy = (ter.lat - config.SITE["lat"]) * _data.DEG_LAT_M / 1000.0
    ext = [xx[0], xx[-1], yy[0], yy[-1]]
    ax.imshow(ter.depth, origin="lower", extent=ext, cmap=cmap, alpha=alpha,
              vmin=-100.0, vmax=vmax, zorder=0, aspect="equal")
    # 等深线
    try:
        ax.contour(xx, yy, ter.depth, levels=contours, colors="0.45",
                   linewidths=0.4, alpha=0.6, zorder=1)
    except Exception:
        pass
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(L("x_km"))
    ax.set_ylabel(L("y_km"))
    return ext


def site_marker(ax, label=True):
    ax.plot(0, 0, marker="*", ms=11, mfc=C["red"], mec="white", mew=0.8, zorder=10)
    if label:
        ax.annotate(L("site"), xy=(0, 0), xytext=(7, 7), textcoords="offset points",
                    fontsize=7.5, color=C["red"], zorder=10,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
