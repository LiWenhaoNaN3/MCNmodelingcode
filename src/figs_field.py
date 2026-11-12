# -*- coding: utf-8 -*-
"""流场与坡度可视化 (P2 补充图):
  fig_p2_current  — HYCOM 四维流场: 分层流场 + 垂向剖面 + 时间变化 + 被动漂移
  fig_p2_slope    — 海底层坡度场 + 滑移路径 + 沿程逐步重算的坡度
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from . import config, data, plotmpl as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
C = P.C
DEG = 111320.0


def _site():
    return config.SITE["lat"], config.SITE["lon"]


def _km_axes(cur):
    """把流场经纬度网格换成相对失事点的 km 坐标 (格点中心)。"""
    la0, lo0 = _site()
    x = (cur.lon - lo0) * DEG * np.cos(np.deg2rad(la0)) / 1000.0
    y = (cur.lat - la0) * DEG / 1000.0
    return x, y


# ============================================================================
# 图 G1: 四维流场
# ============================================================================
def fig_current(levels=(0, 100, 300, 500)):
    P.setup()
    cur = data.CurrentField4D()
    ter = data.Terrain()
    x, y = _km_axes(cur)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10.6, 8.0))
    gs = fig.add_gridspec(3, 3, hspace=0.36, wspace=0.42)

    # ---- 上两排: 5 个深度的水平流场 ----
    panels = [(0, 0, 0.0), (0, 1, 100.0), (0, 2, 300.0),
              (1, 0, 500.0), (1, 1, 1000.0)]
    vmax = 0.0
    for _, _, zlev in panels:
        ia = int(np.argmin(np.abs(cur.z - zlev)))
        vmax = max(vmax, float(np.nanpercentile(np.hypot(cur.u[0, ia], cur.v[0, ia]), 98)))
    for r, c, zlev in panels:
        ax = fig.add_subplot(gs[r, c])
        ia = int(np.argmin(np.abs(cur.z - zlev)))
        U = cur.u[0, ia]     # 第 0 时刻
        V = cur.v[0, ia]
        spd = np.hypot(U, V)
        sm = ax.pcolormesh(X, Y, spd, shading="auto", cmap="YlGnBu", alpha=0.9,
                           vmin=0, vmax=vmax)
        ax.quiver(X, Y, U, V, color="0.15", scale=5.5, width=0.005,
                  headwidth=3.2, headlength=3.5)
        ax.plot(0, 0, "*", ms=11, mfc=C["red"], mec="white", mew=0.8, zorder=10)
        ax.set_title("z = %d m" % int(cur.z[ia]), fontsize=8.5)
        ax.set_xlabel("东向 (km)"); ax.set_ylabel("北向 (km)")
        ax.set_aspect("equal")
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1])
        cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("m/s", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)

    # ---- 下排左: 垂向剖面 (0 时刻与 48 时刻对比) ----
    ax = fig.add_subplot(gs[1, 2])
    la0, lo0 = _site()
    for t_h, ls in ((0.0, "-"), (48.0, "--")):
        U, V = cur.uv_t(np.full(cur.nlev, la0), np.full(cur.nlev, lo0),
                        cur.z, t_h)
        ax.plot(np.hypot(U, V), cur.z, ls, color=C["blue"] if t_h == 0 else C["orange"],
                lw=1.3, label="t = %g h" % t_h)
    ax.invert_yaxis()
    ax.set_xlabel("流速 (m/s)"); ax.set_ylabel("深度 (m)")
    ax.legend(framealpha=0.9)
    ax.set_title("失事点处的垂向流速剖面", fontsize=8.5)

    # ---- 下排中: 时间变化 (300 m 处) ----
    ax = fig.add_subplot(gs[2, 0])
    th = cur.time_h
    d300 = cur.z[int(np.argmin(np.abs(cur.z - 300.0)))]
    uu = np.array([float(cur.uv_t(la0, lo0, d300, t)[0]) for t in th])
    vv = np.array([float(cur.uv_t(la0, lo0, d300, t)[1]) for t in th])
    ax.plot(th, uu, "-", color=C["blue"], lw=1.2, label="u (东向)")
    ax.plot(th, vv, "-", color=C["orange"], lw=1.2, label="v (北向)")
    ax.plot(th, np.hypot(uu, vv), "-", color=C["black"], lw=1.4, label="流速")
    ax.set_xlabel("时间 (h)"); ax.set_ylabel("流速 (m/s)")
    ax.legend(framealpha=0.9, ncol=3)
    ax.set_title("失事点 %d m 处的流速随时间变化" % int(d300), fontsize=8.5)

    # ---- 下排右: 被动漂移 (不同深度) ----
    ax = fig.add_subplot(gs[2, 1])
    for zlev, col in ((0.0, C["sky"]), (100.0, C["green"]),
                      (300.0, C["orange"]), (500.0, C["red"])):
        px, py = [0.0], [0.0]
        xk, yk = 0.0, 0.0
        dt = 900.0
        for k in range(1, int(96 * 3600 / dt) + 1):
            la = la0 + yk / DEG
            lo = lo0 + xk / (DEG * np.cos(np.deg2rad(la0)))
            u, v = cur.uv_t(la, lo, zlev, k * dt / 3600.0)
            xk += float(u) * dt
            yk += float(v) * dt
            px.append(xk / 1000.0); py.append(yk / 1000.0)
        ax.plot(px, py, "-", color=col, lw=1.4, label="z = %d m" % int(zlev))
        ax.plot(px[-1], py[-1], "o", color=col, ms=3)
    ax.plot(0, 0, "*", ms=11, mfc=C["red"], mec="white", mew=0.8)
    ax.set_xlabel("东向 (km)"); ax.set_ylabel("北向 (km)")
    ax.set_aspect("equal")
    ax.legend(framealpha=0.9)
    ax.set_title("被动漂移 96 h (不含随机误差)", fontsize=8.5)

    ax = fig.add_subplot(gs[2, 2])
    ax.set_axis_off()
    txt = ["四维流场统计", ""]
    for zlev in (0, 100, 300, 500, 1000):
        ia = int(np.argmin(np.abs(cur.z - zlev)))
        spd = np.hypot(cur.u[:, ia], cur.v[:, ia])
        txt.append("z=%4d m  平均 %.3f  峰值 %.3f m/s" % (zlev, np.nanmean(spd), np.nanmax(spd)))
    txt += ["", "时间: 17 个时刻 (6 h 间隔, 0–96 h)", "空间: 20×17 格 ≈ 46×33 km", "深度: 6 层 (0–1000 m)"]
    ax.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=7.2,
            linespacing=1.6)
    fig.suptitle("HYCOM 四维流场 (时间 %d 个时刻 / 深度 %d 层 / %d×%d 网格)"
                 % (cur.nt, cur.nlev, cur.ny, cur.nx), fontsize=9.5)
    return P.save(fig, "fig_p2_current")


# ============================================================================
# 图 G2: 坡度场与滑移路径 (说明坡度按位置逐步重算)
# ============================================================================
def _slide_trace(ter, la, lo, mu_s, ds=20.0, nmax=4000):
    """单点滑移轨迹, 返回 (la[], lo[], slope[], depth[])。"""
    th = np.degrees(np.arctan(mu_s))
    LA, LO, SL, DE = [la], [lo], [float(ter.slope_at(la, lo))], [float(ter.depth_at(la, lo))]
    for _ in range(nmax):
        s = float(ter.slope_at(la, lo))
        if s <= th or not bool(ter.inside(la, lo)):
            break
        ge, gn = ter.grad_at(la, lo)
        ge, gn = float(ge), float(gn)
        n = max(np.hypot(ge, gn), 1e-9)
        la = la + gn / n * ds / DEG
        lo = lo + ge / n * ds / (DEG * np.cos(np.deg2rad(la)))
        LA.append(la); LO.append(lo)
        SL.append(float(ter.slope_at(la, lo))); DE.append(float(ter.depth_at(la, lo)))
    return np.array(LA), np.array(LO), np.array(SL), np.array(DE)


def fig_slope():
    P.setup()
    ter = data.Terrain()
    la0, lo0 = _site()

    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.6),
                            gridspec_kw=dict(width_ratios=[1.2, 1.0, 0.9]))

    # (a) 坡度场 + 若干条滑移路径
    ax = axs[0]
    x = (ter.lon - lo0) * DEG * np.cos(np.deg2rad(la0)) / 1000.0
    y = (ter.lat - la0) * DEG / 1000.0
    X, Y = np.meshgrid(x, y)
    sm = ax.pcolormesh(X, Y, ter.slope_deg, shading="auto", cmap="magma_r",
                       vmin=0, vmax=30)
    ax.contour(X, Y, ter.depth, levels=(200, 500, 876, 1200, 1600, 2000),
               colors="0.6", linewidths=0.35, alpha=0.8)
    rng = np.random.default_rng(0)
    for mu in rng.uniform(0.20, 0.40, 40):
        LA, LO, SL, DE = _slide_trace(ter, la0, lo0, mu)
        ax.plot((LO - lo0) * DEG * np.cos(np.deg2rad(la0)) / 1000.0,
                (LA - la0) * DEG / 1000.0, "-", color="white", lw=0.6, alpha=0.8)
    ax.contour(X, Y, ter.slope_deg, levels=[np.degrees(np.arctan(np.tan(np.deg2rad(21.19))))],
               colors="cyan", linewidths=0.7, linestyles="--")
    ax.plot(0, 0, "*", ms=12, mfc=C["red"], mec="white", mew=0.8, zorder=10)
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("东向 (km)"); ax.set_ylabel("北向 (km)")
    ax.set_title("(a) 坡度场与滑移路径 (白线)", fontsize=8.5)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("坡度 (°)", fontsize=7); cb.ax.tick_params(labelsize=6.5)

    # (b) 沿程坡度: 每 20 m 重算一次
    ax = axs[1]
    LA, LO, SL, DE = _slide_trace(ter, la0, lo0, 0.30)
    d = np.arange(SL.size) * 20.0
    ax.plot(d, SL, "-", color="0.35", lw=1.0, label="沿程坡度 (每 20 m 重算)")
    for mu, col in ((0.22, C["red"]), (0.30, C["orange"]), (0.34, C["green"]),
                    (0.38, C["blue"])):
        LA2, LO2, SL2, DE2 = _slide_trace(ter, la0, lo0, mu)
        d2 = np.arange(SL2.size) * 20.0
        ax.plot(d2[-1], SL2[-1], "o", color=col, ms=5.5, zorder=6,
                label="μ_s=%.2f: 停在 %d m (水深 %d m)" % (mu, d2[-1], DE2[-1]))
        ax.axhline(np.degrees(np.arctan(mu)), color=col, lw=0.7, ls=":")
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("沿下坡方向已走距离 (m)"); ax.set_ylabel("当地坡度 (°)")
    ax.legend(framealpha=0.9)
    ax.set_title("(b) 坡度按位置逐步重算 (虚线 = 摩擦角)", fontsize=8.5)

    # (c) 起点坡度分布 (5000 个落点 ±100 m) vs 单点 21.19°
    ax = axs[2]
    rng = np.random.default_rng(1)
    n = 5000
    xs = rng.normal(0, 100, n); ys = rng.normal(0, 100, n)
    las = la0 + ys / DEG
    los = lo0 + xs / (DEG * np.cos(np.deg2rad(la0)))
    sl = np.atleast_1d(ter.slope_at(las, los))
    ax.hist(sl, bins=np.arange(10, 32, 1.0), color=C["gray"], alpha=0.85,
            edgecolor="white", linewidth=0.4)
    ax.axvline(21.19, color=C["red"], lw=1.4, label="失事点单格值 21.19°")
    ax.axvline(np.percentile(sl, 5), color=C["blue"], lw=0.9, ls="--")
    ax.axvline(np.percentile(sl, 95), color=C["blue"], lw=0.9, ls="--",
               label="5–95%%: %.1f–%.1f°" % (np.percentile(sl, 5), np.percentile(sl, 95)))
    ax.set_xlabel("落点处坡度 (°)"); ax.set_ylabel("样本数")
    ax.legend(framealpha=0.9)
    ax.set_title("(c) 落点坡度分布 (±100 m 定位误差)", fontsize=8.5)

    fig.tight_layout()
    return P.save(fig, "fig_p2_slope")


def main():
    print("流场与坡度图件:")
    fig_current()
    fig_slope()


if __name__ == "__main__":
    main()
