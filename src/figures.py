# -*- coding: utf-8 -*-
"""生成报告用图。全部数据来自 analyze.py 的实测结果。"""
import os

import numpy as np

from . import analyze, config, plotlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

PROF = analyze.Profile(z0=300.0)
TER = analyze.Terrain()
VEH = analyze.vehicle(float(np.interp(PROF.z0, PROF.z, PROF.rho)))


def fig1_terrain():
    X0, X1 = TER.site[1] - 0.12, TER.site[1] + 0.12
    Y0, Y1 = TER.site[0] - 0.12, TER.site[0] + 0.12
    img, d = plotlib.canvas(1200, 1270)
    box = (200, 90, 840, 790)
    p = plotlib.Panel(d, box, (X0, X1), (Y0, Y1),
                      [round(X0, 2), round((X0 + X1) / 2, 2), round(X1, 2)],
                      [round(Y0, 2), round((Y0 + Y1) / 2, 2), round(Y1, 2)],
                      "经度 (°E)", "纬度 (°N)", "潜点局部地形与下滑轨迹", label_size=22, ylab_offset=120)
    i0, i1 = np.searchsorted(TER.lon, X0), np.searchsorted(TER.lon, X1)
    j0, j1 = np.searchsorted(TER.lat, Y0), np.searchsorted(TER.lat, Y1)
    sub = TER.depth[j0:j1, i0:i1]
    img.paste(plotlib.depth_image(sub, box[2] - box[0], box[3] - box[1],
                                  float(sub.min()), float(sub.max())), (box[0], box[1]))
    d.rectangle(box, outline=(40, 40, 40), width=2)

    cols = {0.20: (220, 40, 40), 0.30: (245, 150, 20), 0.40: (30, 110, 210)}
    ends = []
    for mu, col in cols.items():
        pa = TER.slide_path(mu)
        ends.append((mu, col, pa))
        if len(pa) > 1:
            p.line(pa[:, 1], pa[:, 0], color=col, width=5)
            d.ellipse([p.px(pa[-1, 1]) - 7, p.py(pa[-1, 0]) - 7,
                       p.px(pa[-1, 1]) + 7, p.py(pa[-1, 0]) + 7], fill=col,
                      outline=(255, 255, 255), width=2)
    px, py = p.px(TER.site[1]), p.py(TER.site[0])
    d.ellipse([px - 12, py - 12, px + 12, py + 12], fill=(255, 0, 0), outline=(255, 255, 255), width=3)
    d.text((px + 22, py + 20), "失事点 %.0f m" % TER.at(TER.depth, *TER.site),
           font=plotlib.font(20), fill=(210, 0, 0))
    d.text((px + 22, py + 48), "坡度 %.1f°" % TER.at(TER.slope, *TER.site),
           font=plotlib.font(18), fill=(210, 0, 0))

    lx = 870
    d.text((lx, 110), "图例", font=plotlib.font(24), fill=(20, 20, 20))
    for k, (mu, col) in enumerate(cols.items()):
        d.line([(lx, 168 + k * 46), (lx + 62, 168 + k * 46)], fill=col, width=5)
        d.text((lx + 74, 168 + k * 46), "μ_s = %.1f" % mu, font=plotlib.font(20), fill=col, anchor="lm")
    for k, txt in enumerate(["地形 ETOPO 2022 15″", "灰度越暗越深", "轨迹沿最陡下降方向",
                             "停止 坡度 ≤ arctan(μ_s)"]):
        d.text((lx, 330 + k * 32), txt, font=plotlib.font(18), fill=(70, 70, 70))
    d.text((lx, 486), "落点水深", font=plotlib.font(19), fill=(20, 20, 20))
    for k, (mu, col, pa) in enumerate(ends):
        d.text((lx, 520 + k * 34), "μ=%.1f → %.0f m" % (mu, TER.at(TER.depth, pa[-1, 0], pa[-1, 1])),
               font=plotlib.font(18), fill=col)

    mus = np.linspace(0.12, 0.55, 24)
    dep = [TER.at(TER.depth, *(TER.slide_path(mu)[-1])) for mu in mus]
    q = plotlib.Panel(d, (240, 950, 840, 1190), (0.12, 0.55), (0, 1800),
                      [0.2, 0.3, 0.4, 0.5], [0, 600, 1200, 1800], "摩擦系数 μ_s", "落点水深 (m)",
                      "滑行终点水深随 μ_s 的变化", tick_size=16, label_size=18, title_size=20)
    q.line([0.12, 0.55], [1000, 1000], color=(150, 150, 150), width=2)
    d.text((q.x1 - 8, q.py(1000) - 10), "车辆额定 1000 m", font=plotlib.font(15),
           fill=(120, 120, 120), anchor="rb")
    q.line(mus, dep, color=(180, 40, 40), width=4)
    mu_thr = np.tan(np.deg2rad(TER.at(TER.slope, *TER.site)))
    q.dots([mu_thr], [TER.at(TER.depth, *TER.site)], r=6, color=(20, 20, 20))
    d.text((q.px(mu_thr) + 10, q.py(TER.at(TER.depth, *TER.site)) - 14),
           "阈值 μ_s = %.3f" % mu_thr, font=plotlib.font(16), fill=(20, 20, 20))
    plotlib.save(img, os.path.join(FIGS, "fig1_terrain_slide.png"))


def fig2_profile():
    img, d = plotlib.canvas(1240, 560)
    x0, gap, wp = 100, 60, 330
    series = [(PROF.T if len(PROF.T) == len(PROF.z) else np.interp(PROF.z, PROF.z_src, PROF.T),
               (60, 60, 200), "温度 T (°C)"),
              (PROF.sig, (200, 60, 60), "位势密度 σ_t"),
              (PROF.spd, (30, 130, 80), "流速 |u| (m/s)")]
    zmax = PROF.z[-1]
    for k, (s, col, tt) in enumerate(series):
        xx = x0 + k * (wp + gap)
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        pad = (hi - lo) * 0.1 or 1.0
        plotlib.Panel(d, (xx, 80, xx + wp, 460), (lo - pad, hi + pad), (zmax, 0),
                      [round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)],
                      [0, 200, 400, 600, 800], tt, "深度 (m)" if k == 0 else "",
                      tick_size=16, label_size=19)
        # 重画曲线
    # 曲线需在 Panel 之后画, 这里重新绘制一遍
    img, d = plotlib.canvas(1240, 560)
    for k, (s, col, tt) in enumerate(series):
        xx = x0 + k * (wp + gap)
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        pad = (hi - lo) * 0.1 or 1.0
        pp = plotlib.Panel(d, (xx, 80, xx + wp, 460), (lo - pad, hi + pad), (zmax, 0),
                           [round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)],
                           [0, 200, 400, 600, 800], tt, "深度 (m)" if k == 0 else "",
                           tick_size=16, label_size=19)
        pp.line(s, PROF.z, color=col, width=4)
    d.text((620, 30), "HYCOM 实测剖面 (%.2fN %.2fE, 2026-11-01T00Z)"
           % (config.SITE["lat"], config.SITE["lon"]), font=plotlib.font(22), fill=(10, 10, 10), anchor="ma")
    plotlib.save(img, os.path.join(FIGS, "fig2_profile.png"))


def fig3_drift():
    drs = np.linspace(-0.32, 0.085, 80)
    dep = np.array([analyze.equilibrium_depth(PROF, x) for x in drs])
    dep = np.where(dep > 1040, np.nan, dep)
    drift = np.array([float(np.interp(z, PROF.z, PROF.spd)) * 86.4 if not np.isnan(z) else np.nan
                      for z in dep])
    img, d = plotlib.canvas(1220, 880)
    B = (185, 100, 845, 680)
    p = plotlib.Panel(d, B, (-0.32, 0.10), (0, 1100), [-0.3, -0.2, -0.1, 0, 0.1],
                      [0, 300, 600, 900], "浮力异常 Δρ (kg/m³)", "平衡深度 (m)",
                      "Δρ 如何决定平衡深度与漂移距离", label_size=21, tick_size=17, ylab_offset=110)
    p.line(drs, dep, color=(40, 90, 190), width=4)
    d.text((p.px(-0.315), p.py(960)), "平衡深度  ←", font=plotlib.font(18), fill=(40, 90, 190))
    XR = B[2]
    d.line([(XR, B[1]), (XR, B[3])], fill=(40, 40, 40), width=2)
    ry = lambda km: B[3] - km / 12.0 * (B[3] - B[1])
    for km in (0, 3, 6, 9, 12):
        d.line([(XR, ry(km)), (XR + 7, ry(km))], fill=(40, 40, 40), width=2)
        d.text((XR + 14, ry(km)), "%g" % km, font=plotlib.font(17), fill=(30, 30, 30), anchor="lm")
    d.text((XR + 120, (B[1] + B[3]) / 2), "24 h 漂移 (km)", font=plotlib.font(21), fill=(20, 20, 20), anchor="mm")
    d.line([(p.px(x), ry(y)) for x, y in zip(drs, drift) if not np.isnan(y)],
           fill=(205, 70, 40), width=4)
    d.text((p.px(-0.315), ry(7.2)), "24 h 漂移  ←", font=plotlib.font(18), fill=(205, 70, 40))
    d.text((p.px(0.082), B[1] + 24), "Δρ > +0.09 无平衡点", font=plotlib.font(16),
           fill=(110, 110, 110), anchor="rt")
    plotlib.save(img, os.path.join(FIGS, "fig3_drift.png"))


def fig4_band():
    img, d = plotlib.canvas(1160, 780)
    q = plotlib.Panel(d, (150, 100, 1060, 640), (0.0, 0.12), (0, 26),
                      [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12], [0, 5, 10, 15, 20, 25],
                      "Δρ 的不确定度 σ (kg/m³)", "24 h 搜索条带长度 L (km)",
                      "条带长度随浮力不确定度的增长 (t = 24 h)", label_size=21, tick_size=17,
                      ylab_offset=96)
    sgs = np.linspace(0.005, 0.11, 60)
    for dr0, col in ((-0.20, (60, 140, 90)), (-0.05, (40, 90, 190)),
                     (0.00, (220, 140, 30)), (0.05, (200, 50, 50))):
        xs, ys = [], []
        for sg in sgs:
            z1, z2 = analyze.equilibrium_depth(PROF, dr0 + sg), analyze.equilibrium_depth(PROF, dr0 - sg)
            if np.isnan(z1) or np.isnan(z2):
                break
            du = abs(float(np.interp(z2, PROF.z, PROF.spd)) - float(np.interp(z1, PROF.z, PROF.spd)))
            xs.append(sg); ys.append(du * 24 * 3.6)
        if len(xs) > 1:
            q.line(xs, ys, color=col, width=4)
            d.text((q.px(xs[-1]) + 12, q.py(ys[-1])), "Δρ=%+.2f" % dr0, font=plotlib.font(17),
                   fill=col, anchor="lm")
            d.ellipse([q.px(xs[-1]) - 5, q.py(ys[-1]) - 5, q.px(xs[-1]) + 5, q.py(ys[-1]) + 5], fill=col)
    d.text((560, 710), "曲线在 σ 超阈值处终止 —— 那一侧已沉底, 属于分支改变而非数值散布",
           font=plotlib.font(17), fill=(90, 90, 90), anchor="ma")
    plotlib.save(img, os.path.join(FIGS, "fig4_bandlen.png"))


def main():
    fig1_terrain(); print("fig1 ok")
    fig2_profile(); print("fig2 ok")
    fig3_drift(); print("fig3 ok")
    fig4_band(); print("fig4 ok")


if __name__ == "__main__":
    main()