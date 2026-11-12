# -*- coding: utf-8 -*-
"""上层接口: 生成"初始位置"集成 (不内置任何初始位置数据, 全部由参数计算)。

定义 (按用户给定):
  1. 失事海域: 以 config.SITE 为中心、4 km × 4 km 的矩形, 水平均匀;
  2. 失事深度: 指数分布, 特征深度 200 m, 即 p(z) ∝ exp(−z/200), 截断于
     [0, min(当地海底深度 − 5, 额定潜深 1000 m)];
  3. 每次搜救时, 搜救方知道目标在"水平半径 100 m、垂直 ±10 m 的圆柱"内
     → 圆柱中心 C 从上述分布抽取, 真实初始位置 = C + 圆柱内均匀偏移。

输出 (供定位模型与搜索模型读取):
  input/initial_positions.npz
    case_id[N]   所属搜救案例
    x_hat[N]     圆柱中心 (东向, m, 相对 SITE)
    y_hat[N]     圆柱中心 (北向, m)
    z_hat[N]     圆柱中心 (深度, m)
    x0,y0,z0[N]  真实初始位置 (x,y 相对 SITE, m; z 深度 m)
  input/meta.json 生成参数
"""
import json
import os

import numpy as np

from . import config, data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "input")

DEG = 111320.0


def _latlon(ter, x, y):
    la = config.SITE["lat"] + np.asarray(y, float) / DEG
    lo = config.SITE["lon"] + np.asarray(x, float) / (
        DEG * np.cos(np.deg2rad(config.SITE["lat"])))
    return la, lo


def _exp_trunc(rng, n, scale, zmax):
    """截断指数分布: p(z) ∝ exp(−z/scale), z ∈ [0, zmax]。"""
    u = rng.random(n)
    fm = 1.0 - np.exp(-zmax / scale)
    return -scale * np.log(1.0 - u * fm)


def generate(n_cases=60, k_per_case=500, half_km=2.0, depth_scale=200.0,
             z_min=200.0, cyl_r=100.0, cyl_dz=10.0, seed=7, save=True):
    """生成初始位置集成。默认 60 个案例 × 500 粒子 = 30 000 粒子。"""
    rng = np.random.default_rng(seed)
    ter = data.Terrain()
    N = int(n_cases) * int(k_per_case)

    # --- 案例中心: 水平均匀 4x4 km, 深度指数(特征 200 m) ---
    cx = rng.uniform(-half_km * 1000.0, half_km * 1000.0, n_cases)
    cy = rng.uniform(-half_km * 1000.0, half_km * 1000.0, n_cases)
    la, lo = _latlon(ter, cx, cy)
    bottom = ter.depth_at(la, lo)
    ztop = float(config.PRIOR["depth_rating_m"])       # 额定潜深 1000 m
    zmax = np.minimum(bottom - 5.0, ztop) - z_min       # 相对 z_min 的可采样厚度
    zmax = np.maximum(zmax, 1.0)
    cz = z_min + np.array([_exp_trunc(rng, 1, depth_scale, float(z))[0] for z in zmax])

    # --- 每个案例内: 圆柱内均匀偏移 (水平圆盘, 垂直 ±cyl_dz) ---
    case_id = np.repeat(np.arange(n_cases), k_per_case)
    x_hat = np.repeat(cx, k_per_case)
    y_hat = np.repeat(cy, k_per_case)
    z_hat = np.repeat(cz, k_per_case)
    r = cyl_r * np.sqrt(rng.random(N))
    th = rng.uniform(0.0, 2.0 * np.pi, N)
    dx = r * np.cos(th)
    dy = r * np.sin(th)
    dz = rng.uniform(-cyl_dz, cyl_dz, N)
    x0 = x_hat + dx
    y0 = y_hat + dy
    z0 = z_hat + dz
    la0, lo0 = _latlon(ter, x0, y0)
    bot0 = ter.depth_at(la0, lo0)
    z0 = np.clip(z0, 5.0, np.maximum(bot0 - 5.0, 5.0))

    meta = {"n_cases": int(n_cases), "k_per_case": int(k_per_case),
            "n_particles": N, "center_lat": config.SITE["lat"],
            "center_lon": config.SITE["lon"], "region_half_km": float(half_km),
            "depth_scale_m": float(depth_scale), "cyl_r_m": float(cyl_r),
            "cyl_dz_m": float(cyl_dz), "seed": int(seed)}
    out = {"case_id": case_id, "x_hat": x_hat, "y_hat": y_hat, "z_hat": z_hat,
           "x0": x0.astype(np.float32), "y0": y0.astype(np.float32),
           "z0": z0.astype(np.float32)}
    tag = getattr(config, "SITE_TAG", None)
    meta["site_tag"] = tag
    meta["site"] = config.SITE.get("region")
    if save:
        os.makedirs(INPUT, exist_ok=True)
        fn = "initial_positions%s.npz" % (("_" + tag) if tag else "")
        np.savez_compressed(os.path.join(INPUT, fn), **out)
        with open(os.path.join(INPUT, "meta%s.json" % (("_" + tag) if tag else "")),
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print("已保存 -> input/%s (%d 粒子, %d 案例)" % (fn, N, n_cases))
    return out, meta


def load(path=None):
    tag = getattr(config, "SITE_TAG", None)
    path = path or os.path.join(INPUT, "initial_positions%s.npz"
                                % (("_" + tag) if tag else ""))
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def summary(d=None):
    d = d or load()
    x, y, z = d["x0"], d["y0"], d["z0"]
    print("初始位置集成: %d 粒子 / %d 案例" % (len(x), int(d["case_id"].max()) + 1))
    print("  水平: 东 %.1f–%.1f km, 北 %.1f–%.1f km" % (x.min() / 1e3, x.max() / 1e3,
                                                    y.min() / 1e3, y.max() / 1e3))
    print("  深度: 分位 1%%/50%%/99%% = %.0f / %.0f / %.0f m"
          % tuple(np.percentile(z, [1, 50, 99])))
    print("  深度均值 %.0f m" % z.mean())


if __name__ == "__main__":
    generate()
    summary()
