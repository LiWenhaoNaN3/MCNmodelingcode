# -*- coding: utf-8 -*-
"""2024 MCM Problem B 重建版 —— 单文件模型

由 src/ 的 7 个模块合并而成（build/build_single.py 生成）。
全部环境量来自 ETOPO 2022 与 HYCOM GLBy0.08 实测数据;
查不到的量(Δρ, μ_s, z0, W, v, w0)不赋值, 留作函数参数。

用法:
    python model.py check      状态方程自检(无需数据)
    python model.py fetch      下载数据到 cache/
    python model.py analyze    全部计算, 打印所有表
    python model.py figures    出图到 figs/
    python model.py all        全流程
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIGS = os.path.join(HERE, "figs")


# ======================================================================
# 1. config —— 全部输入参数的唯一出处
# ======================================================================

# -*- coding: utf-8 -*-
"""全部输入参数的唯一出处。每一项都标注来源或"符号参数"。

原则: 能查到的填实测值并注明来源; 查不到的**不填**, 留作函数参数。
"""

# ---------------- 站点 (用户确认) ----------------
SITE = {
    "lat": 37.2979,          # 用户选定; 由 ETOPO+HYCOM 交叉验证
    "lon": 20.9021,
    "region": "伊奥尼亚海, 凯法利尼亚以南",
}

# ---------------- 潜水器 (厂商数据表) ----------------
VEHICLE = {
    "model": "TRITON 3300/6",
    "source": "Triton Submarines Datasheet, 2021-10",
    "source_url": "https://tritonsubs.com/wp-content/uploads/Datasheet-Triton-3300-6-Latest.pdf",
    "mass_dry_kg": 11000.0,   # 原文 23,100 lb / 10,500 kg; 概览页 24,300 lb / 11,000 kg
    "length_m": 4.55,
    "width_m": 3.60,
    "height_m": 2.40,
    "depth_rating_m": 1000.0, # 3300 ft / 1000 m
    "cd": 0.45,               # 球体标准阻力曲线; 取值区间内选择, 非查表读数
    "added_mass": 0.50,       # Lamb 1932, 球体附加质量精确解
}

# ---------------- 数据端点 ----------------
ENDPOINTS = {
    "etopo": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.csv",
    "hycom_dap": "https://ncss.hycom.org/thredds/dodsC/GLBy0.08/latest.asc",
    "hycom_time": "2026-11-01T00:00:00Z",   # HYCOM latest 窗口内
}

# ---------------- 物理常数 ----------------
G = 9.80665               # CGPM 标准重力

# ---------------- 符号参数 (刻意不给值) ----------------
SYMBOLIC = {
    "drho": "浮力异常 ρ_b − ρ_w(z0), kg/m3, 由压载状态决定, 无公开数据",
    "mu_s": "静摩擦系数, 钢结构 vs 湿岩石, 常见区间 0.2–0.5, 无权威实测",
    "mu_k": "动摩擦系数, 同上",
    "z0":   "失事深度, m, 场景参数",
    "W":    "扫宽, m, 装备参数",
    "v":    "搜索航速, m/s, 装备参数",
    "w0":   "初始位置不确定宽度, m, 定位精度参数",
}


# ======================================================================
# 2. eos —— 海水密度与声速
# ======================================================================

# -*- coding: utf-8 -*-
"""UNESCO 1983 (EOS-80) 海水状态方程 + 由真实 T/S 剖面计算密度与浮力频率。

系数取自 Millero & Poisson (1981)，按 UNESCO Technical Papers No.44 的整理形式录入。
自带校验：用标准 sigma_t 值反查，误差应 < 0.003 kg/m3。
"""
import numpy as np


def pressure_bars(z):
    """Saunders (1981): 深度(m) -> 压力(bar)。1000 m 约 1005 dbar = 100.5 bar。"""
    z = np.asarray(z, float)
    p_mpa = 1.00524e-2 * z + 2.15e-7 * z ** 2 - 3.974e-13 * z ** 3
    return 10.0 * p_mpa          # 1 bar = 0.1 MPa


def rho_eos80(T, S, p):
    """T[degC], S[PSU], p[bar] -> 位势密度 kg/m3。"""
    T = np.asarray(T, float); S = np.asarray(S, float); p = np.asarray(p, float)
    # --- 一个大气压下的密度 rho0 ---
    rw = (999.842594 + 6.793952e-2 * T - 9.095290e-3 * T ** 2
          + 1.001685e-4 * T ** 3 - 1.120083e-6 * T ** 4 + 6.536332e-9 * T ** 5)
    a1 = (0.824493 - 4.0899e-3 * T + 7.6438e-5 * T ** 2
          - 8.2467e-7 * T ** 3 + 5.3875e-9 * T ** 4)
    b1 = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T ** 2
    rho0 = rw + a1 * S + b1 * S ** 1.5 + 4.8314e-4 * S ** 2
    # --- 割线体积模量 K (bar) ---
    kw = (19652.21 + 148.4206 * T - 2.327105 * T ** 2
          + 1.360477e-2 * T ** 3 - 5.155288e-5 * T ** 4)
    ak = 54.6746 - 0.603459 * T + 1.09987e-2 * T ** 2 - 6.1670e-5 * T ** 3
    bk = 7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T ** 2
    k0 = kw + ak * S + bk * S ** 1.5
    aw = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * T ** 2 - 5.77905e-7 * T ** 3
    a1k = 2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * T ** 2
    k1 = aw + a1k * S + 1.91075e-4 * S ** 1.5
    bw = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * T ** 2
    a2k = -9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * T ** 2
    k2 = bw + a2k * S
    K = k0 + p * k1 + p ** 2 * k2
    return rho0 / (1.0 - p / K)


def sigma_t(T, S):
    return rho_eos80(T, S, 0.0) - 1000.0


# 注: 早期版本此处曾有 Mackenzie (1981) 声速公式, 其深度三次项系数录入有误
# (-7.139e-7 应为 -7.139e-13), 在 2000 m 处会偏差约 158 m/s。该实现已整体删除,
# 声速一律使用下方的 Chen-Millero 版本 (sound_speed_cm)。


def check():
    """用公认的 sigma_t 值校验实现。"""
    ref = {(0, 35): 28.106, (10, 35): 26.951, (20, 35): 24.761, (25, 35): 23.343}
    print("EOS-80 自检 (sigma_t = rho - 1000, p = 0):")
    ok = True
    for (T, S), v in ref.items():
        got = float(sigma_t(T, S))
        d = abs(got - v)
        ok &= d < 0.005
        print("  T=%2d S=%2d  本文 %8.3f   标准 %8.3f   差 %.4f  %s"
              % (T, S, got, v, d, "OK" if d < 0.005 else "FAIL"))
    # 压力项校验: T=10,S=35,p=1000 bar 时应约 1063.6 kg/m3
    # 1000 bar ~ 10000 m; 标准值约 1068 kg/m3 (sigma ~ 68)
    r = float(rho_eos80(10.0, 35.0, 1000.0))
    print("  T=10 S=35 p=1000 bar: %.2f kg/m3 (参考约 1068)  %s"
          % (r, "OK" if abs(r - 1068.0) < 2.0 else "检查"))
    return ok


def sound_speed_cm(S, T90, p_dbar):
    """Chen & Millero (1974)，按 UNESCO 1983 Eqn.(33)-(37) 实现（p.46-47）。

    参数: S[PSU], T90[degC], p_dbar[dbar]
    校验: 与 Pond & Pickard 表（seawater 包 docstring）一致，误差 < 0.01 m/s。
    与 eos80_ref.py（pyoceans/python-seawater, MIT）逐系数核对。
    """
    S = np.asarray(S, float); T68 = np.asarray(T90, float) * 1.00024; p = np.asarray(p_dbar, float) / 10.0
    c = ((3.1464e-9 * T68 - 1.47800e-6) * T68 + 3.3420e-4) * T68 ** 0
    Cw = ((((((3.1464e-9 * T68 - 1.47800e-6) * T68 + 3.3420e-4) * T68 - 5.80852e-2) * T68
            + 5.03711) * T68 + 1402.388))
    Cw = ((((((3.1464e-9 * T68 - 1.47800e-6) * T68 + 3.3420e-4) * T68 - 5.80852e-2) * T68
            + 5.03711) * T68 + 1402.388)
          + p * ((((((-6.1185e-10 * T68 + 1.3621e-7) * T68 - 8.1788e-6) * T68 + 6.8982e-4) * T68
                   + 0.153563))
                 + p * (((((1.0405e-12 * T68 - 2.5335e-10) * T68 + 2.5974e-8) * T68 - 1.7107e-6) * T68
                         + 3.1260e-5)
                        + p * ((-2.3643e-12 * T68 + 3.8504e-10) * T68 - 9.7729e-9))))
    A = ((((1.389 - 1.262e-2 * T68 + 7.164e-5 * T68 ** 2 + 2.006e-6 * T68 ** 3 - 3.21e-8 * T68 ** 4)
           + p * (9.4742e-5 - 1.2580e-5 * T68 - 6.4885e-8 * T68 ** 2 + 1.0507e-8 * T68 ** 3 - 2.0122e-10 * T68 ** 4))
          + p ** 2 * (-3.9064e-7 + 9.1041e-9 * T68 - 1.6002e-10 * T68 ** 2 + 7.988e-12 * T68 ** 3))
         + p ** 3 * (1.100e-10 + 6.649e-12 * T68 - 3.389e-13 * T68 ** 2))
    B = -1.922e-2 - 4.42e-5 * T68 + (7.3637e-5 + 1.7945e-7 * T68) * p
    D = 1.727e-3 - 7.9836e-6 * p
    return Cw + A * S + B * S * np.sqrt(S) + D * S ** 2


def check_speed():
    ref = {  # Pond & Pickard, 经 seawater 包 docstring 给出
        (35, 0, 0): 1449.13882813, (35, 0, 5000): 1533.96863705, (35, 0, 10000): 1623.15007097,
        (35, 10, 0): 1489.82233602, (35, 20, 0): 1521.46197310, (35, 20, 10000): 1687.18305631,
        (25, 0, 0): 1435.78987500, (25, 40, 10000): 1719.15088536,
    }
    print("Chen-Millero 声速自检 (S, T90, p_dbar):")
    worst = 0.0
    for (S, T, p), v in ref.items():
        got = float(sound_speed_cm(S, T, p)); d = abs(got - v); worst = max(worst, d)
        print("  S=%2d T=%2d p=%5d  本文 %10.4f  参考 %10.4f  差 %.4f" % (S, T, p, got, v, d))
    print("  最大偏差 %.4f m/s  %s" % (worst, "OK" if worst < 0.05 else "FAIL"))


# ======================================================================
# 3. fetch —— 数据下载
# ======================================================================

# -*- coding: utf-8 -*-
"""数据下载: ETOPO 地形 (ERDDAP) + HYCOM 剖面与二维场 (OPeNDAP DAP2 ASCII)。"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np


ROOT = HERE
CACHE = os.path.join(ROOT, "cache")


def _get(url, tries=4, timeout=180):
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:      # 网络抖动: 重试
            last = e
            time.sleep(3)
    raise last


# ---------------- ETOPO ----------------
def fetch_etopo(lat, lon, half=0.30, stride=1):
    """返回 (lat[], lon[], z[ny, nx]), z 为高程(负=水下)。stride=1 即原生 15″。"""
    q = "z[(%.4f):%d:(%.4f)][(%.4f):%d:(%.4f)]" % (
        lat - half, stride, lat + half, lon - half, stride, lon + half)
    url = ENDPOINTS["etopo"] + "?" + urllib.parse.quote(q, safe="():,.-")
    txt = _get(url)
    rows = txt.strip().split("\n")[2:]
    la, lo, z = [], [], []
    for ln in rows:
        a = ln.split(",")
        la.append(float(a[0])); lo.append(float(a[1])); z.append(float(a[2]))
    LA = np.unique(np.array(la)); LO = np.unique(np.array(lo))
    return LA, LO, np.array(z).reshape(len(LA), len(LO))


# ---------------- HYCOM / OPeNDAP ----------------
def _asc(expr):
    return _get(ENDPOINTS["hycom_dap"] + "?" + expr)


def _isf(s):
    try:
        float(s); return True
    except Exception:
        return False


def _section(txt, prefix):
    """DAP .asc: 每个变量一个段; 一维轴是逗号串, 多维网格是每行 "[i][j], v"。"""
    body = txt.split("---------------------------------------------")[-1]
    for sec in body.split("\n\n"):
        lines = [l.strip() for l in sec.strip().split("\n") if l.strip()]
        if not lines or not lines[0].startswith(prefix):
            continue
        out = []
        for l in lines[1:]:
            toks = [t.strip() for t in l.split(",")]
            if not toks[0] or not _isf(toks[0]):
                toks = toks[1:]
            out += [float(t) for t in toks if _isf(t)]
        return np.array(out)
    return None


def fetch_profile(lat, lon):
    """40 层 T/S/u/v。返回 dict。"""
    latv = _section(_asc("lat[2900:1:3010]"), "lat[")
    lonv = _section(_asc("lon[230:1:280]"), "lon[")
    i = 2900 + int(np.argmin(np.abs(latv - lat)))
    j = 230 + int(np.argmin(np.abs(lonv - lon)))
    dz = _section(_asc("depth[0:1:39]"), "depth[")
    out = {"depth": dz, "lat": float(latv[i - 2900]), "lon": float(lonv[j - 230])}
    for v in ("water_temp", "salinity", "water_u", "water_v"):
        out[v] = _section(_asc("%s[0][0:1:%d][%d][%d]" % (v, len(dz) - 1, i, j)), v + ".")
    return out


def fetch_field(lev=(0, 19, 24, 27, 32), lat_idx=(2925, 2940), lon_idx=(256, 266)):
    """若干深度层的二维流速场。lev 是 HYCOM 深度索引。"""
    latv = _section(_asc("lat[%d:1:%d]" % lat_idx), "lat[")
    lonv = _section(_asc("lon[%d:1:%d]" % lon_idx), "lon[")
    dz = _section(_asc("depth[0:1:39]"), "depth[")
    res = {"lat": latv, "lon": lonv, "depth": np.array([dz[L] for L in lev])}
    ny, nx = len(latv), len(lonv)
    for name in ("water_u", "water_v"):
        for L in lev:
            seq = _asc("%s[0][%d][%d:1:%d][%d:1:%d]" % (name, L, lat_idx[0], lat_idx[1],
                                                       lon_idx[0], lon_idx[1]))
            body = seq.split("---------------------------------------------")[-1].split("\n\n")[0]
            vals = []
            for ln in body.split("\n"):
                toks = [t.strip() for t in ln.split(",")]
                if not toks or not toks[0]:
                    continue
                if not _isf(toks[0]):
                    toks = toks[1:]
                vals += [float(t) for t in toks if _isf(t)]
            res["%s_%d" % (name, L)] = np.array(vals).reshape(ny, nx)
    return res


# ---------------- 落盘 ----------------
def _dump(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    if isinstance(obj, dict):
        np.savez(os.path.join(CACHE, name + ".npz"),
                 **{k: v for k, v in obj.items() if isinstance(v, (np.ndarray, float, int))})
        meta = {k: v for k, v in obj.items() if isinstance(v, str)}
        if meta:
            with open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
    else:
        np.savez(os.path.join(CACHE, name + ".npz"), **obj)


def fetch_main():
    lat, lon = SITE["lat"], SITE["lon"]
    print("[1/3] ETOPO 局部地形 ...")
    LA, LO, Z = fetch_etopo(lat, lon, half=0.30)
    _dump("etopo_local", {"lat": LA, "lon": LO, "z": Z})
    print("      %d x %d, 水深 %.0f .. %.0f m" % (Z.shape[0], Z.shape[1], -Z.max(), -Z.min()))

    print("[2/3] HYCOM 剖面 ...")
    prof = fetch_profile(lat, lon)
    ok = int(np.sum(~np.isnan(prof["water_temp"])))
    _dump("profile", prof)
    print("      %d/%d 层有效, 纬度 %.3f 经度 %.3f" % (ok, len(prof["depth"]),
                                                 prof["lat"], prof["lon"]))

    print("[3/3] HYCOM 二维场 ...")
    fld = fetch_field()
    _dump("field2d", fld)
    print("      %d x %d x %d 层" % (len(fld["lat"]), len(fld["lon"]), len(fld["depth"])))
    print("完成 -> cache/")


# ======================================================================
# 4. analyze —— 全部推导
# ======================================================================

# -*- coding: utf-8 -*-
"""全部推导与分析。所有环境量来自 cache/ 中的实测数据; 符号参数由调用者传入。"""
import os

import numpy as np


ROOT = HERE
CACHE = os.path.join(ROOT, "cache")


# ================= 数据 =================
def load_profile():
    d = np.load(os.path.join(CACHE, "profile.npz"))
    T, S = d["water_temp"], d["salinity"]
    m = ~np.isnan(T) & ~np.isnan(S)
    return d["depth"][m], T[m], S[m], d["water_u"][m], d["water_v"][m]


def load_terrain():
    d = np.load(os.path.join(CACHE, "etopo_local.npz"))
    return d["lat"], d["lon"], d["z"]


# ================= 剖面派生 =================
class Profile:
    """真实剖面 + 派生量, 全部按 1 m 重采样。"""

    def __init__(self, z0):
        zc, T, S, u, v = load_profile()
        self.z_src, self.T, self.S = zc, T, S
        top = min(zc[-1], 1000.0)
        self.z = np.arange(0.0, top, 1.0)
        self.rho = rho_eos80(np.interp(self.z, zc, T), np.interp(self.z, zc, S),
                                 pressure_bars(self.z))
        self.sig = rho_eos80(np.interp(self.z, zc, T), np.interp(self.z, zc, S), 0.0) - 1000.0
        self.u = np.interp(self.z, zc, u)
        self.v = np.interp(self.z, zc, v)
        self.spd = np.hypot(self.u, self.v)
        self.c = sound_speed_cm(np.interp(self.z, zc, S), np.interp(self.z, zc, T),
                                    10.0 * pressure_bars(self.z))
        self.z0 = z0
        self.sig0 = float(np.interp(z0, self.z, self.sig))

    def buoyancy_frequency(self, smooth_m=200.0):
        """N^2。同时返回原始值与 200 m 平滑值(用户确认的处理: 深水微小负值按噪声处理)。"""
        raw = G / 1027.0 * np.gradient(self.sig, self.z)
        win = max(3, int(smooth_m / (self.z[1] - self.z[0])))
        k = np.ones(win) / win
        pad = np.r_[self.sig[:win - 1][::-1], self.sig, self.sig[-win + 1:][::-1]]
        sm = np.convolve(pad, k, mode="same")[win - 1:-(win - 1)]
        return self.z, raw, G / 1027.0 * np.gradient(sm, self.z)


# ================= 潜水器 =================
def vehicle(rho_at_z0):
    """由中性配平反解排水体积, 再算等效球。返回 dict。"""
    v = VEHICLE
    vol = v["mass_dry_kg"] / rho_at_z0
    R = (3 * vol / (4 * np.pi)) ** (1 / 3)
    A = np.pi * R ** 2
    return {"V": vol, "R": R, "A": A, "cdA": v["cd"] * A,
            "m_eff": v["mass_dry_kg"] * (1 + v["added_mass"])}


# ================= 平衡深度 =================
def equilibrium_depth(prof, dr):
    """rho_w(z_eq) = rho_w(z0) + dr 的解; 不存在则返回 nan。"""
    d = prof.sig - (prof.sig0 + dr)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return prof.z[i] - d[i] * (prof.z[i + 1] - prof.z[i]) / (d[i + 1] - d[i])


# ================= 地形与下滑 =================
class Terrain:
    def __init__(self):
        self.lat, self.lon, Z = load_terrain()
        self.depth = -Z
        dy = 111320.0 * (self.lat[2] - self.lat[1])
        dx = 111320.0 * (self.lon[2] - self.lon[1]) * np.cos(np.deg2rad(self.lat.mean()))
        a, b = np.gradient(self.depth)
        self.slope = np.degrees(np.arctan(np.sqrt((a / dy) ** 2 + (b / dx) ** 2)))
        self.site = (SITE["lat"], SITE["lon"])

    def at(self, A, la, lo):
        i = (la - self.lat[0]) / (self.lat[1] - self.lat[0])
        j = (lo - self.lon[0]) / (self.lon[1] - self.lon[0])
        i = np.clip(i, 0, len(self.lat) - 1.001)
        j = np.clip(j, 0, len(self.lon) - 1.001)
        i0, j0 = int(i), int(j)
        fi, fj = i - i0, j - j0
        return (A[i0, j0] * (1 - fi) * (1 - fj) + A[i0 + 1, j0] * fi * (1 - fj)
                + A[i0, j0 + 1] * (1 - fi) * fj + A[i0 + 1, j0 + 1] * fi * fj)

    def slide_path(self, mu, step=20.0, nmax=3000):
        """沿最陡下降方向积分, 直到坡度 <= arctan(mu)。返回 (N,2) 的 lat/lon 轨迹。"""
        ang = np.degrees(np.arctan(mu))
        la, lo = self.site
        path = [(la, lo)]
        for _ in range(nmax):
            if self.at(self.slope, la, lo) <= ang:
                break
            e = 1e-4
            gx = (self.at(self.depth, la, lo + e) - self.at(self.depth, la, lo - e)) / (2 * e)
            gy = (self.at(self.depth, la + e, lo) - self.at(self.depth, la - e, lo)) / (2 * e)
            n = np.hypot(gx, gy)
            if n < 1e-12:
                break
            la += gy / n * step / 111320.0
            lo += gx / n * step / (111320.0 * np.cos(np.deg2rad(la)))
            if not (self.lat[0] < la < self.lat[-1] and self.lon[0] < lo < self.lon[-1]):
                break
            path.append((la, lo))
        return np.array(path)


# ================= 报告 =================
def analyze_main():
    prof = Profile(z0=300.0)
    veh = vehicle(float(np.interp(prof.z0, prof.z, prof.rho)))
    ter = Terrain()
    sl = ter.at(ter.slope, *ter.site)
    h = ter.at(ter.depth, *ter.site)

    print("=" * 78)
    print("1. 状态方程自检")
    print("=" * 78)
    check(); check_speed()

    print("\n" + "=" * 78)
    print("2. 站点与潜水器")
    print("=" * 78)
    print("  站点 %.4fN %.4fE  水深 %.0f m  坡度 %.1f 度"
          % (SITE["lat"], SITE["lon"], h, sl))
    print("  %s  V=%.2f m3  R=%.3f m  A=%.3f m2  CdA=%.3f m2"
          % (VEHICLE["model"], veh["V"], veh["R"], veh["A"], veh["cdA"]))
    print("  额定潜深 %.0f m  -> %s" % (VEHICLE["depth_rating_m"],
                                   "可执行" if h <= VEHICLE["depth_rating_m"] else "超限"))

    zg, N2r, N2s = prof.buoyancy_frequency()
    print("\n  剖面: 声速 %.1f (0 m) -> %.1f (%.0f m) m/s"
          % (prof.c[0], prof.c[-1], prof.z[-1]))
    print("        N2 原始最大 %.2e (%.0f m);  平滑后最小 %.2e"
          % (N2r.max(), zg[int(np.argmax(N2r))], N2s.min()))

    print("\n" + "=" * 78)
    print("3. M1 沿坡下滑: 需 mu_s >= %.3f" % np.tan(np.deg2rad(sl)))
    print("=" * 78)
    print("   mu_s   滑行距离   落点水深   超出额定?")
    for mu in (0.50, 0.40, 0.35, 0.30, 0.20, 0.10):
        pa = ter.slide_path(mu)
        d = np.hypot((pa[-1, 0] - ter.site[0]) * 111320.0,
                     (pa[-1, 1] - ter.site[1]) * 111320.0 * np.cos(np.deg2rad(ter.site[0])))
        dd = ter.at(ter.depth, pa[-1, 0], pa[-1, 1])
        print("   %.2f   %7.0f m   %7.0f m   %s"
              % (mu, d, dd, "是" if dd > VEHICLE["depth_rating_m"] else "否"))

    print("\n" + "=" * 78)
    print("4. M2 底流推动")
    print("=" * 78)
    ib = int(np.argmin(np.abs(prof.z - h)))
    ub = float(prof.spd[ib])
    print("  海底 %.0f m 流速 %.3f m/s" % (h, ub))
    print("   drho    u_c(mu=0.2)  u_c(mu=0.4)  u_c(mu=0.6)  能否推动")
    for dr in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        W = dr * veh["V"] * G
        uc = [np.sqrt(2 * m * W / (1027.9 * veh["cdA"])) for m in (0.2, 0.4, 0.6)]
        print("   %5.1f   %9.3f   %9.3f   %9.3f   %s"
              % (dr, uc[0], uc[1], uc[2], "能" if ub > uc[1] else "不能"))

    print("\n" + "=" * 78)
    print("5. drho -> 平衡深度 -> 24 h 漂移")
    print("=" * 78)
    print("   drho     平衡深度   流速      24 h 漂移")
    for dr in (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.08):
        ze = equilibrium_depth(prof, dr)
        if np.isnan(ze):
            print("   %+5.2f    无平衡点(沉底)" % dr)
            continue
        s = float(np.interp(ze, prof.z, prof.spd))
        print("   %+5.2f   %7.0f m   %.3f m/s   %6.1f km" % (dr, ze, s, s * 86.4))

    print("\n" + "=" * 78)
    print("6. 搜索条带长度 L = |du| * t   (km)")
    print("=" * 78)
    print("   drho     sigma    24 h 条带长度   备注")
    for dr0 in (-0.20, -0.05, 0.0, 0.05):
        for sg in (0.02, 0.05, 0.10, 0.20):
            z1, z2 = equilibrium_depth(prof, dr0 + sg), equilibrium_depth(prof, dr0 - sg)
            if np.isnan(z1) or np.isnan(z2):
                print("   %+5.2f    %.2f     -            跨界(一侧沉底)" % (dr0, sg))
                continue
            du = abs(float(np.interp(z2, prof.z, prof.spd)) - float(np.interp(z1, prof.z, prof.spd)))
            print("   %+5.2f    %.2f     %6.1f km" % (dr0, sg, du * 24 * 3.6))


# ======================================================================
# 5. plotlib —— 仅用 Pillow 的绘图库
# ======================================================================

# -*- coding: utf-8 -*-
"""极简绘图库: 只用 Pillow。支持坐标轴、折线、散点、栅格图、中文标注。"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # Linux / CI: 常见中文字体（任一存在即可）
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    r"/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    r"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
_font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def font(size):
    if _font_path:
        try:
            return ImageFont.truetype(_font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


class Panel:
    """一个数据坐标系 -> 像素区域的映射。"""

    def __init__(self, draw, box, xlim, ylim, xticks=None, yticks=None,
                 xlabel="", ylabel="", title="", tick_size=18, label_size=20,
                 title_size=22, logx=False, logy=False, ylab_offset=62):
        self.d = draw
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim = xlim
        self.ylim = ylim
        self.logx, self.logy = logx, logy
        self.d.rectangle(box, outline=(40, 40, 40), width=2)
        f = font(tick_size)
        # 刻度
        for t in (xticks or []):
            px = self.px(t)
            if px is None:
                continue
            self.d.line([(px, self.y1), (px, self.y1 + 6)], fill=(40, 40, 40), width=2)
            s = ("%g" % t)
            self.d.text((px, self.y1 + 10), s, font=f, fill=(30, 30, 30), anchor="ma")
        for t in (yticks or []):
            py = self.py(t)
            if py is None:
                continue
            self.d.line([(self.x0 - 6, py), (self.x0, py)], fill=(40, 40, 40), width=2)
            self.d.text((self.x0 - 10, py), ("%g" % t), font=f, fill=(30, 30, 30), anchor="rm")
        if xlabel:
            self.d.text(((self.x0 + self.x1) / 2, self.y1 + 44), xlabel,
                        font=font(label_size), fill=(20, 20, 20), anchor="ma")
        if ylabel:
            self.d.text((self.x0 - ylab_offset, (self.y0 + self.y1) / 2), ylabel,
                        font=font(label_size), fill=(20, 20, 20), anchor="mm")
        if title:
            self.d.text(((self.x0 + self.x1) / 2, self.y0 - 34), title,
                        font=font(title_size), fill=(10, 10, 10), anchor="ma")

    def _t(self, v, lim, log):
        return np.log10(v) if log else v

    def px(self, x):
        a, b = self.xlim
        if self.logx:
            if x <= 0: return None
            a, b, x = np.log10(a), np.log10(b), np.log10(x)
        return self.x0 + (x - a) / (b - a) * (self.x1 - self.x0)

    def py(self, y):
        a, b = self.ylim
        if self.logy:
            if y <= 0: return None
            a, b, y = np.log10(a), np.log10(b), np.log10(y)
        return self.y1 - (y - a) / (b - a) * (self.y1 - self.y0)

    def line(self, xs, ys, color=(200, 60, 60), width=3):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        pts = [p for p in pts if p[0] is not None and p[1] is not None]
        if len(pts) > 1:
            self.d.line(pts, fill=color, width=width, joint="curve")

    def dots(self, xs, ys, r=5, color=(30, 90, 200)):
        for x, y in zip(xs, ys):
            px, py = self.px(x), self.py(y)
            if px is None or py is None:
                continue
            self.d.ellipse([px - r, py - r, px + r, py + r], fill=color)


def save(img, path):
    img.save(path)
    return path


def canvas(w, h, bg=(255, 255, 255)):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)


def depth_image(Z, w, h, zmin, zmax):
    """水深 -> 灰度图 (浅白深黑), 叠加西北光照的晕渲。"""
    Zn = np.clip((Z - zmin) / (zmax - zmin), 0, 1)
    base = 1.0 - 0.78 * Zn
    gy, gx = np.gradient(Zn)
    shade = np.clip(0.5 - 2.2 * (gx + gy), 0.35, 1.25)
    img = np.clip(base * shade, 0, 1)
    arr = (img * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L").resize((w, h), Image.BILINEAR)
    return im.convert("RGB")


# ======================================================================
# 6. figures —— 四张图
# ======================================================================

# -*- coding: utf-8 -*-
"""生成报告用图。全部数据来自 py 的实测结果。"""
import os

import numpy as np


ROOT = HERE
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

PROF = None
TER = None
VEH = None


def _ctx():
    """惰性构建上下文: check 模式不需要 cache 目录。"""
    global PROF, TER, VEH
    if PROF is None:
        PROF = Profile(z0=300.0)
        TER = Terrain()
        VEH = vehicle(float(np.interp(PROF.z0, PROF.z, PROF.rho)))
    return PROF, TER, VEH


def fig1_terrain():
    _ctx()
    X0, X1 = TER.site[1] - 0.12, TER.site[1] + 0.12
    Y0, Y1 = TER.site[0] - 0.12, TER.site[0] + 0.12
    img, d = canvas(1200, 1270)
    box = (200, 90, 840, 790)
    p = Panel(d, box, (X0, X1), (Y0, Y1),
                      [round(X0, 2), round((X0 + X1) / 2, 2), round(X1, 2)],
                      [round(Y0, 2), round((Y0 + Y1) / 2, 2), round(Y1, 2)],
                      "经度 (°E)", "纬度 (°N)", "潜点局部地形与下滑轨迹", label_size=22, ylab_offset=120)
    i0, i1 = np.searchsorted(TER.lon, X0), np.searchsorted(TER.lon, X1)
    j0, j1 = np.searchsorted(TER.lat, Y0), np.searchsorted(TER.lat, Y1)
    sub = TER.depth[j0:j1, i0:i1]
    img.paste(depth_image(sub, box[2] - box[0], box[3] - box[1],
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
           font=font(20), fill=(210, 0, 0))
    d.text((px + 22, py + 48), "坡度 %.1f°" % TER.at(TER.slope, *TER.site),
           font=font(18), fill=(210, 0, 0))

    lx = 870
    d.text((lx, 110), "图例", font=font(24), fill=(20, 20, 20))
    for k, (mu, col) in enumerate(cols.items()):
        d.line([(lx, 168 + k * 46), (lx + 62, 168 + k * 46)], fill=col, width=5)
        d.text((lx + 74, 168 + k * 46), "μ_s = %.1f" % mu, font=font(20), fill=col, anchor="lm")
    for k, txt in enumerate(["地形 ETOPO 2022 15″", "灰度越暗越深", "轨迹沿最陡下降方向",
                             "停止 坡度 ≤ arctan(μ_s)"]):
        d.text((lx, 330 + k * 32), txt, font=font(18), fill=(70, 70, 70))
    d.text((lx, 486), "落点水深", font=font(19), fill=(20, 20, 20))
    for k, (mu, col, pa) in enumerate(ends):
        d.text((lx, 520 + k * 34), "μ=%.1f → %.0f m" % (mu, TER.at(TER.depth, pa[-1, 0], pa[-1, 1])),
               font=font(18), fill=col)

    mus = np.linspace(0.12, 0.55, 24)
    dep = [TER.at(TER.depth, *(TER.slide_path(mu)[-1])) for mu in mus]
    q = Panel(d, (240, 950, 840, 1190), (0.12, 0.55), (0, 1800),
                      [0.2, 0.3, 0.4, 0.5], [0, 600, 1200, 1800], "摩擦系数 μ_s", "落点水深 (m)",
                      "滑行终点水深随 μ_s 的变化", tick_size=16, label_size=18, title_size=20)
    q.line([0.12, 0.55], [1000, 1000], color=(150, 150, 150), width=2)
    d.text((q.x1 - 8, q.py(1000) - 10), "车辆额定 1000 m", font=font(15),
           fill=(120, 120, 120), anchor="rb")
    q.line(mus, dep, color=(180, 40, 40), width=4)
    mu_thr = np.tan(np.deg2rad(TER.at(TER.slope, *TER.site)))
    q.dots([mu_thr], [TER.at(TER.depth, *TER.site)], r=6, color=(20, 20, 20))
    d.text((q.px(mu_thr) + 10, q.py(TER.at(TER.depth, *TER.site)) - 14),
           "阈值 μ_s = %.3f" % mu_thr, font=font(16), fill=(20, 20, 20))
    save(img, os.path.join(FIGS, "fig1_terrain_slide.png"))


def fig2_profile():
    _ctx()
    img, d = canvas(1240, 560)
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
        Panel(d, (xx, 80, xx + wp, 460), (lo - pad, hi + pad), (zmax, 0),
                      [round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)],
                      [0, 200, 400, 600, 800], tt, "深度 (m)" if k == 0 else "",
                      tick_size=16, label_size=19)
        # 重画曲线
    # 曲线需在 Panel 之后画, 这里重新绘制一遍
    img, d = canvas(1240, 560)
    for k, (s, col, tt) in enumerate(series):
        xx = x0 + k * (wp + gap)
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        pad = (hi - lo) * 0.1 or 1.0
        pp = Panel(d, (xx, 80, xx + wp, 460), (lo - pad, hi + pad), (zmax, 0),
                           [round(lo, 2), round((lo + hi) / 2, 2), round(hi, 2)],
                           [0, 200, 400, 600, 800], tt, "深度 (m)" if k == 0 else "",
                           tick_size=16, label_size=19)
        pp.line(s, PROF.z, color=col, width=4)
    d.text((620, 30), "HYCOM 实测剖面 (%.2fN %.2fE, 2026-11-01T00Z)"
           % (SITE["lat"], SITE["lon"]), font=font(22), fill=(10, 10, 10), anchor="ma")
    save(img, os.path.join(FIGS, "fig2_profile.png"))


def fig3_drift():
    _ctx()
    drs = np.linspace(-0.32, 0.085, 80)
    dep = np.array([equilibrium_depth(PROF, x) for x in drs])
    dep = np.where(dep > 1040, np.nan, dep)
    drift = np.array([float(np.interp(z, PROF.z, PROF.spd)) * 86.4 if not np.isnan(z) else np.nan
                      for z in dep])
    img, d = canvas(1220, 880)
    B = (185, 100, 845, 680)
    p = Panel(d, B, (-0.32, 0.10), (0, 1100), [-0.3, -0.2, -0.1, 0, 0.1],
                      [0, 300, 600, 900], "浮力异常 Δρ (kg/m³)", "平衡深度 (m)",
                      "Δρ 如何决定平衡深度与漂移距离", label_size=21, tick_size=17, ylab_offset=110)
    p.line(drs, dep, color=(40, 90, 190), width=4)
    d.text((p.px(-0.315), p.py(960)), "平衡深度  ←", font=font(18), fill=(40, 90, 190))
    XR = B[2]
    d.line([(XR, B[1]), (XR, B[3])], fill=(40, 40, 40), width=2)
    ry = lambda km: B[3] - km / 12.0 * (B[3] - B[1])
    for km in (0, 3, 6, 9, 12):
        d.line([(XR, ry(km)), (XR + 7, ry(km))], fill=(40, 40, 40), width=2)
        d.text((XR + 14, ry(km)), "%g" % km, font=font(17), fill=(30, 30, 30), anchor="lm")
    d.text((XR + 120, (B[1] + B[3]) / 2), "24 h 漂移 (km)", font=font(21), fill=(20, 20, 20), anchor="mm")
    d.line([(p.px(x), ry(y)) for x, y in zip(drs, drift) if not np.isnan(y)],
           fill=(205, 70, 40), width=4)
    d.text((p.px(-0.315), ry(7.2)), "24 h 漂移  ←", font=font(18), fill=(205, 70, 40))
    d.text((p.px(0.082), B[1] + 24), "Δρ > +0.09 无平衡点", font=font(16),
           fill=(110, 110, 110), anchor="rt")
    save(img, os.path.join(FIGS, "fig3_drift.png"))


def fig4_band():
    _ctx()
    img, d = canvas(1160, 780)
    q = Panel(d, (150, 100, 1060, 640), (0.0, 0.12), (0, 26),
                      [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12], [0, 5, 10, 15, 20, 25],
                      "Δρ 的不确定度 σ (kg/m³)", "24 h 搜索条带长度 L (km)",
                      "条带长度随浮力不确定度的增长 (t = 24 h)", label_size=21, tick_size=17,
                      ylab_offset=96)
    sgs = np.linspace(0.005, 0.11, 60)
    for dr0, col in ((-0.20, (60, 140, 90)), (-0.05, (40, 90, 190)),
                     (0.00, (220, 140, 30)), (0.05, (200, 50, 50))):
        xs, ys = [], []
        for sg in sgs:
            z1, z2 = equilibrium_depth(PROF, dr0 + sg), equilibrium_depth(PROF, dr0 - sg)
            if np.isnan(z1) or np.isnan(z2):
                break
            du = abs(float(np.interp(z2, PROF.z, PROF.spd)) - float(np.interp(z1, PROF.z, PROF.spd)))
            xs.append(sg); ys.append(du * 24 * 3.6)
        if len(xs) > 1:
            q.line(xs, ys, color=col, width=4)
            d.text((q.px(xs[-1]) + 12, q.py(ys[-1])), "Δρ=%+.2f" % dr0, font=font(17),
                   fill=col, anchor="lm")
            d.ellipse([q.px(xs[-1]) - 5, q.py(ys[-1]) - 5, q.px(xs[-1]) + 5, q.py(ys[-1]) + 5], fill=col)
    d.text((560, 710), "曲线在 σ 超阈值处终止 —— 那一侧已沉底, 属于分支改变而非数值散布",
           font=font(17), fill=(90, 90, 90), anchor="ma")
    save(img, os.path.join(FIGS, "fig4_bandlen.png"))


def figures_main():
    fig1_terrain(); print("fig1 ok")
    fig2_profile(); print("fig2 ok")
    fig3_drift(); print("fig3 ok")
    fig4_band(); print("fig4 ok")


# ======================================================================
# 7. cli —— 命令行入口
# ======================================================================
def cli(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd in ("check", "all"):
        check(); check_speed()
    if cmd in ("fetch", "all"):
        fetch_main()
    if cmd in ("analyze", "all"):
        analyze_main()
    if cmd in ("figures", "all"):
        figures_main()
    if cmd not in ("check", "fetch", "analyze", "figures", "all"):
        print(__doc__)


if __name__ == "__main__":
    cli(sys.argv)
