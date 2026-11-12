# -*- coding: utf-8 -*-
"""P2 数据层: 现场密度剖面、向量化地形、二维流场。

与 v2 的 analyze.py 并存:
  - analyze.py 保留作"局部机理校核"(滑移阈值 / 底流阈值 / 平衡深度表);
  - 本模块提供 P2 集合预报需要的一维/二维**向量化**访问接口。

三条关键修正 (见 docs/建模方案_v3.md §0.1):
  1. 浮力用**现场密度** rho_in_situ, 不用位势密度 sigma_t;
  2. 地形坡度/梯度统一换算到 m/m, 并在平滑后的深度场上取方向(避免 ETOPO 噪声锯齿);
  3. HYCOM 二维场 500/1000 m 层为 NaN, 深于 300 m 一律用 300 m 层截断(报告中声明)。
"""
import os

import numpy as np

from . import config, eos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")

DEG_LAT_M = 111320.0


def _npz(name):
    """按当前海域 (config.SITE_TAG) 读取数据缓存; 主算例无后缀。"""
    tag = getattr(config, "SITE_TAG", None)
    fn = name + (("_" + tag) if tag else "")
    return np.load(os.path.join(CACHE, fn + ".npz"))


# ============================================================================
# 一维剖面
# ============================================================================
class Profile:
    """HYCOM 站点剖面 → 1 m 均匀网格的现场密度、位势密度、流速、声速。"""

    def __init__(self):
        d = _npz("profile")
        m = ~np.isnan(d["water_temp"])
        zc = d["depth"][m]
        T, S = d["water_temp"][m], d["salinity"][m]
        self.z_src = zc
        TOP = min(float(zc[-1]), 1000.0)
        self.z = np.arange(0.0, TOP, 1.0)
        T = np.interp(self.z, zc, T)
        S = np.interp(self.z, zc, S)
        p = eos.pressure_bars(self.z)
        self.T, self.S = T, S
        self.rho = eos.rho_eos80(T, S, p)            # 现场密度 (浮力用这个)
        self.sig = eos.rho_eos80(T, S, 0.0) - 1000.0  # 位势密度 (仅层结讨论用)
        self.u = np.interp(self.z, zc, d["water_u"][m])
        self.v = np.interp(self.z, zc, d["water_v"][m])
        self.spd = np.hypot(self.u, self.v)
        self.c = eos.sound_speed_cm(S, T, 10.0 * p)

    # --- 查询 (标量或数组均可, 超出深度范围时取端点) ---
    def rho_at(self, z):
        return np.interp(z, self.z, self.rho)

    def sig_at(self, z):
        return np.interp(z, self.z, self.sig)

    def uv_at(self, z):
        return np.interp(z, self.z, self.u), np.interp(z, self.z, self.v)

    def spd_at(self, z):
        return np.interp(z, self.z, self.spd)

    def n2(self, smooth_m=200.0):
        """浮力频率平方; 返回 (原始, 平滑)。"""
        g = config.G
        raw = g / 1027.0 * np.gradient(self.sig, self.z)
        win = max(3, int(smooth_m / (self.z[1] - self.z[0])))
        if win >= len(self.sig):
            return raw, raw
        k = np.ones(win) / win
        pad = np.r_[self.sig[:win - 1][::-1], self.sig, self.sig[-win + 1:][::-1]]
        sm = np.convolve(pad, k, mode="same")[win - 1:-(win - 1)]
        return raw, g / 1027.0 * np.gradient(sm, self.z)


# ============================================================================
# 地形
# ============================================================================
class Terrain:
    """ETOPO 局部栅格; 提供向量化双线性查询与坡度/梯度。"""

    def __init__(self, smooth_cells=1):
        d = _npz("etopo_local")
        self.lat = d["lat"].astype(float)
        self.lon = d["lon"].astype(float)
        self.depth = -d["z"].astype(float)                 # 水深 (向下为正)
        self.lat0 = float(self.lat[0])
        self.lon0 = float(self.lon[0])
        self.dlat = float(self.lat[1] - self.lat[0])
        self.dlon = float(self.lon[1] - self.lon[0])
        self.ny, self.nx = self.depth.shape
        self.dy_m = DEG_LAT_M * self.dlat
        self.dx_m = DEG_LAT_M * np.cos(np.deg2rad(self.lat.mean())) * self.dlon
        self.lat_mean = float(self.lat.mean())

        # 平滑深度 (只用于梯度/坡度, 抑制 15" 栅格噪声): 0.5*自身 + 0.125*四邻
        dep = self.depth
        for _ in range(max(0, int(smooth_cells))):
            pad = np.pad(dep, 1, mode="edge")
            dep = 0.5 * dep + 0.125 * (pad[:-2, 1:-1] + pad[2:, 1:-1]
                                       + pad[1:-1, :-2] + pad[1:-1, 2:])
        self.depth_s = dep

        # 梯度 (m/m): g_n 向北, g_e 向东; 指向水深增大方向(即下坡方向取负)
        self.g_n = np.gradient(dep, axis=0) / self.dy_m
        self.g_e = np.gradient(dep, axis=1) / self.dx_m
        self.slope_deg = np.degrees(np.arctan(np.hypot(self.g_n, self.g_e)))

    # --- 双线性插值 ---
    def _idx(self, la, lo):
        la = np.asarray(la, float)
        lo = np.asarray(lo, float)
        fi = (la - self.lat0) / self.dlat
        fj = (lo - self.lon0) / self.dlon
        fi = np.clip(fi, 0.0, self.ny - 1.001)
        fj = np.clip(fj, 0.0, self.nx - 1.001)
        i0 = fi.astype(np.int64)
        j0 = fj.astype(np.int64)
        return i0, j0, fi - i0, fj - j0

    def at(self, A, la, lo):
        A = np.asarray(A, float)
        i0, j0, di, dj = self._idx(la, lo)
        return (A[i0, j0] * (1 - di) * (1 - dj) + A[i0 + 1, j0] * di * (1 - dj)
                + A[i0, j0 + 1] * (1 - di) * dj + A[i0 + 1, j0 + 1] * di * dj)

    def depth_at(self, la, lo):
        """返回真实水深 (未平滑, 用于触底判定与额度潜深比较)。"""
        return self.at(self.depth, la, lo)

    def slope_at(self, la, lo):
        return self.at(self.slope_deg, la, lo)

    def grad_at(self, la, lo):
        """返回 (g_e, g_n) 单位 m/m, 指向水深增大方向 (下坡)。"""
        return self.at(self.g_e, la, lo), self.at(self.g_n, la, lo)

    def inside(self, la, lo):
        return (self.lat0 <= la) & (la <= self.lat[-1]) & (self.lon0 <= lo) & (lo <= self.lon[-1])

    # --- 辅助: 局部 xy(km) <-> 经纬度 ---
    def latlon_of(self, x_m, y_m):
        la = config.SITE["lat"] + np.asarray(y_m, float) / DEG_LAT_M
        lo = config.SITE["lon"] + np.asarray(x_m, float) / (
            DEG_LAT_M * np.cos(np.deg2rad(config.SITE["lat"])))
        return la, lo

    def xy_of(self, la, lo):
        x = (np.asarray(lo, float) - config.SITE["lon"]) * DEG_LAT_M * np.cos(
            np.deg2rad(config.SITE["lat"]))
        y = (np.asarray(la, float) - config.SITE["lat"]) * DEG_LAT_M
        return x, y


# ============================================================================
# 二维流场
# ============================================================================
def _fill_nan(A, n_iter=20):
    """用四邻域迭代均值填补二维场的 NaN (HYCOM 陆地点)。"""
    import warnings
    B = np.array(A, float)
    if not np.isnan(B).any():
        return B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(int(n_iter)):
            m = np.isnan(B)
            if not m.any():
                break
            pad = np.pad(B, 1, mode="constant", constant_values=np.nan)
            nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
            fill = np.nanmean(nb, axis=0)
            B = np.where(m & np.isfinite(fill), fill, B)
        if np.isnan(B).any():
            B = np.where(np.isnan(B), np.nanmean(B), B)
    return B


class CurrentField:
    """HYCOM 多层二维流场; 空间双线性 + 深度线性插值, 深于最深层用最深层截断。

    v2 的文档说 500/1000 m 层"全为 NaN", 实测只有 6/32 个格点为 NaN, 因此 5 层全用,
    先做邻域迭代补洞。
    """

    def __init__(self):
        d = _npz("field2d")
        self.lat = d["lat"].astype(float)
        self.lon = d["lon"].astype(float)
        zlev = d["depth"].astype(float)
        self.z, us, vs = [], [], []
        # fetch.fetch_field 的默认 lev; depth[] 的第 k 个元素对应 lev[k]
        lev = config.ENDPOINTS.get("hycom_lev", (0, 19, 24, 27, 32))
        for k, L in enumerate(lev):
            ku, kv = "water_u_%d" % L, "water_v_%d" % L
            if ku not in d.files:
                continue
            u, v = d[ku], d[kv]
            if np.all(np.isnan(u)) or np.all(np.isnan(v)):
                continue
            self.z.append(float(zlev[k]))
            us.append(_fill_nan(u))
            vs.append(_fill_nan(v))
        self.z = np.array(self.z)
        self.u = np.array(us)     # (nlev, nlat, nlon)
        self.v = np.array(vs)
        self.lat0, self.lon0 = float(self.lat[0]), float(self.lon[0])
        self.dlat = float(self.lat[1] - self.lat[0])
        self.dlon = float(self.lon[1] - self.lon[0])

    def _bilin(self, A, la, lo):
        ny, nx = A.shape
        la = np.atleast_1d(np.asarray(la, float))
        lo = np.atleast_1d(np.asarray(lo, float))
        fi = np.clip((la - self.lat0) / self.dlat, 0.0, ny - 1.001)
        fj = np.clip((lo - self.lon0) / self.dlon, 0.0, nx - 1.001)
        i0, j0 = fi.astype(np.int64), fj.astype(np.int64)
        di, dj = fi - i0, fj - j0
        return (A[i0, j0] * (1 - di) * (1 - dj) + A[i0 + 1, j0] * di * (1 - dj)
                + A[i0, j0 + 1] * (1 - di) * dj + A[i0 + 1, j0 + 1] * di * dj)

    def uv(self, la, lo, z):
        """返回 (u, v), 单位 m/s。标量/数组混用会自动广播。"""
        la, lo, z = np.broadcast_arrays(np.asarray(la, float), np.asarray(lo, float),
                                        np.asarray(z, float))
        z = z.ravel(); la = la.ravel(); lo = lo.ravel()
        nlev = len(self.z)
        if nlev == 1:
            return self._bilin(self.u[0], la, lo), self._bilin(self.v[0], la, lo)
        # 各层双线性
        uu = np.stack([self._bilin(self.u[k], la, lo) for k in range(nlev)])
        vv = np.stack([self._bilin(self.v[k], la, lo) for k in range(nlev)])
        # 深度插值
        k1 = np.searchsorted(self.z, z, side="right") - 1
        k1 = np.clip(k1, 0, nlev - 2)
        k2 = k1 + 1
        w = (z - self.z[k1]) / (self.z[k2] - self.z[k1])
        w = np.clip(w, 0.0, 1.0)
        idx = np.arange(z.size)
        u = uu[k1, idx] * (1 - w) + uu[k2, idx] * w
        v = vv[k1, idx] * (1 - w) + vv[k2, idx] * w
        return u, v

    def uv_t(self, la, lo, z, t_h):
        """与 CurrentField4D 统一接口 (二维场不含时间, t_h 被忽略)。"""
        return self.uv(la, lo, z)


# ============================================================================
# 四维流场 (时间 × 层 × 纬 × 经)
# ============================================================================
class CurrentField4D:
    """HYCOM 四维流速场: 空间双线性 + 深度线性 + 时间线性插值。

    数据: cache/hycom4d.npz
      time_h[nt]          时刻 (小时, 相对分析时刻)
      depth[nlev]         层深度 (m)
      lat[ny], lon[nx]    格点
      water_u / water_v   形状 (nt, nlev, ny, nx), m/s

    查询: uv_t(lat, lon, z, t_h) -> (u, v), 支持标量/数组广播。
    """

    def __init__(self, path=None):
        d = _npz("hycom4d") if path is None else np.load(path)
        self.lat = d["lat"].astype(float)
        self.lon = d["lon"].astype(float)
        self.z = d["depth"].astype(float)
        self.time_h = d["time_h"].astype(float)
        self.lat0, self.lon0 = float(self.lat[0]), float(self.lon[0])
        self.dlat = float(self.lat[1] - self.lat[0])
        self.dlon = float(self.lon[1] - self.lon[0])
        u = d["water_u"].astype(float)
        v = d["water_v"].astype(float)
        nt, nlev, ny, nx = u.shape
        for t in range(nt):
            for k in range(nlev):
                if np.isnan(u[t, k]).any():
                    u[t, k] = _fill_nan(u[t, k])
                if np.isnan(v[t, k]).any():
                    v[t, k] = _fill_nan(v[t, k])
        self.u, self.v = u, v
        self.nt, self.nlev, self.ny, self.nx = u.shape
        self.z_max = float(self.z[-1])

    def _field_at(self, t_h):
        """时间线性插值, 返回该时刻的 (nlev, ny, nx) 两个分量。"""
        th = float(np.clip(t_h, self.time_h[0], self.time_h[-1]))
        i1 = int(np.searchsorted(self.time_h, th, side="right")) - 1
        i1 = min(max(i1, 0), self.nt - 2)
        i2 = i1 + 1
        dt = self.time_h[i2] - self.time_h[i1]
        w = 0.0 if dt <= 0 else float(np.clip((th - self.time_h[i1]) / dt, 0.0, 1.0))
        if w == 0.0:
            return self.u[i1], self.v[i1]
        return ((1 - w) * self.u[i1] + w * self.u[i2],
                (1 - w) * self.v[i1] + w * self.v[i2])

    def uv_t(self, la, lo, z, t_h):
        la, lo, z = np.broadcast_arrays(np.asarray(la, float), np.asarray(lo, float),
                                        np.asarray(z, float))
        la = la.ravel(); lo = lo.ravel(); z = z.ravel()
        ut, vt = self._field_at(t_h)
        fi = np.clip((la - self.lat0) / self.dlat, 0.0, self.ny - 1.001)
        fj = np.clip((lo - self.lon0) / self.dlon, 0.0, self.nx - 1.001)
        i0 = fi.astype(np.int64)
        j0 = fj.astype(np.int64)
        di, dj = fi - i0, fj - j0
        zc = np.clip(z, 0.0, self.z_max)
        k1 = np.searchsorted(self.z, zc, side="right") - 1
        k1 = np.clip(k1, 0, self.nlev - 2)
        k2 = k1 + 1
        dz = self.z[k2] - self.z[k1]
        wz = np.clip((zc - self.z[k1]) / np.where(dz > 0, dz, 1.0), 0.0, 1.0)
        base = self.ny * self.nx
        idx = k1 * base + i0 * self.nx + j0
        out = []
        for A in (ut, vt):
            f = A.ravel()
            w000 = (1 - wz) * (1 - di) * (1 - dj)
            w100 = wz * (1 - di) * (1 - dj)
            w010 = (1 - wz) * di * (1 - dj)
            w110 = wz * di * (1 - dj)
            w001 = (1 - wz) * (1 - di) * dj
            w101 = wz * (1 - di) * dj
            w011 = (1 - wz) * di * dj
            w111 = wz * di * dj
            out.append(f[idx] * w000 + f[idx + base] * w100
                       + f[idx + self.nx] * w010 + f[idx + base + self.nx] * w110
                       + f[idx + 1] * w001 + f[idx + base + 1] * w101
                       + f[idx + self.nx + 1] * w011 + f[idx + base + self.nx + 1] * w111)
        return out[0], out[1]


def make_current(prefer_4d=True):
    """优先用四维场; 没有缓存时退回二维单时刻场。"""
    if prefer_4d:
        try:
            return CurrentField4D()
        except Exception as e:      # noqa: BLE001
            print("  [提示] 四维流场不可用 (%s), 退回二维单时刻流场" % type(e).__name__)
    return CurrentField()
