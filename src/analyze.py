# -*- coding: utf-8 -*-
"""全部推导与分析。所有环境量来自 cache/ 中的实测数据; 符号参数由调用者传入。"""
import os

import numpy as np

from . import config, eos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        self.rho = eos.rho_eos80(np.interp(self.z, zc, T), np.interp(self.z, zc, S),
                                 eos.pressure_bars(self.z))
        self.sig = eos.rho_eos80(np.interp(self.z, zc, T), np.interp(self.z, zc, S), 0.0) - 1000.0
        self.u = np.interp(self.z, zc, u)
        self.v = np.interp(self.z, zc, v)
        self.spd = np.hypot(self.u, self.v)
        self.c = eos.sound_speed_cm(np.interp(self.z, zc, S), np.interp(self.z, zc, T),
                                    10.0 * eos.pressure_bars(self.z))
        self.z0 = z0
        self.sig0 = float(np.interp(z0, self.z, self.sig))

    def buoyancy_frequency(self, smooth_m=200.0):
        """N^2。同时返回原始值与 200 m 平滑值(用户确认的处理: 深水微小负值按噪声处理)。"""
        raw = config.G / 1027.0 * np.gradient(self.sig, self.z)
        win = max(3, int(smooth_m / (self.z[1] - self.z[0])))
        k = np.ones(win) / win
        pad = np.r_[self.sig[:win - 1][::-1], self.sig, self.sig[-win + 1:][::-1]]
        sm = np.convolve(pad, k, mode="same")[win - 1:-(win - 1)]
        return self.z, raw, config.G / 1027.0 * np.gradient(sm, self.z)


# ================= 潜水器 =================
def vehicle(rho_at_z0):
    """由中性配平反解排水体积, 再算等效球。返回 dict。"""
    v = config.VEHICLE
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
        self.site = (config.SITE["lat"], config.SITE["lon"])

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
def main():
    prof = Profile(z0=300.0)
    veh = vehicle(float(np.interp(prof.z0, prof.z, prof.rho)))
    ter = Terrain()
    sl = ter.at(ter.slope, *ter.site)
    h = ter.at(ter.depth, *ter.site)

    print("=" * 78)
    print("1. 状态方程自检")
    print("=" * 78)
    eos.check(); eos.check_speed()

    print("\n" + "=" * 78)
    print("2. 站点与潜水器")
    print("=" * 78)
    print("  站点 %.4fN %.4fE  水深 %.0f m  坡度 %.1f 度"
          % (config.SITE["lat"], config.SITE["lon"], h, sl))
    print("  %s  V=%.2f m3  R=%.3f m  A=%.3f m2  CdA=%.3f m2"
          % (config.VEHICLE["model"], veh["V"], veh["R"], veh["A"], veh["cdA"]))
    print("  额定潜深 %.0f m  -> %s" % (config.VEHICLE["depth_rating_m"],
                                   "可执行" if h <= config.VEHICLE["depth_rating_m"] else "超限"))

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
              % (mu, d, dd, "是" if dd > config.VEHICLE["depth_rating_m"] else "否"))

    print("\n" + "=" * 78)
    print("4. M2 底流推动")
    print("=" * 78)
    ib = int(np.argmin(np.abs(prof.z - h)))
    ub = float(prof.spd[ib])
    print("  海底 %.0f m 流速 %.3f m/s" % (h, ub))
    print("   drho    u_c(mu=0.2)  u_c(mu=0.4)  u_c(mu=0.6)  能否推动")
    for dr in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        W = dr * veh["V"] * config.G
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


if __name__ == "__main__":
    main()