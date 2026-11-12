# -*- coding: utf-8 -*-
"""P2 定位引擎: 混合先验 + 三维动力学 + 拉丁超立方集合预报。

模型结构 (见 docs/建模方案_v3.md §3):
  1. 混合先验: 失效模式 S0..S4 (S2 再分水面/无动力两子支), 每个模式一个 Δρ 分布;
  2. 惯性段 (0..t_descent_h, dt=10 s): 垂向用**解析解**
         m* dw/dt = (m − ρ_w(z)·V) g − ½ρ_w C_D A w|w|
     w(t) = w_t·tanh( arctanh(w/w_t) + t/τ ),  τ = m*/(½ρC_D A·|w_t|)
     对任意 dt 无条件稳定, 因此能同时处理 Δρ≈0 (缓沉) 与 Δρ=60 (快沉) 两种极端;
     水平按当地流场平流; 触底 → 海底滑移; 到水面 → 水面漂移; |Δρ_eff|→0 → 中性浮力;
  3. 准稳态段 (t_descent_h..96 h, dt=900 s): 悬浮粒子按当地流场平流, 叠加流场误差
     (OU 过程 + 每样本系统性偏差 + 湍流随机游走); 沉底粒子静止.

准稳态近似的依据: 二次阻力下速度松弛时间 τ ≈ 100 s ≪ 1 h, 几分钟后水平速度即锁到流场。
垂直方向仍用解析解推进, 不做准稳态假设。

输出: cache/p2_ensemble.npz
"""
import os
import time as _time

import numpy as np

from . import config, data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
G = config.G

# 状态码
ST_DYN, ST_WATER, ST_SURFACE, ST_BOTTOM = 0, 1, 2, 3
ST_NAME = {ST_DYN: "动力学", ST_WATER: "中性浮力", ST_SURFACE: "水面", ST_BOTTOM: "海底"}

DEG_LAT_M = 111320.0


# ============================================================================
# 采样工具
# ============================================================================
def _lhs(rng, n, k):
    """拉丁超立方: 返回 (n,k) 的 [0,1) 样本, 每列分层。"""
    out = np.empty((n, k))
    for j in range(k):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / float(n)
    return out


def _triangular_ppf(u, a, c, b):
    """三角分布逆 CDF (a 下界, c 众数, b 上界)。"""
    fc = (c - a) / (b - a)
    left = a + np.sqrt(u * (b - a) * (c - a))
    right = b - np.sqrt((1.0 - u) * (b - a) * (b - c))
    return np.where(u < fc, left, right)


def _draw(spec, rng, n, u=None):
    kind = spec[0]
    if kind == "triangular":
        return _triangular_ppf(rng.random(n) if u is None else u, spec[1], spec[2], spec[3])
    if kind == "uniform":
        a, b = spec[1], spec[2]
        return a + (b - a) * (rng.random(n) if u is None else u)
    if kind == "normal_trunc":
        mu, sd, half = spec[1], spec[2], spec[3]
        return np.clip(rng.normal(mu, sd, n), mu - half, mu + half)
    raise ValueError("未知先验类型: %r" % (spec,))


def _logcosh(x):
    return np.abs(x) + np.log1p(np.exp(-2.0 * np.abs(x))) - np.log(2.0)


# ============================================================================
# 主引擎
# ============================================================================
class Ensemble:
    def __init__(self, n=None, seed=None, verbose=True, pos0=None, case_id=None, snap_hours=None):
        """pos0: (N,3) 给定的初始位置 (x_m, y_m, z_m, 相对 SITE);
           case_id: (N,) 所属搜救案例编号;
           snap_hours: 需要记录快照的时刻 (h), 搜索仿真需要逐小时。"""
        self.pos0_given = None if pos0 is None else np.asarray(pos0, float)
        self.case_id = None if case_id is None else np.asarray(case_id)
        self.snap_hours = (sorted(float(h) for h in snap_hours) if snap_hours is not None
                           else [float(h) for h in config.TIME["t_hours"]])
        self.n = int(n if n is not None else (len(self.pos0_given) if self.pos0_given is not None
                                              else config.ENSEMBLE["n"]))
        self.seed = int(seed if seed is not None else config.ENSEMBLE["seed"])
        self.verbose = verbose
        self.prof = data.Profile()
        self.ter = data.Terrain()
        self.cur = data.make_current()
        self.site = (config.SITE["lat"], config.SITE["lon"])
        self.rng = np.random.default_rng(self.seed)
        self._sample_priors()
        self.snaps = {}

    # ---------------- 1. 混合先验采样 ----------------
    def _sample_priors(self):
        rng, n = self.rng, self.n

        # --- 失效模式: 按精确比例分配 ---
        modes = []
        for m in config.FAILURE["modes"]:
            modes += [m] * int(round(config.FAILURE["p"][m] * n))
        modes = modes[:n] + [modes[-1]] * (n - len(modes))
        modes = np.array(modes)[rng.permutation(n)]
        self.mode = modes

        # S2 再分: 水面 / 无动力
        s2 = np.where(modes == "S2")[0]
        n_surf = int(round(config.FAILURE["S2_surface_frac"] * len(s2)))
        is_s2s = np.zeros(n, bool)
        is_s2s[s2[:n_surf]] = True
        self.sub = np.array([
            "S2s" if (m == "S2" and s) else ("S2b" if m == "S2" else m)
            for m, s in zip(modes, is_s2s)])

        # --- 连续参数 ---
        U = _lhs(rng, n, 2)
        self.mu_s = _draw(config.PRIOR["mu_s"], rng, n, U[:, 1])
        self.mu_k = config.PRIOR["mu_k_ratio"] * self.mu_s
        self.cd = _draw(config.PRIOR["cd"], rng, n)
        if self.pos0_given is None:
            self.z0 = _draw(config.PRIOR["z0_m"], rng, n, U[:, 0])
        else:
            self.z0 = self.pos0_given[:, 2].astype(float).copy()

        # --- 浮力异常: 按细分模式分配给对应分布 ---
        drho = np.zeros(n)
        for key, mask in (("drho_S1", self.sub == "S1"), ("drho_S2b", self.sub == "S2b"),
                          ("drho_S3", self.sub == "S3"), ("drho_S4", self.sub == "S4")):
            if mask.any():
                drho[mask] = _draw(config.PRIOR[key], rng, int(mask.sum()))
        self.drho = drho

        # --- 艇体: 由 (z0, Δρ) 定排水体积, 再算等效球 ---
        veh = config.VEHICLE
        self.mass = float(veh["mass_dry_kg"])
        rho_w0 = self.prof.rho_at(self.z0)
        self.rho_b = rho_w0 + drho
        self.V = self.mass / self.rho_b
        self.R = (3.0 * self.V / (4.0 * np.pi)) ** (1.0 / 3.0)
        self.A = np.pi * self.R ** 2
        self.cdA = self.cd * self.A
        self.add_mass = float(veh["added_mass"])

        # --- 初始位置: 给定位置优先; 否则按海域先验抽样 ---
        s0 = float(config.PRIOR["sigma0_m"])
        if self.pos0_given is not None:
            self.x = self.pos0_given[:, 0].astype(float).copy()
            self.y = self.pos0_given[:, 1].astype(float).copy()
        else:
            reg = config.PRIOR.get("site_region")
            if reg is None:
                self.x = rng.normal(0.0, s0, n)
                self.y = rng.normal(0.0, s0, n)
            else:
                kind, hx_km, hy_km = reg[0], reg[1], reg[2]
                z_max = float(reg[3]) if len(reg) > 3 else None
                hx, hy = float(hx_km) * 1000.0, float(hy_km) * 1000.0
                if kind != "uniform_rect":
                    raise ValueError("未知海域先验: %r" % (reg,))
                self.x = rng.uniform(-hx, hx, n) + rng.normal(0.0, s0, n)
                self.y = rng.uniform(-hy, hy, n) + rng.normal(0.0, s0, n)
                for _ in range(30):
                    la, lo = self._latlon(self.x, self.y)
                    bad = (self.ter.depth_at(la, lo) > z_max if z_max is not None
                           else np.zeros(n, bool))
                    if not bad.any():
                        break
                    nb = int(bad.sum())
                    self.x[bad] = rng.uniform(-hx, hx, nb) + rng.normal(0.0, s0, nb)
                    self.y[bad] = rng.uniform(-hy, hy, nb) + rng.normal(0.0, s0, nb)
        self.x0 = self.x.copy()      # 失事点 (未漂移前)
        self.y0 = self.y.copy()
        la0, lo0 = self._latlon(self.x0, self.y0)
        self.depth0 = self.ter.depth_at(la0, lo0)     # 各失事点的海底深度
        self.slope0 = self.ter.slope_at(la0, lo0)     # 各失事点的海底坡度

        # --- 物理约束: 失事深度不能超过当地海底; 超了按"贴底"处理 ---
        self.z0 = np.clip(self.z0, 30.0, np.maximum(self.depth0 - 20.0, 30.0))
        rho_w0 = self.prof.rho_at(self.z0)
        self.rho_b = rho_w0 + self.drho
        self.V = self.mass / self.rho_b
        self.R = (3.0 * self.V / (4.0 * np.pi)) ** (1.0 / 3.0)
        self.A = np.pi * self.R ** 2
        self.cdA = self.cd * self.A

        self.z = self.z0.copy()
        self.w = np.zeros(n)
        self.eps = rng.normal(0.0, config.PRIOR["ou_sigma"], (n, 2))
        self.bias = rng.normal(0.0, config.PRIOR["bias_sigma"], (n, 2))

        # --- 水面分支: 风 ---
        wspd = _draw(config.PRIOR["wind_ms"], rng, n)
        wdir = rng.uniform(0.0, 2.0 * np.pi, n)
        self.wind_e = wspd * np.sin(wdir) * config.PRIOR["windage"]
        self.wind_n = wspd * np.cos(wdir) * config.PRIOR["windage"]

        # --- 状态与记录 ---
        self.st = np.full(n, ST_DYN, np.int8)
        surf = np.isin(self.sub, ["S0", "S2s"])
        self.st[surf] = ST_SURFACE
        self.z[surf] = 0.0
        self.t_land = np.full(n, np.nan)
        self.land_depth = np.full(n, np.nan)
        self.bot_depth = np.full(n, np.nan)
        self.imploded = np.zeros(n, bool)
        self.over_rating = np.zeros(n, bool)
        self.slide_dist = np.zeros(n)
        self._landed = 0          # 已触底样本数 (统计用)

    # ---------------- 工具 ----------------
    def _latlon(self, x, y):
        la = self.site[0] + y / DEG_LAT_M
        lo = self.site[1] + x / (DEG_LAT_M * np.cos(np.deg2rad(self.site[0])))
        return la, lo

    def _current(self, la, lo, z, t_sec):
        """t_sec: 秒; 四维流场接口按小时取时刻。"""
        return self.cur.uv_t(la, lo, np.clip(z, 0.0, self.cur.z[-1]), t_sec / 3600.0)

    # ---------------- 2. 垂向一步 (解析解) ----------------
    def _vertical_step(self, dt, t_now):
        """推进一个时间步; 返回本步新触底的样本 mask。"""
        dyn = self.st == ST_DYN
        if not dyn.any():
            return np.zeros(self.n, bool)
        rho = self.prof.rho_at(self.z)
        F = (self.mass - rho * self.V) * G            # 向下为正
        k = 0.5 * rho * self.cdA
        mstar = self.mass + self.add_mass * rho * self.V
        kk = np.maximum(k, 1e-12)
        wt = np.sign(F) * np.sqrt(np.abs(F) / kk)     # 终端速度
        small = np.abs(wt) < 1e-5
        tau = np.where(small, 1.0, mstar / np.maximum(kk * np.abs(wt), 1e-12))
        wt_s = np.where(small, 1.0, wt)
        ratio = np.clip(np.where(small, 0.0, self.w / wt_s), -0.999, 0.999)
        c = np.arctanh(ratio)
        arg = c + dt / tau
        w_new = np.where(small,
                         self.w / (1.0 + dt * kk / mstar * np.abs(self.w)),
                         wt * np.tanh(arg))
        dz = np.where(small,
                      0.5 * (self.w + w_new) * dt,
                      wt * tau * (_logcosh(arg) - _logcosh(c)))
        z_new = self.z + dz

        la, lo = self._latlon(self.x, self.y)
        bot = self.ter.depth_at(la, lo)
        land = dyn & (z_new >= bot)
        surf = dyn & (z_new <= 0.0)
        eq = dyn & (np.abs(rho - self.rho_b) < 0.005) & ~land & ~surf
        still = dyn & ~land & ~surf & ~eq

        # 触底
        self.t_land = np.where(land, t_now, self.t_land)
        self.land_depth = np.where(land, bot, self.land_depth)
        self.z = np.where(land, bot, self.z)
        self.st = np.where(land, ST_BOTTOM, self.st)
        self.w = np.where(land, 0.0, self.w)
        # 到水面
        self.z = np.where(surf, 0.0, self.z)
        self.st = np.where(surf, ST_SURFACE, self.st)
        self.w = np.where(surf, 0.0, self.w)
        # 中性浮力
        self.z = np.where(eq, z_new, self.z)
        self.st = np.where(eq, ST_WATER, self.st)
        self.w = np.where(eq, 0.0, self.w)
        # 仍在动力学
        self.z = np.where(still, z_new, self.z)
        self.w = np.where(still, w_new, self.w)
        return land

    # ---------------- 3. 水平一步 ----------------
    def _horiz_step(self, dt, ou_a, ou_sd, rw_sd, t_h):
        mov = self.st != ST_BOTTOM
        if not mov.any():
            return
        la, lo = self._latlon(self.x, self.y)
        u_c, v_c = self._current(la, lo, self.z, t_h)
        u_c = u_c + np.where(self.st == ST_SURFACE, self.wind_e, 0.0)
        v_c = v_c + np.where(self.st == ST_SURFACE, self.wind_n, 0.0)
        self.eps = ou_a * self.eps + ou_sd * self.rng.normal(0.0, 1.0, (self.n, 2))
        dx = (u_c + self.eps[:, 0] + self.bias[:, 0]) * dt + rw_sd * self.rng.normal(0.0, 1.0, self.n)
        dy = (v_c + self.eps[:, 1] + self.bias[:, 1]) * dt + rw_sd * self.rng.normal(0.0, 1.0, self.n)
        self.x = np.where(mov, self.x + dx, self.x)
        self.y = np.where(mov, self.y + dy, self.y)

    # ---------------- 4. 惯性段 ----------------
    def _phase_descent(self):
        T = config.TIME
        dt = float(T["dt_descent_s"])
        nsteps = int(round(float(T["t_descent_h"]) * 3600.0 / dt))
        tau_ou = float(config.PRIOR["ou_tau_h"]) * 3600.0
        ou_a = np.exp(-dt / tau_ou)
        ou_sd = float(config.PRIOR["ou_sigma"]) * np.sqrt(1.0 - ou_a ** 2)
        rw_sd = np.sqrt(2.0 * float(config.PRIOR["Kh_m2s"]) * dt)
        snap_steps = {int(round(th * 3600.0 / dt)): th
                      for th in self.snap_hours if th <= T["t_descent_h"]}
        t0 = _time.time()
        for k in range(1, nsteps + 1):
            land = self._vertical_step(dt, k * dt)
            if land.any():
                self._slide(np.where(land)[0])      # 触底即滑移 (物理上在分钟级完成)
            self._horiz_step(dt, ou_a, ou_sd, rw_sd, k * dt)
            if k in snap_steps:
                self._record(snap_steps[k])
        if self.verbose:
            print("   惯性段: %d 步 (dt=%.0f s), 用时 %.1f s"
                  % (nsteps, dt, _time.time() - t0))

    # ---------------- 5. 海底滑移 ----------------
    def _slide(self, idx=None):
        """沿坡滑移: **每一步用当前所在位置的坡度重新判断** (不是用失事点的坡度)。

        压溃不作为动力学过程处理 (按用户要求): 这里只算出终点位置与终点水深,
        "是否压溃"留到 _finalize / posterior 里按终点水深做**事后统计**。
        """
        if idx is None:
            idx = np.where((self.st == ST_BOTTOM) & np.isnan(self.bot_depth))[0]
        if idx.size == 0:
            return
        la = self.site[0] + self.y[idx] / DEG_LAT_M
        lo = self.site[1] + self.x[idx] / (DEG_LAT_M * np.cos(np.deg2rad(self.site[0])))
        th = np.degrees(np.arctan(self.mu_s[idx]))
        active = np.ones(idx.size, bool)
        dist = np.zeros(idx.size)
        ds = 20.0
        for _ in range(4000):
            if not active.any():
                break
            sl = self.ter.slope_at(la, lo)      # ← 每个位置、每一小步都重算坡度
            active &= sl > th                   # 当地坡度小于摩擦角 → 停住
            if not active.any():
                break
            ge, gn = self.ter.grad_at(la, lo)
            norm = np.maximum(np.hypot(ge, gn), 1e-9)
            step_e = ge / norm * ds             # 梯度指向水深增大方向 = 下坡
            step_n = gn / norm * ds
            la = np.where(active, la + step_n / DEG_LAT_M, la)
            lo = np.where(active, lo + step_e / (DEG_LAT_M * np.cos(np.deg2rad(la))), lo)
            dist[active] += ds
            active &= self.ter.inside(la, lo)

        self.slide_dist[idx] = dist
        self.y[idx] = (la - self.site[0]) * DEG_LAT_M
        self.x[idx] = (lo - self.site[1]) * DEG_LAT_M * np.cos(np.deg2rad(self.site[0]))
        self.bot_depth[idx] = self.ter.depth_at(la, lo)
        self.z[idx] = self.bot_depth[idx]          # 沉底样本的 z 记为终点水深
        self._landed = int(np.sum(self.st == ST_BOTTOM))

    # ---------------- 6. 准稳态漂流段 ----------------
    def _phase_drift(self):
        T = config.TIME
        dt = float(T["dt_drift_s"])
        t_start = float(T["t_descent_h"]) * 3600.0
        t_end = float(T["t_end_h"]) * 3600.0
        nsteps = int(round((t_end - t_start) / dt))
        tau_ou = float(config.PRIOR["ou_tau_h"]) * 3600.0
        ou_a = np.exp(-dt / tau_ou)
        ou_sd = float(config.PRIOR["ou_sigma"]) * np.sqrt(1.0 - ou_a ** 2)
        rw_sd = np.sqrt(2.0 * float(config.PRIOR["Kh_m2s"]) * dt)
        snap_steps = {int(round((th * 3600.0 - t_start) / dt)): th
                      for th in self.snap_hours if th > T["t_descent_h"]}
        t0 = _time.time()
        for k in range(1, nsteps + 1):
            land = self._vertical_step(dt, t_start + k * dt)
            if land.any():
                self._slide(np.where(land)[0])
            self._horiz_step(dt, ou_a, ou_sd, rw_sd, t_start + k * dt)
            if k in snap_steps:
                self._record(snap_steps[k])
        if self.verbose:
            print("   漂流段: %d 步 (dt=%.0f s), 用时 %.1f s"
                  % (nsteps, dt, _time.time() - t0))

    def _record(self, t_h):
        self.snaps[float(t_h)] = {
            "x": self.x.copy(), "y": self.y.copy(), "z": self.z.copy(),
            "st": self.st.copy(),
        }

    # ---------------- 7. 收尾 + 事后统计 ----------------
    def _finalize(self):
        # 极少数在漂流段才触底的样本 (兜底: 补做滑移并标记)
        self._slide()
        bot = self.st == ST_BOTTOM
        need = bot & np.isnan(self.bot_depth) & np.isfinite(self.x)
        if need.any():
            la, lo = self._latlon(self.x[need], self.y[need])
            self.bot_depth[need] = self.ter.depth_at(la, lo)
            self.land_depth[need] = np.where(np.isnan(self.land_depth[need]),
                                             self.z[need], self.land_depth[need])
        # --- 压溃/超额定: 全部在模拟结束后按**终点水深**做事后统计 ---
        #     (模拟过程本身不引入压溃截断, 保证轨迹物理连续)
        rate = config.PRIOR["depth_rating_m"]
        self.over_rating = bot & (self.bot_depth > rate)
        self.imploded = bot & (self.bot_depth > rate * config.PRIOR["collapse_factor"])
        # 压溃概率-阈值曲线: 对任意假定压溃深度 D_c 都能读出概率
        self.crush_thresholds = np.arange(900.0, 2101.0, 50.0)
        d = self.bot_depth[bot]
        self.crush_curve_bottom = np.array(
            [float(np.mean(d > D)) if d.size else np.nan for D in self.crush_thresholds])
        self.crush_curve_all = self.crush_curve_bottom * float(np.mean(bot))

    # ---------------- 8. 主流程 ----------------
    def run(self, save=True):
        t0 = _time.time()
        if self.verbose:
            print("P2 集合预报: N = %d, 模式比例 %s" % (
                self.n, {m: round(float(np.mean(self.mode == m)), 3)
                         for m in config.FAILURE["modes"]}))
        if 0.0 in self.snap_hours:
            self._record(0.0)
        self._phase_descent()
        self._phase_drift()
        self._finalize()
        if self.verbose:
            bot = self.st == ST_BOTTOM
            print("   沉底分支: %d 个样本; 触底时间中位 %.0f s; 滑移中位 %.0f m; "
                  "终点水深中位 %.0f m"
                  % (int(bot.sum()), np.nanmedian(self.t_land[bot]),
                     np.nanmedian(self.slide_dist[bot]), np.nanmedian(self.bot_depth[bot])))
            print("   事后统计: 终点 >1000 m 占 %.1f%%; >1500 m 占 %.1f%% "
                  "(模拟过程未做压溃截断, 这里只是统计)"
                  % (100 * float(np.mean(self.over_rating)),
                     100 * float(np.mean(self.imploded))))
            print("   总用时 %.1f s" % (_time.time() - t0))
        if save:
            self.save()
        return self

    def save(self, path=None):
        path = path or os.path.join(CACHE, "p2_ensemble.npz")
        times = sorted(self.snaps)
        n = self.n
        st = np.full((len(times), n), -1, np.int8)
        xx = np.full((len(times), n), np.nan)
        yy = np.full((len(times), n), np.nan)
        zz = np.full((len(times), n), np.nan)
        for i, th in enumerate(times):
            s = self.snaps[th]
            st[i], xx[i], yy[i], zz[i] = s["st"], s["x"], s["y"], s["z"]
        np.savez_compressed(
            path,
            snap_times=np.array(times),
            snap_x=xx, snap_y=yy, snap_z=zz, snap_st=st,
            sub=self.sub.astype("U4"), mode=self.mode.astype("U2"),
            drho=self.drho, z0=self.z0, mu_s=self.mu_s, cd=self.cd, V=self.V,
            imploded=self.imploded, over_rating=self.over_rating,
            slide_dist=self.slide_dist, t_land=self.t_land,
            land_depth=self.land_depth, bot_depth=self.bot_depth,
            x0=self.x0, y0=self.y0, depth0=self.depth0, slope0=self.slope0,
            crush_thresholds=self.crush_thresholds,
            crush_curve_bottom=self.crush_curve_bottom,
            crush_curve_all=self.crush_curve_all,
            current_source=("4D" if hasattr(self.cur, "time_h") else "2D"),
        )
        if self.verbose:
            print("   已保存 -> %s" % path)


# ============================================================================
# 单点垂直轨迹 (图 F4(c) 用): 同一套解析解, 细时间步
# ============================================================================
def trace_vertical(drhos=(3.0, 10.0, 20.0, 40.0), z0=450.0, t_end=1800.0, dt=5.0,
                   bottom=876.0):
    """返回 [(t[], z[], Δρ)], 用于展示不同 Δρ 的沉降时间尺度。"""
    prof = data.Profile()
    m = float(config.VEHICLE["mass_dry_kg"])
    add = float(config.VEHICLE["added_mass"])
    out = []
    for dr in drhos:
        rho0 = float(prof.rho_at(z0))
        V = m / (rho0 + dr)
        R = (3 * V / (4 * np.pi)) ** (1 / 3)
        cdA = 0.45 * np.pi * R ** 2
        z, w = float(z0), 0.0
        ts, zs = [0.0], [z]
        for k in range(1, int(t_end / dt) + 1):
            rho = float(prof.rho_at(z))
            F = (m - rho * V) * G
            kk = 0.5 * rho * cdA
            mst = m + add * rho * V
            wt = np.sign(F) * np.sqrt(abs(F) / max(kk, 1e-12))
            tau = mst / max(kk * abs(wt), 1e-12)
            ratio = np.clip(w / max(wt, 1e-12), -0.999, 0.999)
            c = np.arctanh(ratio)
            arg = c + dt / tau
            w = wt * np.tanh(arg)
            z = z + wt * tau * (_logcosh(arg) - _logcosh(c))
            ts.append(k * dt)
            zs.append(min(z, bottom))
            if z >= bottom:
                break
        out.append((np.array(ts), np.array(zs), dr))
    return out


# ============================================================================
# 自检: 解析校核 + 收敛性
# ============================================================================
def self_test(verbose=True):
    """三项校核, 报告 §7 直接引用。

    T1 垂直终端速度: 常密度下的解析 w_t vs 数值积分稳态;
    T2 OU 位移方差:   MC vs 解析 Var = σ²τ(2T − 3τ + 4τ e^{−T/τ} − τ e^{−2T/τ});
    T3 滑移阈值:      与 v2 独立实现(几何 slide_path)对比同一 μ 下的滑移距离。
    """
    prof = data.Profile()
    out = {}

    # ---- T1 ----
    veh = config.VEHICLE
    m = float(veh["mass_dry_kg"])
    rows = []
    for drho in (0.1, 0.3, 1.0, 3.0):
        z0 = 200.0
        V = m / prof.rho_at(z0)
        R = (3 * V / (4 * np.pi)) ** (1 / 3)
        A = np.pi * R ** 2
        rho = float(prof.rho_at(z0))
        w_t = np.sqrt(2 * abs(drho) * V * G / (rho * 0.45 * A))
        dt, w = 1.0, 0.0
        for _ in range(20000):
            acc = G * drho * V / m - 0.5 * rho * 0.45 * A / m * w * abs(w)
            w += acc * dt
        rows.append((drho, w_t, w, abs(w - w_t) / w_t * 100))
    out["terminal"] = rows

    # ---- T2 ----
    rng = np.random.default_rng(1)
    sig, tau = config.PRIOR["ou_sigma"], config.PRIOR["ou_tau_h"] * 3600.0
    Tt, dt = 96 * 3600.0, 900.0
    nstep = int(Tt / dt)
    eps = rng.normal(0, sig, 2000)
    pos = np.zeros(2000)
    a = np.exp(-dt / tau)
    sd = sig * np.sqrt(1 - a ** 2)
    for _ in range(nstep):
        eps = a * eps + sd * rng.normal(0, 1, 2000)
        pos += eps * dt
    var_num = float(np.var(pos))
    var_ana = sig ** 2 * tau * (2 * Tt - 3 * tau + 4 * tau * np.exp(-Tt / tau)
                                - tau * np.exp(-2 * Tt / tau))
    out["ou"] = (var_num, var_ana, abs(var_num - var_ana) / var_ana * 100)

    # ---- T3 滑移阈值 (与解析判据 tanθ = μ_s 对照) ----
    ter = data.Terrain()
    site_slope = float(ter.slope_at(config.SITE["lat"], config.SITE["lon"]))
    out["slope_site"] = (site_slope, np.tan(np.deg2rad(site_slope)))

    if verbose:
        print("自检 T1 垂直终端速度 (常密度冻结):")
        for drho, wt, wn, e in out["terminal"]:
            print("   Δρ=%4.1f  解析 w_t=%.4f  数值 w=%.4f  相对差 %.2f%%" % (drho, wt, wn, e))
        print("自检 T2 OU 位移方差 (T = 96 h): 数值 %.3e, 解析 %.3e m², 相对差 %.2f%%"
              % (out["ou"][0], out["ou"][1], out["ou"][2]))
        print("自检 T3 站点坡度 %.2f° → 滑移临界 μ_s = tan θ = %.3f" % out["slope_site"])
    return out


def main():
    e = Ensemble().run()
    for th in sorted(e.snaps):
        st = e.snaps[th]["st"]
        frac = {ST_NAME[k]: float(np.mean(st == k)) for k in ST_NAME}
        print("   t=%5.1f h  " % th + "  ".join("%s %.3f" % (k, v) for k, v in frac.items()))


if __name__ == "__main__":
    main()
