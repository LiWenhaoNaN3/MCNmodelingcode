# -*- coding: utf-8 -*-
"""P2 后验统计: 由集合预报生成 p(x,y,t)、包含域 A50/A95、模式后验、深度分布。

输出:
  cache/p2_posterior.npz  给图件模块用
  data/T5_关键机理结果.csv 报告表 T5 的数据源
终端打印: 各时刻的包含域与分支概率 (P2 评审用)
"""
import csv
import os

import numpy as np

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

UNDER = ("S1", "S2b", "S3", "S4")   # 水下分支 (排除 S0 与 S2s 水面分支)
WATER = ("S1", "S2b")                # 悬浮/漂流分支
BOTTOM = ("S3", "S4")                # 沉底分支


def _area_of_containment(H, dxy_km, frac):
    """由二维概率网格算包含域面积 (km²)。"""
    p = H.ravel().astype(float)
    p = p / p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, frac)) + 1
    return float(k * dxy_km * dxy_km)


class Posterior:
    def __init__(self, path=None):
        d = np.load(path or os.path.join(CACHE, "p2_ensemble.npz"), allow_pickle=False)
        self.times = d["snap_times"].astype(float)
        self.x = d["snap_x"].astype(float)
        self.y = d["snap_y"].astype(float)
        self.z = d["snap_z"].astype(float)
        self.st = d["snap_st"]
        self.sub = d["sub"].astype(str)
        self.mode = d["mode"].astype(str)
        self.drho = d["drho"]
        self.z0 = d["z0"]
        self.mu_s = d["mu_s"]
        self.imploded = d["imploded"]
        self.over_rating = d["over_rating"]
        self.slide_dist = d["slide_dist"]
        self.bot_depth = d["bot_depth"]
        self.t_land = d["t_land"]
        self.crush_thresholds = d["crush_thresholds"]
        self.crush_curve_bottom = d["crush_curve_bottom"]
        self.crush_curve_all = d["crush_curve_all"]
        self.n = self.x.shape[1]
        g = config.GRID
        self.half = float(g["half_km"])
        self.dxy = float(g["dxy_km"])
        nb = int(round(2 * self.half / self.dxy))
        self.edges = np.linspace(-self.half, self.half, nb + 1)
        self.nb = nb
        self.und = np.isin(self.sub, UNDER)
        self.wat = np.isin(self.sub, WATER)
        self.bot = np.isin(self.sub, BOTTOM)
        self.grids = np.zeros((len(self.times), nb, nb))       # 水下分支混合 (平滑, 供出图)
        self.grids_w = np.zeros((len(self.times), nb, nb))     # 仅悬浮分支 (平滑)
        self.grids_b = np.zeros((len(self.times), nb, nb))     # 仅沉底分支 (平滑)
        self.grids_raw = np.zeros((len(self.times), nb, nb))   # 未平滑: 混合分支
        self.grids_w_raw = np.zeros((len(self.times), nb, nb))  # 未平滑: 悬浮分支
        self.grids_b_raw = np.zeros((len(self.times), nb, nb))  # 未平滑: 沉底分支
        self.zhist = np.zeros((len(self.times), int(g["z_max_m"] / g["dz_m"])))
        self.z_edges = np.arange(0, float(g["z_max_m"]) + g["dz_m"], g["dz_m"])

    @staticmethod
    def _blur(H, n=2):
        """3×3 邻域平滑 (0.5 自身 + 0.125 四邻), 抑制 1 km 网格的抽样噪声。"""
        H = np.array(H, float)
        for _ in range(int(n)):
            Hp = np.pad(H, 1, mode="constant")
            H = 0.5 * H + 0.125 * (Hp[:-2, 1:-1] + Hp[2:, 1:-1]
                                   + Hp[1:-1, :-2] + Hp[1:-1, 2:])
        return H

    def _grid(self, x, y, smooth=True):
        H, _, _ = np.histogram2d(y / 1000.0, x / 1000.0, bins=[self.edges, self.edges])
        H = H.astype(float)
        s = H.sum()
        H = H / s if s > 0 else H
        return self._blur(H) if smooth else H

    # ---------------- 主计算 ----------------
    def compute(self):
        out = []
        for i, th in enumerate(self.times):
            sel = self.und
            xk = self.x[i, sel]
            yk = self.y[i, sel]
            self.grids_raw[i] = self._grid(xk, yk, smooth=False)
            self.grids[i] = self._grid(xk, yk)
            self.grids_w_raw[i] = self._grid(self.x[i, self.wat], self.y[i, self.wat], smooth=False)
            self.grids_w[i] = self._grid(self.x[i, self.wat], self.y[i, self.wat])
            self.grids_b_raw[i] = self._grid(self.x[i, self.bot], self.y[i, self.bot], smooth=False)
            self.grids_b[i] = self._grid(self.x[i, self.bot], self.y[i, self.bot])
            inw = sel & (self.st[i] == 1)
            zh, _ = np.histogram(self.z[i, inw], bins=self.z_edges)
            self.zhist[i] = zh / max(zh.sum(), 1)

            st = self.st[i]
            n = float(self.n)
            r_und = np.hypot(self.x[i, sel], self.y[i, sel]) / 1000.0
            r_sur = np.hypot(self.x[i, ~self.und], self.y[i, ~self.und]) / 1000.0
            row = {
                "t_h": float(th),
                "P_S1": float(np.mean(self.sub == "S1")),
                "P_S2b": float(np.mean(self.sub == "S2b")),
                "P_S3": float(np.mean(self.sub == "S3")),
                "P_S4": float(np.mean(self.sub == "S4")),
                "P_S0": float(np.mean(self.sub == "S0")),
                "P_S2s": float(np.mean(self.sub == "S2s")),
                "P_water": float(np.mean(np.isin(self.sub, WATER))),
                "P_bottom": float(np.mean(np.isin(self.sub, BOTTOM))),
                "n_water_now": int(np.sum(inw)),
                "n_bottom_now": int(np.sum(sel & (st == 3))),
                "A50_km2": _area_of_containment(self.grids_raw[i], self.dxy, 0.50),
                "A95_km2": _area_of_containment(self.grids_raw[i], self.dxy, 0.95),
                "A50w_km2": _area_of_containment(self.grids_w_raw[i], self.dxy, 0.50),
                "A95w_km2": _area_of_containment(self.grids_w_raw[i], self.dxy, 0.95),
                "A50b_km2": _area_of_containment(self.grids_b_raw[i], self.dxy, 0.50),
                "A95b_km2": _area_of_containment(self.grids_b_raw[i], self.dxy, 0.95),
                "r50_km": float(np.percentile(r_und, 50)),
                "r95_km": float(np.percentile(r_und, 95)),
                "surface_r90_km": float(np.percentile(r_sur, 90)) if r_sur.size else np.nan,
                "P_imploded": float(np.mean(np.isin(self.sub, BOTTOM) & self.imploded)),
                "P_over_rating": float(np.mean(np.isin(self.sub, BOTTOM) & self.over_rating)),
                "z_water_med": float(np.median(self.z[i, inw])) if inw.any() else np.nan,
            }
            # 沉底分支: 触底时间 / 终点水深 / 滑移距离 (与时间无关, 只写一次)
            bot = np.isin(self.sub, BOTTOM)
            if bot.any():
                row["bot_depth_med"] = float(np.nanmedian(self.bot_depth[bot]))
                row["slide_med_m"] = float(np.nanmedian(self.slide_dist[bot]))
                row["t_land_med_s"] = float(np.nanmedian(self.t_land[bot]))
            else:
                row["bot_depth_med"] = np.nan
                row["slide_med_m"] = np.nan
                row["t_land_med_s"] = np.nan
            out.append(row)
        self.rows = out
        return out

    # ---------------- 输出 ----------------
    def save(self, path=None):
        path = path or os.path.join(CACHE, "p2_posterior.npz")
        np.savez_compressed(
            path,
            times=self.times, grids=self.grids, grids_w=self.grids_w, grids_b=self.grids_b,
            grids_raw=self.grids_raw, grids_w_raw=self.grids_w_raw,
            grids_b_raw=self.grids_b_raw, edges=self.edges,
            zhist=self.zhist, z_edges=self.z_edges,
            keys=np.array(list(self.rows[0].keys())),
            values=np.array([[r[k] for k in self.rows[0].keys()] for r in self.rows]),
            drift_dist_km=self._drift_dist(),
            crush_thresholds=self.crush_thresholds,
            crush_curve_bottom=self.crush_curve_bottom,
            crush_curve_all=self.crush_curve_all,
        )
        print("已保存 -> %s" % path)

    def _drift_dist(self):
        """悬浮分支 10/50/90 分位漂移距离 (km), 用于图 F6。"""
        out = []
        for i in range(len(self.times)):
            r = np.hypot(self.x[i, self.wat], self.y[i, self.wat]) / 1000.0
            out.append([np.percentile(r, 10), np.percentile(r, 50), np.percentile(r, 90)])
        return np.array(out)

    def save_csv(self):
        os.makedirs(DATA, exist_ok=True)
        path = os.path.join(DATA, "T5_关键机理结果.csv")
        cols = list(self.rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.rows:
                w.writerow(["%.4f" % r[c] if isinstance(r[c], float) else r[c] for c in cols])
        print("已保存 -> %s" % path)

    def report(self):
        print("=" * 104)
        print("P2 后验统计 (水下分支条件分布; 网格 %d×%d, 格距 %.1f km)" % (self.nb, self.nb, self.dxy))
        print("=" * 104)
        print("  t(h)   P(S1) P(S2b) P(S3)  P(S4)   A50    A95  | 悬浮A50 悬浮A95 | 沉底A95   r50   r95  悬浮深度中位")
        for r in self.rows:
            print("  %5.1f  %.3f %.3f  %.3f %.3f  %6.0f %6.0f |  %6.0f %6.0f  |  %6.0f  %5.1f %5.1f   %6.0f m"
                  % (r["t_h"], r["P_S1"], r["P_S2b"], r["P_S3"], r["P_S4"],
                     r["A50_km2"], r["A95_km2"], r["A50w_km2"], r["A95w_km2"], r["A95b_km2"],
                     r["r50_km"], r["r95_km"], r["z_water_med"]))
        r = self.rows[-1]
        print("  --- 沉底分支不随时间变的量 (来自 S3+S4, 占总概率 %.1f%%):" % (100 * r["P_bottom"]))
        print("      触底时间中位 %.0f s; 终点水深中位 %.0f m; 滑移距离中位 %.0f m"
              % (r["t_land_med_s"], r["bot_depth_med"], r["slide_med_m"]))
        print("      超额定潜深(>1000 m) 占全体 %.1f%%" % (100 * r["P_over_rating"]))
        print("      事后统计 (模拟过程不含压溃截断, 按终点水深判):")
        for D in (1000, 1250, 1500, 1750, 2000):
            j = int(np.argmin(np.abs(self.crush_thresholds - D)))
            print("        若压溃深度 D_c=%4.0f m -> 占全体 %.1f%%, 占沉底分支 %.1f%%"
                  % (D, 100 * self.crush_curve_all[j], 100 * self.crush_curve_bottom[j]))
        print("  注: A50/A95 为水下分支的 50%%/95%% 包含域面积 (km²); r50/r95 为距失事点的漂移距离分位数;")
        print("      水面分支(S0+S2s, 占 %.0f%%)漂移极快, 应在数小时内被雷达/目视发现, 不进入声学搜索域。"
              % (100 * (r["P_S0"] + r["P_S2s"])))

    def convergence_check(self, t_h=24.0):
        """集合收敛性: 随机二分法 (bootstrap) 比较两半样本的 A95 相对差。"""
        i = int(np.argmin(np.abs(self.times - t_h)))
        sel = self.und
        xk = self.x[i, sel] / 1000.0
        yk = self.y[i, sel] / 1000.0
        rng = np.random.default_rng(7)
        perm = rng.permutation(xk.size)
        m = xk.size // 2
        a = []
        for part in (perm[:m], perm[m:]):
            H, _, _ = np.histogram2d(yk[part], xk[part], bins=[self.edges, self.edges])
            a.append(_area_of_containment(H, self.dxy, 0.95))
        H, _, _ = np.histogram2d(yk, xk, bins=[self.edges, self.edges])
        a_all = _area_of_containment(H, self.dxy, 0.95)
        print("收敛性检查 (t=%.0f h): 全样本 A95 = %.0f km²; 随机两半分别 %.0f / %.0f km², "
              "相对差 %.1f%% (< 5%% 判合格)" % (t_h, a_all, a[0], a[1],
                                                abs(a[0] - a[1]) / a_all * 100))
        return a_all, a


def main():
    p = Posterior()
    p.compute()
    p.report()
    p.convergence_check()
    p.save()
    p.save_csv()


if __name__ == "__main__":
    main()
