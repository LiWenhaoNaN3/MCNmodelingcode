# -*- coding: utf-8 -*-
"""多船 + 多 AUV 协同搜索 (15 种组合)。

设定 (按用户指定):
  * 每艘船 = 与母船相同的 M4 配置: 雷达 + 深水拖曳侧扫 + 深水多波束 (+ ROV 仅确认);
  * t = 0 时只有母船在失事点上方; 其余增援船与 AUV 在 t = 3 h 到位;
  * 增援船从出发岛 (Moni Strofadon) 出发, 转场约 0.5-1 h;
  * AUV 由母船携带, 3 h 后开始投放;
  * 组合: 船数 2-4 × AUV 数 0-4 = 15 种;
  * 协调: 每小时执行"认领 + 贝叶斯更新", 已被某资产认领/搜索的格子不再派给其他资产,
    从而避免重复扫描。

成本 (单次事故估计, 万美元):
    总成本 = 母船 M4 基线 182 + 70 × (船数 − 1) + 57 × AUV 数
"""
import csv
import os

import numpy as np

from . import search as SE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

T_START = 3.0          # 增援到位时间 (h)
COST_BASE = 182.0      # 母船 M4 配置基线
COST_SHIP = 70.0       # 每增加一艘同配置船 (动员+船时+设备)
COST_AUV = 57.0        # 每增加一台 AUV

SHIP_ASSET_NAMES = ("radar", "sidescan", "multibeam")


def cost(n_ships, n_auv):
    return COST_BASE + COST_SHIP * (n_ships - 1) + COST_AUV * n_auv


def asset_factory(n_ships, n_auv):
    def factory(ctr):
        assets = []
        short = {"radar": "雷达", "sidescan": "侧扫", "multibeam": "多波束"}
        # 母船 (0 号船): 失事点上方, t=0 起
        for name in SHIP_ASSET_NAMES:
            a = SE.Asset(name, ctr[0], ctr[1])
            a.active_from = 0.0
            a.label = "母船" + short[name]
            assets.append(a)
        # 增援船 (1..n-1): 从出发岛出发, t=3 h 起
        for k in range(1, n_ships):
            for name in SHIP_ASSET_NAMES:
                dep = SE.depart_km()
                a = SE.Asset(name, dep[0], dep[1])
                a.active_from = T_START
                a.label = "增援船%d" % k + short[name]
                assets.append(a)
        # AUV: 母船携带, t=3 h 起投放
        for k in range(n_auv):
            a = SE.Asset("auv", ctr[0], ctr[1])
            a.active_from = T_START
            a.label = "AUV%d" % (k + 1)
            assets.append(a)
        return assets
    return factory


def run_all(truth=None, tag=None):
    tag = getattr(SE.config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    truth = truth or SE.load_truth(tag)
    rows = []
    curves = []
    print("\n多船 + 多 AUV 协同搜索 (15 种组合; 增援于 %.0f h 到位):" % T_START)
    for n_ships in (2, 3, 4):
        for n_auv in (0, 1, 2, 3, 4):
            res, _ = SE.run_assets(asset_factory(n_ships, n_auv), truth, want_track=False)
            res["n_ships"] = n_ships
            res["n_auv"] = n_auv
            res["cost_wan"] = cost(n_ships, n_auv)
            rows.append(res)
            curves.append(res["found"])
            print("  船 %d + AUV %d | 成本 %5.0f 万 | P72 %.3f P96 %.3f | 动作 %d | 重复率 %.1f%%"
                  % (n_ships, n_auv, res["cost_wan"], res["p72"], res["p96"],
                     res["actions"], 100 * res["dup_rate"]))
    # 保存
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "多船多AUV组合%s.csv" % suf)
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["船数", "AUV数", "成本估计(万美元)", "P24", "P48", "P72", "P96",
                    "中位时间(h)", "期望时间(h)", "搜索动作数", "重复扫描率"])
        for r in rows:
            w.writerow([r["n_ships"], r["n_auv"], "%.0f" % r["cost_wan"],
                        "%.3f" % r["p24"], "%.3f" % r["p48"], "%.3f" % r["p72"],
                        "%.3f" % r["p96"],
                        "%.1f" % r["T_med"] if np.isfinite(r["T_med"]) else ">96",
                        "%.1f" % r["E_T"], r["actions"], "%.3f" % r["dup_rate"]])
    np.savez_compressed(os.path.join(CACHE, "multi_results%s.npz" % suf),
                        curves=np.array(curves),
                        n_ships=np.array([r["n_ships"] for r in rows]),
                        n_auv=np.array([r["n_auv"] for r in rows]),
                        costs=np.array([r["cost_wan"] for r in rows]))
    print("已保存 -> %s" % fn)
    return rows, curves


def save_plan_example(n_ships=3, n_auv=2, asset_combo=None, tag=None):
    """保存一个示例案例的多资产搜索航迹 (供 fig_multi_plan)。"""
    import json
    tag = getattr(SE.config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    truth = SE.load_truth(tag)
    res, track = SE.run_assets(asset_factory(n_ships, n_auv), truth, want_track=True)
    if not track:
        print("  [警告] 没有航迹数据")
        return None
    names, tracks = [], {}
    for (h, nm, x, y, c, cov) in track:
        if nm not in tracks:
            names.append(nm)
            tracks[nm] = []
        tracks[nm].append([round(float(x), 2), round(float(y), 2), round(float(h), 1)])
    fn = os.path.join(CACHE, "multi_plan_example%s.json" % suf)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"n_ships": n_ships, "n_auv": n_auv, "names": names,
                   "tracks": tracks, "P72": res["p72"]}, f, ensure_ascii=False)
    print("已保存 -> cache/multi_plan_example.json (%d 条航迹)" % len(names))
    return fn


def tune_beta(betas=(0.0, 0.2, 0.5, 1.0, 2.0, 5.0), n_ships=2, n_auv=0, tag=None):
    """重复扫描惩罚系数 β 的调优实验 (协调查阅: 成功率 vs 重复率)。"""
    tag = getattr(SE.config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    truth = SE.load_truth(tag)
    rows = []
    print("\nβ 调优 (船 %d + AUV %d):" % (n_ships, n_auv))
    for b in betas:
        SE.SEARCH_CFG["revisit_penalty"] = float(b)
        r, _ = SE.run_assets(asset_factory(n_ships, n_auv), truth)
        rows.append({"beta": b, "p72": r["p72"], "p96": r["p96"],
                     "E_T": r["E_T"], "dup": r["dup_rate"]})
        print("  β=%.1f | P72 %.3f | 期望时间 %.1f h | 跨小时重复率 %.1f%%"
              % (b, r["p72"], r["E_T"], 100 * r["dup_rate"]))
    SE.SEARCH_CFG["revisit_penalty"] = 1.0     # 恢复默认
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "协同调优_beta%s.csv" % suf)
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["beta", "P72", "P96", "期望时间(h)", "跨小时重复率"])
        for r in rows:
            w.writerow(["%.1f" % r["beta"], "%.3f" % r["p72"], "%.3f" % r["p96"],
                        "%.1f" % r["E_T"], "%.3f" % r["dup"]])
    print("已保存 -> %s" % fn)
    return rows


def delay_sensitivity(delays=(0.0, 3.0, 6.0, 12.0),
                      combos=((2, 0), (3, 2), (4, 4)), tag=None):
    """增援到位时间敏感性 (基准 3 h)。"""
    global T_START
    tag = getattr(SE.config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    truth = SE.load_truth(tag)
    rows = []
    print("\n增援到位时间敏感性:")
    for d in delays:
        T_START = float(d)
        for ns, na in combos:
            r, _ = SE.run_assets(asset_factory(ns, na), truth)
            rows.append({"delay": d, "n_ships": ns, "n_auv": na,
                         "cost": cost(ns, na), "p72": r["p72"],
                         "p96": r["p96"], "E_T": r["E_T"],
                         "dup": r["dup_rate"]})
            print("  到位 %4.1f h | 船%d AUV%d | P72 %.3f | 期望 %.1f h"
                  % (d, ns, na, r["p72"], r["E_T"]))
    T_START = 3.0
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "多船增援延迟%s.csv" % suf)
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["增援到位(h)", "船数", "AUV数", "成本(万)", "P72", "P96",
                    "期望时间(h)", "跨小时重复率"])
        for r in rows:
            w.writerow(["%.0f" % r["delay"], r["n_ships"], r["n_auv"],
                        "%.0f" % r["cost"], "%.3f" % r["p72"], "%.3f" % r["p96"],
                        "%.1f" % r["E_T"], "%.3f" % r["dup"]])
    print("已保存 -> %s" % fn)
    return rows


if __name__ == "__main__":
    run_all()
