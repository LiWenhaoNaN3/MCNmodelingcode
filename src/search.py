# -*- coding: utf-8 -*-
"""搜索仿真: 给定初始位置集成 (上层) → 定位演化 → 7 种装备组合的搜救 → 成功率/时间/费用。

关键设计
--------
1. 30 000 个粒子 = 60 个"搜救案例" × 500 个粒子。
   每个案例: 搜救方只知道目标在"水平半径 100 m、垂直 ±10 m 的圆柱"内;
   圆柱中心与真值均由上层分布 (4×4 km 均匀 × 深度指数 200 m) 生成。
2. 搜救先验 = 该案例的 500 个粒子 (既是预测分布, 也是蒙特卡洛真值)。
3. 每个粒子有"存活概率" S_i。搜索某格后: S_i ← S_i·exp(−coverage)。
   P(已找到 | t) = 1 − mean(S_i)。
4. 规划: 每小时, 每台装备在**能力范围内且不超过其最大工作深度**的格子上,
   按 score = p_类别网格 / (转场时间 + 扫完一格时间) 选最高分; 扫完更新网格与 S_i。
5. 可设置"失联 → 开始搜救"的延迟 t_start (h): 延迟期间不搜索。
"""
import csv
import os

import numpy as np

from . import config, data, equipment as EQ, upper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

T_MAX = 96                 # 搜索时长 (h)
CELL = 1.0                 # 网格边长 (km)
HALF = 60.0                # 网格半宽 (km)
NG = int(round(2 * HALF / CELL))
CELL_AREA = 1e6            # m² (1 km²)
SURF, WATER, BOT = EQ.SURF, EQ.WATER, EQ.BOT

# 默认搜索参数 (可被 run_scheme / main 覆盖)
# 出发岛相对当前 SITE 的局部坐标 (km) —— 运行时按当前海域计算
def depart_km():
    dla = config.SITE["depart_lat"] - config.SITE["lat"]
    dlo = config.SITE["depart_lon"] - config.SITE["lon"]
    return (dlo * 111.32 * np.cos(np.deg2rad(config.SITE["lat"])), dla * 111.32)

_STATS = {"total": 0, "dup": 0, "dup_hour": 0}

SEARCH_CFG = {
    "t_start_h": 0.0,          # 失联到开始搜救的延迟 (h)
    "bot_shallow_m": 1000.0,   # "浅层海底"分界 (供 AUV 等深度受限装备规划用)
    "cooldown_h": 0.0,         # 硬冷却 (h): 0 = 不限制重复扫描
    "visit_coverage": 1.0,     # 单次到访的目标覆盖倍数 (1 → POD≈63%)
    "revisit_penalty": 1.0,    # 重复扫描惩罚系数: score 除以 (1+β·累计覆盖); 调优见文档
}


# ============================================================================
# 1. 真值轨迹 (只算一次, 7 个方案共用)
# ============================================================================
def build_truth(force=False, verbose=True, tag=None):
    tag = getattr(config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    path = os.path.join(CACHE, "truth_search%s.npz" % suf)
    if os.path.exists(path) and not force:
        d = np.load(path, allow_pickle=False)
        if verbose:
            print("真值轨迹: 已有缓存 cache/truth_search%s.npz (%d 粒子)"
                  % (suf, d["pos"].shape[1]))
        return {k: d[k] for k in d.files}
    pos_in = upper.load()
    N = len(pos_in["x0"])
    if verbose:
        print("生成真值轨迹: %d 粒子" % N)
    from . import locate
    ens = locate.Ensemble(pos0=np.stack([pos_in["x0"], pos_in["y0"], pos_in["z0"]], 1),
                          case_id=pos_in["case_id"],
                          snap_hours=np.arange(0, T_MAX + 1, 1.0),
                          verbose=verbose)
    ens.run(save=False)
    times = sorted(ens.snaps)
    pos = np.stack([np.stack([ens.snaps[t]["x"], ens.snaps[t]["y"], ens.snaps[t]["z"]], 1)
                    for t in times]).astype(np.float32)      # (T+1, N, 3)
    case_id = pos_in["case_id"].astype(np.int32)
    ter = data.Terrain()
    cls = np.empty((len(times), N), np.int8)
    la0, lo0 = config.SITE["lat"], config.SITE["lon"]
    cosl = np.cos(np.deg2rad(la0))
    for i in range(len(times)):
        z = pos[i, :, 2]
        la = la0 + pos[i, :, 1] / 111320.0
        lo = lo0 + pos[i, :, 0] / (111320.0 * cosl)
        bot = ter.depth_at(la, lo)
        c = np.ones(N, np.int8)
        c[z <= 2.0] = 0
        c[z >= bot - 6.0] = 2
        cls[i] = c
    out = {"pos": pos, "case_id": case_id, "cls": cls, "times": np.array(times)}
    np.savez_compressed(path, **out)
    if verbose:
        print("已保存 -> cache/truth_search%s.npz" % suf)
    return out


def load_truth(tag=None):
    tag = getattr(config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    d = np.load(os.path.join(CACHE, "truth_search%s.npz" % suf), allow_pickle=False)
    return {k: d[k] for k in d.files}


# ============================================================================
# 2. 规划与单案例仿真
# ============================================================================
def _cells(pos_km):
    """(x,y) km → 网格平坦索引; 出界返回 −1。"""
    j = ((pos_km[:, 0] + HALF) / CELL).astype(np.int32)
    i = ((pos_km[:, 1] + HALF) / CELL).astype(np.int32)
    ok = (j >= 0) & (j < NG) & (i >= 0) & (i < NG)
    return np.where(ok, i * NG + j, -1).astype(np.int32)


def _smooth(g):
    p = np.pad(g, 1, mode="edge")
    return 0.5 * g + 0.125 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])


class Asset:
    def __init__(self, name, x=0.0, y=0.0):
        spec = EQ.ASSETS[name]
        self.name = name
        self.cap = spec["cap"]
        self.v = spec["speed"]                    # 搜索速度
        self.v_t = spec.get("transit_speed", spec["speed"])   # 转场速度
        self.depth_max = spec["depth_max"]
        self.active_from = 0.0          # 何时可用 (h)
        self.label = name               # 显示名 (多船时可含编号)
        self.x, self.y = x, y

    def move_to(self, tx, ty):
        self.x, self.y = tx, ty

    def plan_grid(self, grids, c):
        if c == BOT and self.depth_max < 1500.0:
            return grids["bot_shallow"]
        return grids[c]


def _choose(asset, grids, ax, ay, claimed=None, h=0.0, last=None, cov_map=None):
    best = (-1.0, None, None, None)
    for c, W in asset.cap.items():
        g = np.asarray(asset.plan_grid(grids, c)).ravel()
        if g.sum() <= 1e-12:
            continue
        dwell = float(SEARCH_CFG["visit_coverage"]) * CELL_AREA / (W * asset.v) / 3600.0
        cand = np.flatnonzero(g > 1e-12)
        if cand.size == 0:
            continue
        if claimed:
            keep = np.array([(c, int(x)) not in claimed for x in cand])
            cand = cand[keep]
            if cand.size == 0:
                continue
        if last:
            cd = float(SEARCH_CFG["cooldown_h"])
            fresh = np.array([(h - last.get((c, int(x)), -1e9)) >= cd for x in cand])
            if fresh.any():
                cand = cand[fresh]
        val = g[cand]
        ci = cand // NG
        cj = cand % NG
        cx = (cj + 0.5) * CELL - HALF
        cy = (ci + 0.5) * CELL - HALF
        tr = np.hypot(cx - ax, cy - ay) / max(asset.v_t * 3.6, 1e-6)
        beta = float(SEARCH_CFG["revisit_penalty"])
        if beta > 0 and cov_map:
            pen = np.array([1.0 + beta * cov_map.get((c, int(x)), 0.0) for x in cand])
        else:
            pen = 1.0
        score = val / ((tr + dwell) * pen)
        k = int(np.argmax(score))
        if score[k] > best[0]:
            best = (float(score[k]), c, int(cand[k]), dwell)
    return best


def _hour_grids(z_h, cls_h, cells_h, surv):
    valid = cells_h >= 0
    g = {}
    for c in (SURF, WATER, BOT):
        a = np.zeros(NG * NG, np.float32)
        m = (cls_h == c) & valid
        if m.any():
            np.add.at(a, cells_h[m], surv[m])
        g[c] = _smooth(a.reshape(NG, NG)).ravel()
    a = np.zeros(NG * NG, np.float32)
    m = (cls_h == BOT) & valid & (z_h <= SEARCH_CFG["bot_shallow_m"])
    if m.any():
        np.add.at(a, cells_h[m], surv[m])
    g["bot_shallow"] = _smooth(a.reshape(NG, NG)).ravel()
    return g


def _run_case(assets, case_idx, truth, want_track=False, want_surv=False):
    pos = truth["pos"][:, case_idx, :].astype(np.float32)
    cls = truth["cls"][:, case_idx]
    K = pos.shape[1]
    pos_km = pos[:, :, :2] / 1000.0
    cells = np.stack([_cells(pos_km[t]) for t in range(pos.shape[0])])

    surv = np.ones(K, np.float32)
    curve = np.empty(T_MAX + 1, np.float32)
    curve[0] = 1.0
    track = [] if want_track else None
    seen = set()              # 已搜索过的 (类别, 格子) —— 用于统计重复扫描率
    last = {}                 # (类别, 格子) -> 最近一次搜索时刻
    cov_map = {}              # (类别, 格子) -> 累计覆盖倍数
    t_start = float(SEARCH_CFG["t_start_h"])
    for h in range(T_MAX):
        if h < t_start:
            curve[h + 1] = float(surv.mean())
            continue
        z_h = pos[h, :, 2]
        grids = _hour_grids(z_h, cls[h], cells[h], surv)
        claimed = set()          # 本小时已被认领的 (类别, 格子) —— 避免重复扫描
        seen_h = set()           # 本小时已经搜过的格子
        for a in assets:
            if not a.cap or h < a.active_from:
                continue
            remaining = 1.0
            for _ in range(400):
                if remaining <= 0.02:
                    break
                sc, c, cell, dwell = _choose(a, grids, a.x, a.y, claimed, h, last, cov_map)
                if c is None or sc <= 0:
                    break
                claimed.add((c, int(cell)))
                ci, cj = divmod(cell, NG)
                tx = (cj + 0.5) * CELL - HALF
                ty = (ci + 0.5) * CELL - HALF
                transit = np.hypot(tx - a.x, ty - a.y) / max(a.v_t * 3.6, 1e-6)
                if transit >= remaining:
                    frac = remaining / transit
                    a.move_to(a.x + (tx - a.x) * frac, a.y + (ty - a.y) * frac)
                    remaining = 0.0
                    break
                a.move_to(tx, ty)
                remaining -= transit
                _STATS["total"] += 1
                if (c, int(cell)) in seen_h:
                    _STATS["dup_hour"] += 1
                seen_h.add((c, int(cell)))
                if (c, int(cell)) in seen:
                    _STATS["dup"] += 1
                seen.add((c, int(cell)))
                last[(c, int(cell))] = h
                tau = min(dwell, remaining)
                W = a.cap[c]
                cov = W * a.v * (tau * 3600.0) / CELL_AREA
                cov_map[(c, int(cell))] = cov_map.get((c, int(cell)), 0.0) + cov
                m = (cls[h] == c) & (cells[h] == cell) & (cell >= 0) & (z_h <= a.depth_max)
                if m.any():
                    surv[m] *= np.float32(np.exp(-cov))
                g = np.asarray(a.plan_grid(grids, c))
                g[cell] *= np.float32(np.exp(-cov))
                remaining -= tau
                if want_track:
                    track.append((h, getattr(a, "label", a.name), a.x, a.y, c, cov))
        curve[h + 1] = float(surv.mean())
    if want_surv:
        return curve, surv, (track if want_track else None)
    return curve, (track if want_track else None)


def run_assets(asset_factory, truth, verbose=True, want_track=False):
    """通用多资产搜索: asset_factory(中心坐标km) -> [Asset]"""
    _STATS["total"] = 0
    _STATS["dup"] = 0
    _STATS["dup_hour"] = 0
    case_id = truth["case_id"]
    cases = np.unique(case_id)
    total = len(case_id)
    idx_by_case = {int(m): np.where(case_id == m)[0] for m in cases}
    acc = np.zeros(T_MAX + 1)
    track_all = None
    for m in cases:
        idx = idx_by_case[int(m)]
        ctr = truth["pos"][0, idx, :2].mean(axis=0) / 1000.0
        assets = asset_factory(ctr)
        curve, track = _run_case(assets, idx, truth, want_track=(want_track and int(m) == 0))
        acc += curve * len(idx)
        if track is not None:
            track_all = track
    acc /= total
    found = 1.0 - acc
    res = {"found": found, "p24": float(found[24]), "p48": float(found[48]),
           "p72": float(found[72]), "p96": float(found[96])}
    res["E_T"] = float(np.sum(1.0 - found[:-1]))
    k = int(np.searchsorted(found, 0.5))
    res["T_med"] = float(k) if (k < len(found) and found[k] >= 0.5) else float("nan")
    res["actions"] = int(_STATS["total"])
    res["dup_rate"] = float(_STATS["dup"] / _STATS["total"]) if _STATS["total"] else float("nan")
    res["dup_hour_rate"] = (float(_STATS["dup_hour"] / _STATS["total"])
                             if _STATS["total"] else float("nan"))
    return res, track_all


def run_scheme(combo, truth, verbose=True, want_track=False, t_start_h=None):
    if t_start_h is not None:
        SEARCH_CFG["t_start_h"] = float(t_start_h)
    _STATS["total"] = 0
    _STATS["dup"] = 0
    _STATS["dup_hour"] = 0
    case_id = truth["case_id"]
    cases = np.unique(case_id)
    total = len(case_id)
    idx_by_case = {int(m): np.where(case_id == m)[0] for m in cases}
    acc = np.zeros(T_MAX + 1)
    track_all = None
    for m in cases:
        idx = idx_by_case[int(m)]
        ctr = truth["pos"][0, idx, :2].mean(axis=0) / 1000.0
        # 雷达随母船 (在失事点上空); 其余装备从出发岛出发
        assets = [Asset("radar", ctr[0], ctr[1])]
        dep = depart_km()
        assets += [Asset(a, dep[0], dep[1]) for a in combo]
        curve, track = _run_case(assets, idx, truth, want_track=(want_track and int(m) == 0))
        acc += curve * len(idx)
        if track is not None:
            track_all = track
    acc /= total
    found = 1.0 - acc
    res = {"combo": combo, "cost_wan": EQ.cost(combo), "found": found,
           "p24": float(found[24]), "p48": float(found[48]),
           "p72": float(found[72]), "p96": float(found[96])}
    res["E_T"] = float(np.sum(1.0 - found[:-1]) * 1.0)
    k = int(np.searchsorted(found, 0.5))
    res["T_med"] = float(k) if (k < len(found) and found[k] >= 0.5) else float("nan")
    res["actions"] = int(_STATS["total"])
    res["dup_rate"] = float(_STATS["dup"] / _STATS["total"]) if _STATS["total"] else float("nan")
    if verbose:
        print("  %-26s 成本 %5.1f 万 | P24 %.2f P48 %.2f P72 %.2f P96 %.2f | T50 %s h"
              % (EQ.scheme_name(combo), res["cost_wan"], res["p24"], res["p48"],
                 res["p72"], res["p96"],
                 "%.1f" % res["T_med"] if np.isfinite(res["T_med"]) else ">96"))
    return res, track_all


# ============================================================================
# 3. 主流程与输出
# ============================================================================
def _save(results, curves, tag="", track=None):
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "search_results%s.npz" % tag),
                        curves=np.array(curves),
                        ids=np.array([r["id"] for r in results]),
                        names=np.array([r["name"] for r in results]),
                        costs=np.array([r["cost_wan"] for r in results]))
    fn = os.path.join(DATA, "搜索方案对比%s.csv" % tag)
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方案", "装备组合", "成本估计(万美元)", "P(24h)", "P(48h)",
                    "P(72h)", "P(96h)", "中位发现时间(h)", "期望发现时间(h)"])
        for r in results:
            w.writerow([r["id"], r["name"], "%.1f" % r["cost_wan"],
                        "%.3f" % r["p24"], "%.3f" % r["p48"], "%.3f" % r["p72"],
                        "%.3f" % r["p96"],
                        "%.1f" % r["T_med"] if np.isfinite(r["T_med"]) else ">96",
                        "%.1f" % r["E_T"]])
    print("已保存 -> %s" % fn)
    if track:
        import json
        names, tracks = [], {}
        for (h, nm, x, y, c, cov) in track:
            if nm not in tracks:
                names.append(nm)
                tracks[nm] = []
            tracks[nm].append([round(float(x), 2), round(float(y), 2)])
        with open(os.path.join(CACHE, "search_plan_example.json"), "w", encoding="utf-8") as f:
            json.dump({"names": names, "tracks": tracks}, f, ensure_ascii=False)


def main(force_truth=False, t_start_h=0.0, tag=None):
    if tag is not None:
        config.set_site(tag)
    tag = getattr(config, "SITE_TAG", None)
    suf = ("_" + tag) if tag else ""
    truth = build_truth(force=force_truth, tag=tag)
    SEARCH_CFG["t_start_h"] = float(t_start_h)
    print("\n7 种装备组合的搜救仿真 (30 000 粒子; 搜救启动延迟 %.0f h):" % t_start_h)
    rows, curves, track_example = [], [], None
    for i, combo in enumerate(EQ.schemes(), 1):
        res, track = run_scheme(combo, truth, want_track=(i == 2))
        res["id"] = "M%d" % i
        res["name"] = EQ.scheme_name(combo)
        rows.append(res)
        curves.append(res["found"])
        if track is not None:
            track_example = track
    _save(rows, curves, tag=suf, track=track_example)
    return rows, curves, track_example


def delay_sensitivity(delays=(0, 6, 12, 24), schemes_idx=(1, 2, 7), tag=None):
    tag = getattr(config, "SITE_TAG", None) if tag is None else tag
    suf = ("_" + tag) if tag else ""
    truth = build_truth(tag=tag)
    rows = []
    print("\n搜救启动延迟敏感性:")
    for t0 in delays:
        SEARCH_CFG["t_start_h"] = float(t0)
        for i in schemes_idx:
            combo = EQ.schemes()[i - 1]
            res, _ = run_scheme(combo, truth, verbose=True, t_start_h=t0)
            res["id"] = "M%d" % i
            res["name"] = EQ.scheme_name(combo)
            res["delay_h"] = t0
            rows.append(res)
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "搜救延迟敏感性%s.csv" % suf)
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["延迟(h)", "方案", "组合", "P(24h)", "P(48h)", "P(72h)", "P(96h)",
                    "中位时间(h)", "期望时间(h)"])
        for r in rows:
            w.writerow([r["delay_h"], r["id"], r["name"], "%.3f" % r["p24"],
                        "%.3f" % r["p48"], "%.3f" % r["p72"], "%.3f" % r["p96"],
                        "%.1f" % r["T_med"] if np.isfinite(r["T_med"]) else ">96",
                        "%.1f" % r["E_T"]])
    print("已保存 -> %s" % fn)
    return rows


if __name__ == "__main__":
    main()
