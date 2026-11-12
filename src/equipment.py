# -*- coding: utf-8 -*-
"""搜索装备: 能力参数、7 种组合方案、费用估计。

目标类别: 0 = 水面, 1 = 水柱, 2 = 海底。
能力用 {目标类别: 有效扫宽 W(m)} 表示; 空 dict 表示不承担面积搜索。
速度按用户给定表 (1 kn = 0.5144 m/s)。费用为**公开市场量级估计**。

关键假设 (按用户指示):
  * 拖曳侧扫: 本场景水深大, 取"深水拖曳" 1.5 kn; 只搜海底, 不搜水柱;
  * 深水多波束: 若要搜水柱, 必须带 WCI (Water Column Imaging) 功能,
    且水柱 POD 明显低于海底 (水柱扫宽取保守值 120 m, 海底 500 m);
    不考虑其水面搜索能力;
  * AUV: 带侧扫/前视声呐; 水柱能力保守 (120 m);
  * ROV: 不承担面积搜索, 只做确认/救援;
  * 水面雷达/目视: 所有方案都有 (母船自带), 5 kn。
"""
import itertools

KN = 0.514444            # 1 knot = 0.514444 m/s

# 目标类别
SURF, WATER, BOT = 0, 1, 2
CLASS_NAME = {SURF: "水面", WATER: "水柱", BOT: "海底"}

# ---------------- 单个装备 ----------------
ASSETS = {
    "radar": {
        "name": "水面雷达/目视",
        "cap": {SURF: 3000.0},
        "speed": 5.0 * KN,        # 5 kn
        "transit_speed": 5.0 * KN,
        "depth_max": 0.0,         # 只搜水面
        "cost_fixed": 0.0,        # 母船已有
        "cost_day": 3.0,          # 万美元/天 (船时+人员)
    },
    "sidescan": {
        "name": "深水拖曳侧扫",
        "cap": {BOT: 200.0},
        "speed": 1.5 * KN,        # 搜索速度: 深水拖曳 1.5 kn
        "transit_speed": 10.0 * KN,   # 转场速度 (母船拖带 10 kn)
        "depth_max": 2000.0,
        "cost_fixed": 10.0,       # 动员/租用固定费 (万美元)
        "cost_day": 2.0,
    },
    "multibeam": {
        "name": "深水多波束(WCI)",
        "cap": {BOT: 500.0, WATER: 120.0},   # 水柱需 WCI, 扫宽保守
        "speed": 2.5 * KN,        # 搜索速度 2.5 kn
        "transit_speed": 10.0 * KN,   # 转场 10 kn
        "depth_max": 3000.0,
        "cost_fixed": 20.0,
        "cost_day": 8.0,
    },
    "auv": {
        "name": "AUV",
        "cap": {BOT: 200.0, WATER: 120.0},
        "speed": 3.0 * KN,        # 搜索速度 3 kn
        "transit_speed": 3.0 * KN,
        "depth_max": 2000.0,      # 2000 m 级 AUV (用户指定)
        "cost_fixed": 25.0,
        "cost_day": 8.0,
    },
    "rov": {
        "name": "工作级 ROV",
        "cap": {},                # 不承担面积搜索 (确认/救援)
        "speed": 0.5 * KN,        # 0.5 kn
        "transit_speed": 10.0 * KN,
        "depth_max": 3000.0,
        "cost_fixed": 40.0,
        "cost_day": 15.0,
    },
}

# 三件可组合装备 (7 种非空组合)
COMBO3 = ["sidescan", "multibeam", "auv"]
ALWAYS = ["radar", "rov"]          # 所有方案都有


def schemes():
    out = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(COMBO3, r):
            out.append(list(combo))
    return out


def scheme_name(combo):
    return "+".join([ASSETS[a]["name"].split("(")[0] for a in combo])


def scheme_assets(combo):
    """参与面积搜索的资产: 雷达(恒有) + 组合内的装备; ROV 不参与。"""
    return ["radar"] + list(combo)


def cost(combo, days=4.0):
    """单次事故部署与使用成本估计 (万美元)。"""
    total = 0.0
    for a in scheme_assets(combo) + ["rov"]:
        spec = ASSETS[a]
        total += spec["cost_fixed"] + spec["cost_day"] * days
    return total


def table():
    rows = []
    for i, combo in enumerate(schemes(), 1):
        rows.append({"id": "M%d" % i, "combo": combo,
                     "name": scheme_name(combo), "cost_wan": cost(combo)})
    return rows


if __name__ == "__main__":
    for r in table():
        print("%s  %-26s 成本 %.1f 万美元" % (r["id"], r["name"], r["cost_wan"]))
