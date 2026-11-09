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
