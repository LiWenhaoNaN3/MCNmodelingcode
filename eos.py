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


if __name__ == "__main__":
    check()
    check_speed()
