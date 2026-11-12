# -*- coding: utf-8 -*-
"""数据下载: ETOPO 地形 (ERDDAP) + HYCOM 剖面与流场 (OPeNDAP DAP2 ASCII)。

支持多海域: 调用前用 config.set_site("carib") 切换, 数据文件自动带 _carib 后缀。
HYCOM GLBy0.08 全球网格: 纬度步长 0.04° (自 −80° 起), 经度步长 0.08° (0–360°)。
"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")

# HYCOM GLBy0.08 网格参数
HLAT0, HLAT_STEP = -80.0, 0.04
HLON0, HLON_STEP = 0.0, 0.08
NLAT, NLON = 4501, 4500


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


def site_index(lat, lon):
    """站点经纬度 → HYCOM 全球格点索引 (lat 自 −80 起 0.04°; lon 自 0 起 0.08°)。"""
    i = int(round((lat - HLAT0) / HLAT_STEP))
    j = int(round((lon % 360.0) / HLON_STEP))
    i = int(np.clip(i, 0, NLAT - 1))
    j = int(np.clip(j, 0, NLON - 1))
    return i, j


# ---------------- ETOPO ----------------
def fetch_etopo(lat, lon, half=0.30, stride=1):
    """返回 (lat[], lon[], z[ny, nx]), z 为高程(负=水下)。stride=1 即原生 15″。"""
    q = "z[(%.4f):%d:(%.4f)][(%.4f):%d:(%.4f)]" % (
        lat - half, stride, lat + half, lon - half, stride, lon + half)
    url = config.ENDPOINTS["etopo"] + "?" + urllib.parse.quote(q, safe="():,.-")
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
    return _get(config.ENDPOINTS["hycom_dap"] + "?" + expr)


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
    """40 层 T/S/u/v 剖面 (取最近格点)。"""
    i, j = site_index(lat, lon)
    i0, i1 = max(i - 2, 0), min(i + 2, NLAT - 1)
    j0, j1 = max(j - 2, 0), min(j + 2, NLON - 1)
    latv = _section(_asc("lat[%d:1:%d]" % (i0, i1)), "lat[")
    lonv = _section(_asc("lon[%d:1:%d]" % (j0, j1)), "lon[")
    ii = i0 + int(np.argmin(np.abs(latv - lat)))
    jj = j0 + int(np.argmin(np.abs(lonv - (lon % 360.0))))   # 经度需按 0–360 比较
    lon_out = float(lonv[jj - j0])
    if lon_out > 180.0:
        lon_out -= 360.0
    dz = _section(_asc("depth[0:1:39]"), "depth[")
    out = {"depth": dz, "lat": float(latv[ii - i0]), "lon": lon_out}
    for v in ("water_temp", "salinity", "water_u", "water_v"):
        out[v] = _section(_asc("%s[0][0:1:%d][%d][%d]" % (v, len(dz) - 1, ii, jj)), v + ".")
    return out


def _fetch_grid(expr, ny, nx):
    """取回单个二维网格 (一段时间/层) → (ny, nx)。"""
    txt = _asc(expr)
    body = txt.split("---------------------------------------------")[-1].split("\n\n")[0]
    vals = []
    for ln in body.split("\n"):
        toks = [t.strip() for t in ln.split(",")]
        if not toks or not toks[0]:
            continue
        if not _isf(toks[0]):
            toks = toks[1:]
        vals += [float(t) for t in toks if _isf(t)]
    if len(vals) != ny * nx:
        raise ValueError("网格尺寸不符: %d vs %d" % (len(vals), ny * nx))
    return np.array(vals).reshape(ny, nx)


def fetch_field(lat_center=None, lon_center=None, lev=(0, 19, 24, 27, 32),
                lat_half=0.4, lon_half=0.8):
    """若干深度层的二维流速场 (单时刻)。"""
    lat_center = config.SITE["lat"] if lat_center is None else lat_center
    lon_center = config.SITE["lon"] if lon_center is None else lon_center
    i, j = site_index(lat_center, lon_center)
    ni = int(round(lat_half / HLAT_STEP))
    nj = int(round(lon_half / HLON_STEP))
    lat_idx = (max(i - ni, 0), min(i + ni, NLAT - 1))
    lon_idx = (max(j - nj, 0), min(j + nj, NLON - 1))
    latv = _section(_asc("lat[%d:1:%d]" % lat_idx), "lat[")
    lonv = _section(_asc("lon[%d:1:%d]" % lon_idx), "lon[")
    lonv = np.where(lonv > 180.0, lonv - 360.0, lonv)
    dz = _section(_asc("depth[0:1:39]"), "depth[")
    res = {"lat": latv, "lon": lonv, "depth": np.array([dz[L] for L in lev])}
    ny, nx = len(latv), len(lonv)
    for name in ("water_u", "water_v"):
        for L in lev:
            expr = "%s[0][%d][%d:1:%d][%d:1:%d]" % (
                name, L, lat_idx[0], lat_idx[1], lon_idx[0], lon_idx[1])
            res["%s_%d" % (name, L)] = _fetch_grid(expr, ny, nx)
    return res


# ---------------- HYCOM 四维场 (空间 × 深度 × 时间) ----------------
def _get_retry(url, tries=12, timeout=20):
    """短超时 + 多次重试: 该服务器偶发挂起, 正常请求 2–5 s 内返回。"""
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read().decode("utf-8", "ignore")
            if len(data) > 200:
                return data
            last = RuntimeError("响应过短 (%d 字节)" % len(data))
        except Exception as e:      # noqa: BLE001
            last = e
    raise last


def fetch_field4d(lat_center=None, lon_center=None,
                  lat_half=0.38, lon_half=0.64,
                  target_depths=(0, 100, 300, 500, 700, 1000),
                  t_end_h=96.0, t_stride=2, workers=8):
    """抓取「多时刻 × 多层 × 区域」的四维流速场。"""
    from concurrent.futures import ThreadPoolExecutor

    def asc_fast(expr, tries=12):
        url = config.ENDPOINTS["hycom_dap"] + "?" + urllib.parse.quote(expr, safe="():,.-[]")
        return _get_retry(url, tries=tries)

    lat_center = config.SITE["lat"] if lat_center is None else lat_center
    lon_center = config.SITE["lon"] if lon_center is None else lon_center
    i, j = site_index(lat_center, lon_center)
    ni = int(round(lat_half / HLAT_STEP))
    nj = int(round(lon_half / HLON_STEP))
    lat_idx = (max(i - ni, 0), min(i + ni, NLAT - 1))
    lon_idx = (max(j - nj, 0), min(j + nj, NLON - 1))

    tvals = _section(asc_fast("time[0:1:60]"), "time[")
    ti = np.where(tvals <= t_end_h + 1e-6)[0]
    ti = ti[::max(1, int(t_stride))]
    dz = _section(asc_fast("depth[0:1:39]"), "depth[")
    levs = sorted({int(np.argmin(np.abs(dz - d))) for d in target_depths})
    latv = _section(asc_fast("lat[%d:1:%d]" % lat_idx), "lat[")
    lonv = _section(asc_fast("lon[%d:1:%d]" % lon_idx), "lon[")
    lonv = np.where(lonv > 180.0, lonv - 360.0, lonv)     # 统一到 −180..180
    ny, nx, nt = len(latv), len(lonv), len(ti)

    tasks = {}
    step = int(ti[1] - ti[0]) if len(ti) > 1 else 1
    for name in ("water_u", "water_v"):
        for L in levs:
            expr = "%s[%d:%d:%d][%d][%d:1:%d][%d:1:%d]" % (
                name, int(ti[0]), step, int(ti[-1]), L, lat_idx[0], lat_idx[1],
                lon_idx[0], lon_idx[1])
            tasks[(name, L)] = expr

    # HYCOM 'latest' 的时间轴起点会随日期滚动; 统一平移到从 0 h 开始
    out = {"time_h": tvals[ti] - float(tvals[ti][0]),
           "depth": np.array([dz[L] for L in levs]),
           "lat": latv, "lon": lonv,
           "lat_idx": np.array(lat_idx), "lon_idx": np.array(lon_idx)}
    cubes = {name: np.full((nt, len(levs), ny, nx), np.nan) for name in ("water_u", "water_v")}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(lambda e=expr: asc_fast(e)): key for key, expr in tasks.items()}
        for n, fut in enumerate(futs, 1):
            name, L = futs[fut]
            cube = cubes[name]
            k = levs.index(L)
            cube[:, k] = _grid_values(fut.result(), nt, ny, nx)
            print("      %2d/%d %s lev=%d ok" % (n, len(futs), name, L))
    out.update(cubes)
    return out


def _grid_values(txt, nt, ny, nx):
    """解析 DAP2 多维网格段 -> (nt, ny, nx) 数组。"""
    body = txt.split("---------------------------------------------")[-1]
    sec = body.split("\n\n")[0]
    vals = []
    for ln in sec.split("\n"):
        toks = [t.strip() for t in ln.split(",")]
        for t in toks:
            if t and _isf(t):
                vals.append(float(t))
    arr = np.array(vals)
    if arr.size != nt * ny * nx:
        raise ValueError("网格尺寸不符: 得到 %d, 期望 %d" % (arr.size, nt * ny * nx))
    return arr.reshape(nt, ny, nx)


# ---------------- 落盘 ----------------
def _tagged(name):
    tag = getattr(config, "SITE_TAG", None)
    return name + (("_" + tag) if tag else "")


def _dump(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    name = _tagged(name)
    if isinstance(obj, dict):
        np.savez(os.path.join(CACHE, name + ".npz"),
                 **{k: v for k, v in obj.items() if isinstance(v, (np.ndarray, float, int))})
        meta = {k: v for k, v in obj.items() if isinstance(v, str)}
        if meta:
            with open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
    else:
        np.savez(os.path.join(CACHE, name + ".npz"), **obj)


def main(tag=None):
    """地形 + 剖面 + 二维流场 (当前海域)。"""
    if tag:
        config.set_site(tag)
    lat, lon = config.SITE["lat"], config.SITE["lon"]
    print("[1/3] ETOPO 局部地形 (%s: %.4f, %.4f) ..." % (tag or "med", lat, lon))
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


def main4d(tag=None):
    """四维流场 (多时刻 × 多层 × 区域) — 当前海域。"""
    import time as _t
    if tag:
        config.set_site(tag)
    t0 = _t.time()
    print("HYCOM 四维流场抓取 (%s) ..." % (tag or "med"))
    fld = fetch_field4d()
    _dump("hycom4d", fld)
    print("      时刻 %d 个 (%.0f–%.0f h), 层 %d 个 (%s)"
          % (len(fld["time_h"]), fld["time_h"][0], fld["time_h"][-1],
             len(fld["depth"]), np.array2string(fld["depth"], precision=0)))
    print("      网格 %d × %d, 有效值 %.1f%%"
          % (len(fld["lat"]), len(fld["lon"]),
             100.0 * np.mean(np.isfinite(fld["water_u"]))))
    print("      用时 %.1f s -> cache/hycom4d%s.npz"
          % (_t.time() - t0, ("_" + tag) if tag else ""))


if __name__ == "__main__":
    main()
