# -*- coding: utf-8 -*-
"""数据下载: ETOPO 地形 (ERDDAP) + HYCOM 剖面与二维场 (OPeNDAP DAP2 ASCII)。"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")


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
    """40 层 T/S/u/v。返回 dict。"""
    latv = _section(_asc("lat[2900:1:3010]"), "lat[")
    lonv = _section(_asc("lon[230:1:280]"), "lon[")
    i = 2900 + int(np.argmin(np.abs(latv - lat)))
    j = 230 + int(np.argmin(np.abs(lonv - lon)))
    dz = _section(_asc("depth[0:1:39]"), "depth[")
    out = {"depth": dz, "lat": float(latv[i - 2900]), "lon": float(lonv[j - 230])}
    for v in ("water_temp", "salinity", "water_u", "water_v"):
        out[v] = _section(_asc("%s[0][0:1:%d][%d][%d]" % (v, len(dz) - 1, i, j)), v + ".")
    return out


def fetch_field(lev=(0, 19, 24, 27, 32), lat_idx=(2925, 2940), lon_idx=(256, 266)):
    """若干深度层的二维流速场。lev 是 HYCOM 深度索引。"""
    latv = _section(_asc("lat[%d:1:%d]" % lat_idx), "lat[")
    lonv = _section(_asc("lon[%d:1:%d]" % lon_idx), "lon[")
    dz = _section(_asc("depth[0:1:39]"), "depth[")
    res = {"lat": latv, "lon": lonv, "depth": np.array([dz[L] for L in lev])}
    ny, nx = len(latv), len(lonv)
    for name in ("water_u", "water_v"):
        for L in lev:
            seq = _asc("%s[0][%d][%d:1:%d][%d:1:%d]" % (name, L, lat_idx[0], lat_idx[1],
                                                       lon_idx[0], lon_idx[1]))
            body = seq.split("---------------------------------------------")[-1].split("\n\n")[0]
            vals = []
            for ln in body.split("\n"):
                toks = [t.strip() for t in ln.split(",")]
                if not toks or not toks[0]:
                    continue
                if not _isf(toks[0]):
                    toks = toks[1:]
                vals += [float(t) for t in toks if _isf(t)]
            res["%s_%d" % (name, L)] = np.array(vals).reshape(ny, nx)
    return res


# ---------------- 落盘 ----------------
def _dump(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    if isinstance(obj, dict):
        np.savez(os.path.join(CACHE, name + ".npz"),
                 **{k: v for k, v in obj.items() if isinstance(v, (np.ndarray, float, int))})
        meta = {k: v for k, v in obj.items() if isinstance(v, str)}
        if meta:
            with open(os.path.join(CACHE, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
    else:
        np.savez(os.path.join(CACHE, name + ".npz"), **obj)


def main():
    lat, lon = config.SITE["lat"], config.SITE["lon"]
    print("[1/3] ETOPO 局部地形 ...")
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


if __name__ == "__main__":
    main()
