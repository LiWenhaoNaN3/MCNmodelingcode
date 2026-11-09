# -*- coding: utf-8 -*-
"""把 src/ 的 7 个模块合并成单文件 model.py。

做四类结构化改写:
  1. 去掉对本地模块的 import（含 `from . import ...`），并把
     config./eos./analyze./plotlib. 前缀去掉
  2. 各模块的 main() 改名, 避免冲突, 最后统一由 cli() 调度
  3. figures.py 顶层的 PROF/TER/VEH 改为惰性初始化(否则 check 模式也要 cache)
  4. 把各模块的 ROOT 改成生成文件头部的 HERE（生成物在仓库根）
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))   # build/
ROOT = os.path.dirname(HERE)                        # repo root
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "model.py")


def read(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        return f.read()


def fix_root(src):
    """生成物在仓库根, 所以 src/ 里的 ROOT=... 要改成 HERE。"""
    return src.replace(
        "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
        "ROOT = HERE")


def strip_tail(src):
    """删掉任意位置的 if __name__ 块(含其后缩进体), 而不是从该处切到文件尾。"""
    out, skipping = [], False
    for ln in src.split("\n"):
        if ln.strip().startswith('if __name__ == "__main__":'):
            skipping = True
            continue
        if skipping:
            if ln.startswith((" ", "\t")) or not ln.strip():
                continue
            skipping = False
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def strip_local_imports(src):
    out = []
    for ln in src.split("\n"):
        s = ln.strip()
        if re.match(r"^import (config|eos|analyze|fetch|figures|plotlib)\b", s):
            continue
        if re.match(r"^from (config|eos|analyze|fetch|figures|plotlib)\b", s):
            continue
        if re.match(r"^from \. import\b", s):
            continue
        if re.match(r"^from src\b", s):
            continue
        out.append(ln)
    return "\n".join(out)


def strip_prefix(src):
    for p in ("config.", "eos.", "analyze.", "plotlib."):
        src = src.replace(p, "")
    return src


HEADER = '''# -*- coding: utf-8 -*-
"""2024 MCM Problem B 重建版 —— 单文件模型

由 src/ 的 7 个模块合并而成（build/build_single.py 生成）。
全部环境量来自 ETOPO 2022 与 HYCOM GLBy0.08 实测数据;
查不到的量(Δρ, μ_s, z0, W, v, w0)不赋值, 留作函数参数。

用法:
    python model.py check      状态方程自检(无需数据)
    python model.py fetch      下载数据到 cache/
    python model.py analyze    全部计算, 打印所有表
    python model.py figures    出图到 figs/
    python model.py all        全流程
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FIGS = os.path.join(HERE, "figs")
'''

parts = [HEADER]

parts.append("\n\n# " + "=" * 70 + "\n# 1. config —— 全部输入参数的唯一出处\n# " + "=" * 70 + "\n\n")
parts.append(strip_tail(read("config.py")))

parts.append("\n\n# " + "=" * 70 + "\n# 2. eos —— 海水密度与声速\n# " + "=" * 70 + "\n\n")
parts.append(strip_tail(strip_local_imports(read("eos.py"))))

parts.append("\n\n# " + "=" * 70 + "\n# 3. fetch —— 数据下载\n# " + "=" * 70 + "\n\n")
f = fix_root(strip_tail(strip_local_imports(read("fetch.py"))))
f = strip_prefix(f).replace("def main():", "def fetch_main():")
parts.append(f)

parts.append("\n\n# " + "=" * 70 + "\n# 4. analyze —— 全部推导\n# " + "=" * 70 + "\n\n")
a = fix_root(strip_tail(strip_local_imports(read("analyze.py"))))
a = strip_prefix(a).replace("def main():", "def analyze_main():")
parts.append(a)

parts.append("\n\n# " + "=" * 70 + "\n# 5. plotlib —— 仅用 Pillow 的绘图库\n# " + "=" * 70 + "\n\n")
parts.append(strip_tail(strip_local_imports(read("plotlib.py"))))

parts.append("\n\n# " + "=" * 70 + "\n# 6. figures —— 四张图\n# " + "=" * 70 + "\n\n")
g = fix_root(strip_tail(strip_local_imports(read("figures.py"))))
g = strip_prefix(g)
g = g.replace("def main():", "def figures_main():")
g = g.replace(
    "PROF = Profile(z0=300.0)\nTER = Terrain()\nVEH = vehicle(float(np.interp(PROF.z0, PROF.z, PROF.rho)))",
    "PROF = None\nTER = None\nVEH = None\n\n\n"
    "def _ctx():\n"
    "    \"\"\"惰性构建上下文: check 模式不需要 cache 目录。\"\"\"\n"
    "    global PROF, TER, VEH\n"
    "    if PROF is None:\n"
    "        PROF = Profile(z0=300.0)\n"
    "        TER = Terrain()\n"
    "        VEH = vehicle(float(np.interp(PROF.z0, PROF.z, PROF.rho)))\n"
    "    return PROF, TER, VEH")
for fn in ("fig1_terrain", "fig2_profile", "fig3_drift", "fig4_band"):
    g = g.replace("def %s():\n" % fn, "def %s():\n    _ctx()\n" % fn)
parts.append(g)

parts.append('''

# ======================================================================
# 7. cli —— 命令行入口
# ======================================================================
def cli(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd in ("check", "all"):
        check(); check_speed()
    if cmd in ("fetch", "all"):
        fetch_main()
    if cmd in ("analyze", "all"):
        analyze_main()
    if cmd in ("figures", "all"):
        figures_main()
    if cmd not in ("check", "fetch", "analyze", "figures", "all"):
        print(__doc__)


if __name__ == "__main__":
    cli(sys.argv)
''')

src = "".join(parts)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(src)
print("已生成 %s, %d 行" % (OUT, len(src.split("\n"))))
