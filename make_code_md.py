# -*- coding: utf-8 -*-
"""把建模源码原样转成 Markdown（代码块形式），供附录/审阅使用。

生成的 md 与源码逐字一致，不做任何改动；每次改代码后重跑本脚本即可同步。
"""
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FENCE = chr(96) * 3

FILES = [
    ("config.py", "全部输入参数的唯一出处（站点、机型、端点、符号参数）"),
    ("eos.py", "海水密度（EOS-80）与声速（Chen-Millero），含两套自检"),
    ("fetch.py", "ETOPO / HYCOM 数据下载与解析"),
    ("analyze.py", "剖面派生、三条判据、下滑积分、漂移与条带"),
    ("figures.py", "四张报告插图"),
    ("plotlib.py", "仅依赖 Pillow 的极简绘图库"),
    ("run.py", "命令行入口"),
]


def read_src(name):
    """读源码并去掉末尾换行，行数以去掉后的为准（全文件统一口径）。"""
    return open(os.path.join(HERE, name), encoding="utf-8").read().rstrip("\n")


def main():
    total = 0
    parts = []
    parts.append("# 建模代码（v2）\n\n")
    parts.append("> 2024 MCM Problem B 重建版 · 7 个源文件\n>\n")
    parts.append("> 本文件由 make_code_md.py 从源码自动生成，内容与源码逐字一致。\n>\n")
    parts.append("> 生成时间：%s\n" % time.strftime("%Y-%m-%d %H:%M"))
    parts.append("\n---\n\n## 目录\n\n")
    parts.append("| # | 文件 | 行数 | 职责 |\n|---|---|---|---|\n")
    counts = []
    for i, (name, desc) in enumerate(FILES, 1):
        n = len(read_src(name).split("\n"))
        counts.append(n)
        total += n
        parts.append("| %d | %s | %d | %s |\n" % (i, name, n, desc))
    parts.append("| | **合计** | **%d** | |\n" % total)

    for i, (name, desc) in enumerate(FILES, 1):
        src = read_src(name)
        n = len(src.split("\n"))
        parts.append("\n---\n\n## %d. %s\n\n" % (i, name))
        parts.append("**%s** · %d 行\n\n" % (desc, n))
        parts.append(FENCE + "python\n")
        parts.append(src + "\n")
        parts.append(FENCE + "\n")

    out = os.path.join(HERE, "建模代码.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("已生成 %s（%d 行源码，%d 个文件）" % (out, total, len(FILES)))


if __name__ == "__main__":
    main()
