# -*- coding: utf-8 -*-
"""命令行入口: python run.py [check|fetch|analyze|figures|all]"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("check", "all"):
        import eos
        eos.check(); eos.check_speed()
    if cmd in ("fetch", "all"):
        import fetch
        fetch.main()
    if cmd in ("analyze", "all"):
        import analyze
        analyze.main()
    if cmd in ("figures", "all"):
        import figures
        figures.main()
    if cmd not in ("check", "fetch", "analyze", "figures", "all"):
        print(__doc__)


if __name__ == "__main__":
    main()
