# -*- coding: utf-8 -*-
"""命令行入口: python run.py <命令>

主算例 (地中海): check|fetch|fetch4d|analyze|figures|locate|posterior|figures2|p2|
                upper|truth|search|searchfigs|delay|multi|multifigs|multidelay|multibeta
加勒比海外推:   fetchcarib|carib|caribmulti|caribfigs
"""
import sys

P2 = ("locate", "posterior", "figures2", "p2")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("check", "all"):
        from . import eos
        eos.check(); eos.check_speed()
    if cmd in ("fetch", "all"):
        from . import fetch
        fetch.main()
    if cmd == "fetch4d":
        from . import fetch
        fetch.main4d()
    if cmd in ("analyze", "all"):
        from . import analyze
        analyze.main()
    if cmd in ("figures", "all"):
        from . import figures
        figures.main()

    # ---------------- v3 / P1+P2 ----------------
    if cmd in ("selftest", "p2", "all"):
        from . import locate
        locate.self_test()
    if cmd in ("locate", "p2", "all"):
        from . import locate
        locate.main()
    if cmd in ("posterior", "p2", "all"):
        from . import posterior
        posterior.main()
    if cmd in ("figures2", "p2", "all"):
        from . import figs_p2
        figs_p2.main()
    if cmd in ("field", "p2"):
        from . import figs_field
        figs_field.main()

    # ---------------- 搜索 / 装备评估 (主算例) ----------------
    if cmd == "upper":
        from . import upper
        upper.generate()
        upper.summary()
    if cmd in ("truth", "search"):
        from . import search
        search.build_truth(force=(cmd == "truth"))
    if cmd in ("search", "searchfigs"):
        from . import search
        rows, curves, track = search.main()
        if cmd == "searchfigs":
            from . import figs_search
            figs_search.main()
    if cmd == "delay":
        from . import search
        search.delay_sensitivity()
    if cmd in ("multi", "multifigs"):
        from . import multiveh
        multiveh.run_all()
        if cmd == "multifigs":
            from . import figs_multi
            figs_multi.main()
    if cmd == "multidelay":
        from . import multiveh
        multiveh.delay_sensitivity()
    if cmd == "multibeta":
        from . import multiveh
        multiveh.tune_beta()

    # ---------------- 加勒比海外推 ----------------
    if cmd in ("fetchcarib", "carib"):
        from . import config, fetch
        config.set_site("carib")
        fetch.main("carib")
        fetch.main4d("carib")
    if cmd == "carib":
        from . import config, upper, search
        config.set_site("carib")
        upper.generate()
        upper.summary()
        search.build_truth(force=True, tag="carib")
        search.main(tag="carib")
    if cmd == "caribmulti":
        from . import multiveh
        multiveh.run_all(tag="carib")
    if cmd in ("caribfigs", "carib"):
        from . import figs_carib
        figs_carib.main()

    known = (("check", "fetch", "fetch4d", "analyze", "figures", "all", "selftest")
             + P2 + ("field", "upper", "truth", "search", "searchfigs", "delay",
                     "multi", "multifigs", "multidelay", "multibeta",
                     "fetchcarib", "carib", "caribmulti", "caribfigs"))
    if cmd not in known:
        print(__doc__)


if __name__ == "__main__":
    main()
