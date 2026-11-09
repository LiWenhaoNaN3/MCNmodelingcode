# v2 —— 数据溯源完整的重建版

## 一句话

所有环境量来自 ETOPO / HYCOM 实测; 查不到的量 (Δρ、μ_s、z₀、W、v、w₀) **不赋值**, 留作函数参数。

## 目录

| 文件 | 作用 |
|---|---|
| config.py | 唯一参数出处, 每项标注来源 |
| eos.py | EOS-80 密度 + Chen-Millero 声速, 含两套自检 |
| fetch.py | 下载 ETOPO 地形 / HYCOM 剖面 / HYCOM 二维场 |
| analyze.py | 全部推导与分析 (密度、判据、滑移、漂移、条带) |
| figures.py | 四张报告用图 |
| plotlib.py | 仅用 Pillow 的极简绘图库 |
| run.py | 命令行入口 |
| 报告.md | 正文：6 节 + 附录 |
| 代码解析.md | 逐模块代码说明与常见错误 |
| 数据来源.md | 数据溯源、符号参数、校验记录 |
| 建模代码.md | 源码全文转 md（由 make_code_md.py 生成）|
| model.py | 单文件版，可直接 python model.py all |
| build_single.py | 从 7 个模块生成 model.py |

## 用法

    python run.py check       # 状态方程自检
    python run.py fetch       # 下载数据到 cache/
    python run.py analyze     # 全部计算, 打印所有表
    python run.py figures     # 出图到 figs/
    python run.py all         # 依次执行

## 数据溯源

见 config.py 的注释与报告第 1 节。已弃用: Mackenzie 声速公式。

---

## 验证记录（本轮实测）

从网络重新下载一遍，再跑分析与出图，全部数字与之前一致：

| 步骤 | 输出 |
|---|---|
| run.py fetch | ETOPO 145×145；HYCOM 剖面 33/40 层有效；二维场 16×11×5 |
| run.py check | EOS-80 最大偏差 0.002 kg/m³；Chen-Millero 最大偏差 0.015 m/s |
| run.py analyze | 站点 37.2979N/20.9021E，水深 876 m，坡度 21.5° |
| | M1 阈值 μ_s ≥ **0.394** |
| | μ_s=0.40 → 滑行 0 m；0.35 → 1398 m（落点 1532 m）；0.20 → 1735 m（落点 1619 m）|
| | Δρ=−0.20 → 平衡深度 121 m → 24 h 漂移 10.5 km |
| run.py figures | 四张图全部生成 |

## 文件清单

| 文件 | 行数级 | 说明 |
|---|---|---|
| config.py | 60 | 全部参数与来源 |
| eos.py | 110 | EOS-80 + Chen-Millero，含自检 |
| fetch.py | 175 | 三个下载函数 + 落盘 |
| analyze.py | 240 | Profile / Terrain / vehicle / equilibrium_depth / 全部表格 |
| figures.py | 210 | 四张图 |
| plotlib.py | 120 | Pillow 绘图库 |
| run.py | 30 | 命令行入口 |
| cache/ | — | 3 个 npz（地形、剖面、二维场）|
| figs/ | — | 4 张 png |

## 复现方式

    cd mcm2024b/v2
    python run.py all

约 3 分钟（主要是两个网络下载）。
