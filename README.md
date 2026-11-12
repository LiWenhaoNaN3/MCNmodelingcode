# v2 —— 数据溯源完整的重建版

> **v3 进展（P1+P2 已完成，待审）**：在 v2 的实测数据与 EOS 基础上，新增概率定位模型
> （失效树混合先验 + 垂向解析动力学 + 20 000 样本蒙特卡洛集合 + 后验/包含域），
> 流场升级为**四维（时间 × 深度 × 空间）**，压溃改为**事后统计**。
> 方案见 `docs/建模方案_v3.md`，本轮结果与待确认事项见 `docs/阶段P2结果.md`。
> 新命令：`python run.py p2`（自检 + 集合预报 + 后验 + 图 F3–F6，约 15 s，离线）；
> `python run.py field`（图 G1 四维流场 / G2 坡度与滑移，约 4 s）；
> `python run.py fetch4d`（抓取 HYCOM 四维场，约 2 分钟，已入库）。
> 新增模块：`src/data.py`、`src/locate.py`、`src/posterior.py`、`src/plotmpl.py`、
> `src/figs_p2.py`、`src/figs_field.py`。

## 一句话

所有环境量来自 ETOPO / HYCOM 实测; 查不到的量 (Δρ、μ_s、z0、W、v、w0) **不赋值**, 留作函数参数。

## 目录

| 路径 | 作用 |
|---|---|
| `run.py` | 仓库根入口（调用 `src/run.py`） |
| `src/config.py` | 唯一参数出处，每项标注来源 |
| `src/eos.py` | EOS-80 密度 + Chen-Millero 声速，含两套自检 |
| `src/fetch.py` | 下载 ETOPO 地形 / HYCOM 剖面 / HYCOM 二维场 |
| `src/analyze.py` | 全部推导与分析（密度、判据、滑移、漂移、条带） |
| `src/figures.py` | 四张报告用图 |
| `src/plotlib.py` | 仅用 Pillow 的极简绘图库 |
| `src/run.py` | 实际 CLI 实现 |
| `build/build_single.py` | 从 `src/` 的 7 个模块生成根目录 `model.py` |
| `build/make_code_md.py` | 生成 `docs/建模代码.md` |
| `model.py` | 单文件版（**生成物，勿手改**），可直接 `python model.py all` |
| `docs/代码解析.md` | 逐模块代码说明与常见错误 |
| `docs/数据来源.md` | 数据溯源、符号参数、校验记录 |
| `docs/建模代码.md` | 源码全文转 md（由 `build/make_code_md.py` 生成） |
| `docs/报告.md` | 正文报告（**待补**；当前仓库只有代码与两份说明） |
| `cache/` | ETOPO/HYCOM 实测数据（**已入库**，离线可复现） |
| `figs/` | 四张报告图（**已入库**） |

## 用法

```bash
pip install -r requirements.txt

python run.py check       # 状态方程自检
python run.py fetch       # 下载数据到 cache/（仅首次/换站时需要）
python run.py analyze     # 全部计算, 打印所有表
python run.py figures     # 出图到 figs/
python run.py all         # 依次执行

# 也可以用模块方式，或直接用单文件
python -m src.run check
python model.py check
```

## 数据溯源

见 `src/config.py` 的注释与 `docs/数据来源.md`。已弃用 Mackenzie 声速公式。

## 验证记录（本轮实测）

从网络重新下载一遍，再跑分析与出图；除漂移一项随 HYCOM 时次更新外，其余与旧记录一致：

| 步骤 | 输出 |
|---|---|
| run.py fetch | ETOPO 145×145；HYCOM 剖面 33/40 层有效；二维场 16×11×5 |
| run.py check | EOS-80 最大偏差 0.002 kg/m³；Chen-Millero 最大偏差 0.015 m/s |
| run.py analyze | 站点 37.2979N/20.9021E，水深 876 m，坡度 21.5° |
| | M1 阈值 μ_s ≥ **0.394** |
| | μ_s=0.40 → 滑行 0 m；0.35 → 1398 m（落点 1532 m）；0.20 → 1735 m（落点 1619 m）|
| | Δρ=−0.20 → 平衡深度 107 m → 24 h 漂移 6.2 km |
| run.py figures | 四张图全部生成 |

> 注：旧记录为“121 m / 10.5 km”（上一时次 HYCOM）；当前 `cache/` 复跑为“107 m / 6.2 km”。以当前 cache 为准；换时次需重跑并更新本表。

## 复现方式

```bash
cd MCNmodelingcode
python run.py all
```

约 3 分钟（主要是两个网络下载；`cache/` 已入库时 `analyze`/`figures` 可离线秒级完成）。
