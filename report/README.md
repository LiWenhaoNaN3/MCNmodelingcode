# 2024 MCM Problem B 报告（中文稿）

## 文件
- `main.tex` —— 主文件（摘要页、目录、正文、参考文献、附录）
- `sections/*.tex` —— 各章节
- `../figs/` —— 全部图件（PNG 与 PDF，`\graphicspath` 指向仓库 `figs/`）
- `main.pdf` —— 编译结果（25 页）

## 编译
需要 TeX Live（含 ctex、xelatex）。在 PowerShell 中：

```powershell
cd report
$env:PATH = "G:\texlive\texlive\2026\bin\windows;" + $env:PATH
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex   # 第二遍解析目录/交叉引用
```

或直接运行 `build.ps1`。

## 结构（25 页，A4）

| 页 | 内容 |
|---|---|
| 1 | Summary Sheet（摘要页） |
| 2 | 目录 |
| 3–4 | §1 问题重述与建模思路 |
| 4–5 | §2 模型假设与符号 |
| 5–8 | §3 数据与参数 |
| 8–10 | §4 定位模型（含验证、压溃事后统计） |
| 10–11 | §5 位置不确定性（A50/A95） |
| 11–13 | §6 搜索模型（探测/规划/贝叶斯） |
| 13–15 | §7 七种单船装备组合（含响应延迟） |
| 15–18 | §8 多资产协同（15 组合，含协调与路径规划） |
| 18–20 | §9 外推一：加勒比海 |
| 20–21 | §10 模型验证与灵敏度 |
| 21–22 | §11 结论与建议 |
| 22–23 | 致希腊政府备忘录（2 页） |
| 24 | 参考文献 |
| 25 | 附录 A：AI 使用报告；附录 B：代码结构与复现 |

> 数值结果全部来自仓库代码的实际运行（data/*.csv、cache/*.npz），
> 随机种子固定（seed=7），可完全复现。
