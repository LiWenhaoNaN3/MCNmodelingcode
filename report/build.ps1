# 编译 2024 MCM Problem B 中文报告 (xelatex ×2)
$ErrorActionPreference = "Stop"
$texbin = "G:\texlive\texlive\2026\bin\windows"
$env:PATH = "$texbin;" + $env:PATH
Push-Location $PSScriptRoot
try {
    xelatex -interaction=nonstopmode main.tex | Out-Null
    xelatex -interaction=nonstopmode main.tex | Out-Null
    Write-Host "编译完成 -> $PSScriptRoot\main.pdf"
} finally {
    Pop-Location
}
