<#
.SYNOPSIS
  第 1 步：为当前 Windows 用户设置 Ollama 环境变量。
.DESCRIPTION
  OLLAMA_HOST  = 0.0.0.0:11434   让 Docker 容器（RAGFlow）能通过 host.docker.internal 访问宿主机上的 Ollama
  OLLAMA_MODELS = 模型存放目录    建议放在 C 盘以外，bge-large-zh 约 0.7 GB，7B 对话模型约 4.7 GB
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\01_set_ollama_env.ps1 -ModelsDir "D:\ollama\models" -Restart
#>
param(
    [string]$ModelsDir = "D:\ollama\models",
    [string]$OllamaHost = "0.0.0.0:11434",
    [switch]$Restart
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
    Write-Host "已创建模型目录: $ModelsDir"
}

[Environment]::SetEnvironmentVariable("OLLAMA_HOST", $OllamaHost, "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelsDir, "User")
$env:OLLAMA_HOST = $OllamaHost
$env:OLLAMA_MODELS = $ModelsDir

Write-Host "OLLAMA_HOST   = $([Environment]::GetEnvironmentVariable('OLLAMA_HOST','User'))"
Write-Host "OLLAMA_MODELS = $([Environment]::GetEnvironmentVariable('OLLAMA_MODELS','User'))"

if ($Restart) {
    # Ollama 只在启动时读取环境变量，修改后必须重启托盘程序和后台服务
    Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    $app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $app) {
        Start-Process $app
        Write-Host "已重启 Ollama：$app"
    } else {
        Write-Warning "未找到 $app，请手动从开始菜单启动 Ollama。"
    }
} else {
    Write-Host "请退出任务栏中的 Ollama 图标后重新打开，环境变量才会生效（或加 -Restart 参数）。"
}

Write-Host "提示：OLLAMA_HOST=0.0.0.0 会让局域网内其他机器也能访问 11434 端口，请只在可信网络中使用，不要暴露到公网。"
