<#
.SYNOPSIS
  第 3 步：克隆 RAGFlow、切换到指定版本、用 Docker Compose 启动，并等待服务就绪。
.DESCRIPTION
  前提：已安装并启动 Docker Desktop（Docker >= 24.0.0，Compose >= v2.26.1）和 Git。
  建议先在 Docker Desktop → Settings → Resources → Advanced → Disk image location 中把镜像位置改到 C 盘以外。
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\03_deploy_ragflow.ps1 -InstallDir "D:\ragflow" -Tag "v1.0.0"
#>
param(
    [string]$InstallDir = "D:\ragflow",
    [string]$Tag = "v1.0.0",
    [int]$WebPort = 80,
    [int]$TimeoutMinutes = 15
)
$ErrorActionPreference = "Stop"

foreach ($cmd in "git", "docker") {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "找不到 $cmd，请先安装。" }
}
try { docker info *> $null } catch { }
if ($LASTEXITCODE -ne 0) { throw "Docker 引擎未运行，请先启动 Docker Desktop。" }
docker --version; docker compose version

# Elasticsearch 要求 vm.max_map_count >= 262144；Docker Desktop 需在它的 Linux 虚拟机里设置
# Windows PowerShell 5.1 在 Stop 模式下重定向原生命令的 stderr 会抛异常，所以包在 try 里
$sysctlOk = $false
try {
    wsl -d docker-desktop -u root sysctl -w vm.max_map_count=262144 2>&1 | Out-Null
    $sysctlOk = ($LASTEXITCODE -eq 0)
} catch { }
if (-not $sysctlOk) {
    Write-Warning "未能自动设置 vm.max_map_count。若 es01 容器反复重启，请按教程 2.1 节手动设置。"
}

if (-not (Test-Path (Join-Path $InstallDir ".git"))) {
    git clone https://github.com/infiniflow/ragflow.git $InstallDir
    if ($LASTEXITCODE -ne 0) { throw "git clone 失败，请检查网络。" }
}
Push-Location (Join-Path $InstallDir "docker")
try {
    git fetch --tags
    git checkout -f $Tag
    if ($LASTEXITCODE -ne 0) { throw "切换到 $Tag 失败，用 git tag -l 'v1.0*' 查看可用版本。" }
    Select-String -Path ".env" -Pattern "^RAGFLOW_IMAGE" | ForEach-Object { Write-Host "镜像: $($_.Line)" }

    # 已在 ragflow/docker 目录中，所以 -f 后面直接写 docker-compose.yml
    docker compose -f docker-compose.yml up -d
    if ($LASTEXITCODE -ne 0) { throw "docker compose 启动失败。" }
} finally { Pop-Location }

$url = "http://localhost:$WebPort/api/v1/system/healthz"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
Write-Host "等待 RAGFlow 就绪（首次启动需要下载镜像并初始化，可能要 5~15 分钟）..."
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        if ($resp.StatusCode -eq 200) { Write-Host "RAGFlow 已就绪：http://localhost:$WebPort" -ForegroundColor Green; exit 0 }
    } catch { }
    Start-Sleep -Seconds 10
}
Write-Warning "超时仍未就绪。请运行 04_check_ragflow.ps1 或 docker compose logs 查看原因。"
exit 1
