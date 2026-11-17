<#
.SYNOPSIS
  第 4 步：检查 RAGFlow 各容器状态，并从 RAGFlow 容器内部测试能否访问宿主机上的 Ollama。
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\04_check_ragflow.ps1
#>
param(
    [int]$WebPort = 80,
    [string]$OllamaFromDocker = "http://host.docker.internal:11434"
)

Write-Host "== 1. 容器状态 =="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-String -Pattern "NAMES|ragflow|es01|es-01|mysql|minio|redis|valkey|kvrocks|nats|infinity"

Write-Host "`n== 2. 健康检查 =="
try {
    $r = Invoke-WebRequest -Uri "http://localhost:$WebPort/api/v1/system/healthz" -UseBasicParsing -TimeoutSec 10
    Write-Host "healthz HTTP $($r.StatusCode)  $($r.Content)" -ForegroundColor Green
} catch { Write-Warning "healthz 访问失败：$($_.Exception.Message)" }

Write-Host "`n== 3. 从 RAGFlow 容器访问 Ollama =="
# 1.0 版服务名为 ragflow-cpu（GPU 版为 ragflow-gpu）；旧版容器名为 ragflow-server
$name = docker ps --filter "label=com.docker.compose.service=ragflow-cpu" --format "{{.Names}}" | Select-Object -First 1
if (-not $name) { $name = docker ps --filter "label=com.docker.compose.service=ragflow-gpu" --format "{{.Names}}" | Select-Object -First 1 }
if (-not $name) { $name = docker ps --filter "name=ragflow-server" --format "{{.Names}}" | Select-Object -First 1 }
if (-not $name) { Write-Warning "没找到 RAGFlow 主容器，请先运行 03_deploy_ragflow.ps1。"; exit 1 }
Write-Host "RAGFlow 主容器：$name"
docker exec $name curl -s -m 10 "$OllamaFromDocker/api/tags"
if ($LASTEXITCODE -ne 0) {
    Write-Warning @"
容器内无法访问 $OllamaFromDocker 。请依次检查：
  1) OLLAMA_HOST 是否为 0.0.0.0:11434，并且修改后已重启 Ollama；
  2) 宿主机上 netstat -ano | findstr 11434 是否显示 0.0.0.0:11434；
  3) Windows 防火墙是否放行 11434 端口（入站规则）。
"@
} else {
    Write-Host "`n容器可以访问 Ollama，RAGFlow 中 Base URL 填写 $OllamaFromDocker 即可。" -ForegroundColor Green
}

Write-Host "`n== 4. 最近日志（主容器最后 30 行） =="
docker logs --tail 30 $name
