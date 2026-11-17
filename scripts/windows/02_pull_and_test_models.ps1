<#
.SYNOPSIS
  第 2 步：下载 Embedding 模型和对话模型，并测试 Ollama 接口。
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\02_pull_and_test_models.ps1
  powershell -ExecutionPolicy Bypass -File .\02_pull_and_test_models.ps1 -ChatModel "qwen2.5:3b"
#>
param(
    [string]$EmbedModel = "quentinz/bge-large-zh-v1.5",
    [string]$ChatModel = "qwen2.5:7b",
    [string]$OllamaUrl = "http://localhost:11434",
    [int]$ExpectedDim = 1024
)
$ErrorActionPreference = "Stop"

function Invoke-OllamaJson([string]$Path, [hashtable]$Body) {
    $json = $Body | ConvertTo-Json -Depth 5
    # 用 UTF-8 字节发送，避免 Windows PowerShell 5.1 把中文按本地编码发送
    Invoke-RestMethod -Uri "$OllamaUrl$Path" -Method Post -ContentType "application/json" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 600
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "找不到 ollama 命令，请先从 https://ollama.com/download 安装 Ollama。"
}

Write-Host "== 1. 下载模型 =="
foreach ($m in @($EmbedModel, $ChatModel)) {
    ollama pull $m
    if ($LASTEXITCODE -ne 0) { throw "下载 $m 失败，请检查网络或模型名。" }
}

Write-Host "`n== 2. 已安装模型 =="
ollama list
ollama show $EmbedModel

Write-Host "`n== 3. 测试 Embedding 接口 =="
$r = Invoke-OllamaJson "/api/embed" @{ model = $EmbedModel; input = "检索增强生成通过检索外部知识来辅助大语言模型回答问题。" }
$dim = $r.embeddings[0].Count
if ($dim -eq $ExpectedDim) { Write-Host "向量维度 = $dim ，符合预期。" -ForegroundColor Green }
else { Write-Warning "向量维度 = $dim ，预期 $ExpectedDim，请确认模型名是否正确。" }

Write-Host "`n== 4. 测试对话接口 =="
$c = Invoke-OllamaJson "/api/chat" @{
    model = $ChatModel; stream = $false
    messages = @(@{ role = "user"; content = "用一句话解释什么是检索增强生成。" })
}
Write-Host $c.message.content

Write-Host "`n全部完成。在 RAGFlow 中填写的模型名必须与 ollama list 中的名称完全一致："
Write-Host "  Embedding: $EmbedModel"
Write-Host "  Chat     : $ChatModel"
