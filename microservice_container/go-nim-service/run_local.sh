#!/bin/bash
# run_local.sh - 本地執行 Go NIM Service（免 Docker）
# ================================================
set -e

cd "$(dirname "$0")"

echo "📦 檢查 Go 環境..."
if ! command -v go &> /dev/null; then
    echo "❌ 請先安裝 Go (https://go.dev/dl/)"
    exit 1
fi

echo "🚀 啟動 Go NIM Service..."
go run main.go

