#!/bin/bash
# git_update_go.sh
# 功能：推送 Go microservice_container 相關程式到 GitHub

set -e

# === 參數設定 ===
PROJECT_DIR="microservice_container/go-nim-service"

# 確認 commit 訊息
if [ -z "$1" ]; then
  echo "❌ 請輸入 commit 訊息"
  echo "👉 用法: ./git_update_go.sh 'Add Go microservice with Dockerfile'"
  exit 1
fi

COMMIT_MSG="$1"

# === 進入專案資料夾 ===
cd $PROJECT_DIR

# === Git 操作 ===
echo "⚙️ Git add Go microservice 檔案..."
git add main.go run_local.sh Dockerfile Makefile

echo "💾 Git commit..."
git commit -m "$COMMIT_MSG" || echo "⚠️ 沒有變更需要 commit"

echo "🔄 Git pull --rebase..."
git pull --rebase origin main || echo "⚠️ pull rebase 發生衝突，請手動解決"

echo "⬆️ Git push..."
git push origin main

echo "✅ Go microservice 已更新並推送到 GitHub！"

