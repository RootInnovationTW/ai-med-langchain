#!/bin/bash
# git_update_go.sh
# 功能: 推送 microservice_container 到 GitHub

set -e

PROJECT_DIR="microservice_container"

# 確認 commit 訊息
if [ -z "$1" ]; then
  echo "❌ 請輸入 commit 訊息"
  echo "👉 用法: ./git_update_go.sh 'Add Go microservice_container with Dockerfile'"
  exit 1
fi

COMMIT_MSG="$1"

# 進入 repo 根目錄
cd ~/ai-med-langchain

echo "📂 當前 Git 狀態:"
git status

echo "➕ 加入 microservice_container 檔案..."
git add $PROJECT_DIR

echo "📝 建立 commit: $COMMIT_MSG"
git commit -m "$COMMIT_MSG" || echo "⚠️ 沒有變更需要 commit"

echo "📥 更新遠端版本 (rebase)..."
git pull --rebase origin main || echo "⚠️ rebase 衝突，請手動解決"

echo "⬆️ 推送到 GitHub..."
git push origin main

echo "✅ microservice_container 已成功推送到 GitHub！"
