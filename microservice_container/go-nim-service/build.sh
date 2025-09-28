#!/bin/bash
# build.sh - 自動化建置 & 推送 Go NIM Service 容器

set -e

IMAGE_NAME="go-nim-service"
TAG="latest"
DOCKER_USER="your-dockerhub-username"  # ⚠️ 改成你自己的帳號

echo "🚀 建立 Docker 映像檔..."
docker build -t $DOCKER_USER/$IMAGE_NAME:$TAG .

echo "🔑 登入 Docker Registry..."
docker login

echo "⬆️ 推送映像檔到 Registry..."
docker push $DOCKER_USER/$IMAGE_NAME:$TAG

echo "✅ 完成！執行方式："
echo "docker run --rm -it $DOCKER_USER/$IMAGE_NAME:$TAG"
