#!/bin/bash
# ================================================
# setup_go_microservice.sh
# 建立 microservice_container/go-nim-service 專案骨架
# ================================================

set -e

# === 建立目錄結構 ===
mkdir -p microservice_container/go-nim-service
cd microservice_container/go-nim-service

# === main.go ===
cat <<'EOF' > main.go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

// HealthCheck 回傳服務狀態
func healthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
	})
}

// Protein Folding 模擬端點
func foldProtein(w http.ResponseWriter, r *http.Request) {
	var input map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	result := map[string]string{
		"sequence": fmt.Sprintf("%v", input["sequence"]),
		"structure": "mock_structure_xyz",
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func main() {
	http.HandleFunc("/health", healthCheck)
	http.HandleFunc("/fold", foldProtein)

	fmt.Println("🚀 Go NIM Service running at http://0.0.0.0:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
EOF

# === go.mod ===
cat <<'EOF' > go.mod
module go-nim-service

go 1.22
EOF

# === Dockerfile ===
cat <<'EOF' > Dockerfile
# === Build Stage ===
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o ai-med-service main.go

# === Run Stage ===
FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/ai-med-service .

CMD ["./ai-med-service"]
EOF

# === build.sh ===
cat <<'EOF' > build.sh
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
EOF

chmod +x build.sh
cd ../..
echo "✅ microservice_container/go-nim-service 已建立完成！"

