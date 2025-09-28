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
