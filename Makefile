BINARY  := ollama-bench
MODULE  := github.com/piercecohen1/ollama-bench
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS := -s -w -X main.version=$(VERSION)

PLATFORMS := \
	darwin/amd64 \
	darwin/arm64 \
	linux/amd64 \
	linux/arm64 \
	windows/amd64

.PHONY: build test clean dist all

build:
	go build -ldflags "$(LDFLAGS)" -o $(BINARY) ./cmd/ollama-bench/

test:
	go test ./... -v

clean:
	rm -f $(BINARY)
	rm -rf dist/

dist: clean
	@mkdir -p dist
	@for platform in $(PLATFORMS); do \
		os=$${platform%/*}; \
		arch=$${platform#*/}; \
		ext=""; \
		if [ "$$os" = "windows" ]; then ext=".exe"; fi; \
		output="dist/$(BINARY)-$${os}-$${arch}$${ext}"; \
		echo "Building $$output..."; \
		GOOS=$$os GOARCH=$$arch go build -ldflags "$(LDFLAGS)" -o $$output ./cmd/ollama-bench/; \
	done
	@echo "Done. Binaries in dist/"
	@ls -lh dist/

all: test build
