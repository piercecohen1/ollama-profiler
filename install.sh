#!/bin/sh
set -e

REPO="piercecohen1/ollama-bench"
BINARY="ollama-bench"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# Detect OS
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
    linux)  OS="linux" ;;
    darwin) OS="darwin" ;;
    *)      echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)   ARCH="amd64" ;;
    aarch64|arm64)   ARCH="arm64" ;;
    *)               echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

ASSET="${BINARY}-${OS}-${ARCH}"
URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"

echo "Downloading ${BINARY} for ${OS}/${ARCH}..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL" -o "${TMP}/${BINARY}"
elif command -v wget >/dev/null 2>&1; then
    wget -q "$URL" -O "${TMP}/${BINARY}"
else
    echo "Error: curl or wget required"
    exit 1
fi

chmod +x "${TMP}/${BINARY}"

# Install
if [ -w "$INSTALL_DIR" ]; then
    mv "${TMP}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
else
    echo "Installing to ${INSTALL_DIR} (requires sudo)..."
    sudo mv "${TMP}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
fi

echo "Installed ${BINARY} to ${INSTALL_DIR}/${BINARY}"
echo ""
echo "Usage:"
echo "  ${BINARY} gemma4:e4b gemma4:26b -n 3"
echo "  ${BINARY} --tui"
