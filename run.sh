#!/usr/bin/env bash
# llm-speedtest 一键启动脚本（macOS / Linux）
# 无需预先安装 Git；自动检测 Python 3.9+，缺失时提供中文指引或一键安装。
# 用法:
#   curl -fsSL https://raw.githubusercontent.com/oktothemoon/llm-speedtest/main/run.sh | bash
# 可选环境变量:
#   LLM_SPEEDTEST_BASE_URL  自定义下载源（默认 GitHub raw）
#   LLM_SPEEDTEST_DIR       本地安装目录（默认 ~/llm-speedtest）

set -euo pipefail

BASE_URL="${LLM_SPEEDTEST_BASE_URL:-https://raw.githubusercontent.com/oktothemoon/llm-speedtest/main}"
BASE_URL="${BASE_URL%/}"
INSTALL_DIR="${LLM_SPEEDTEST_DIR:-$HOME/llm-speedtest}"
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"

PYTHON_BIN=""
FOUND_OLD_PYTHON=""

info()  { printf '\033[36m[llm-speedtest]\033[0m %s\n' "$*"; }
ok()    { printf '\033[32m[llm-speedtest]\033[0m %s\n' "$*"; }
warn()  { printf '\033[33m[llm-speedtest]\033[0m %s\n' "$*" >&2; }
err()   { printf '\033[31m[llm-speedtest]\033[0m %s\n' "$*" >&2; }

has_tty() {
  [ -e /dev/tty ] && (: </dev/tty) 2>/dev/null
}

ask_yes_no() {
  local prompt="$1"
  local default="${2:-Y}"
  local reply=""
  if ! has_tty; then
    return 1
  fi
  if [ "$default" = "Y" ]; then
    printf '%s [Y/n]: ' "$prompt" >/dev/tty
  else
    printf '%s [y/N]: ' "$prompt" >/dev/tty
  fi
  if ! read -r reply </dev/tty; then
    return 1
  fi
  reply="$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]')"
  if [ -z "$reply" ]; then
    [ "$default" = "Y" ]
    return $?
  fi
  [ "$reply" = "y" ] || [ "$reply" = "yes" ]
}

is_macos_clt_installed() {
  [ "$OS_NAME" != "Darwin" ] || xcode-select -p >/dev/null 2>&1
}

is_macos_stub() {
  local bin_path="$1"
  [ "$OS_NAME" = "Darwin" ] && [ "$bin_path" = "/usr/bin/python3" ] && ! is_macos_clt_installed
}

check_python_candidate() {
  local cand="$1"
  [ -n "$cand" ] || return 1
  [ -x "$cand" ] || return 1
  if is_macos_stub "$cand"; then
    return 1
  fi
  local ver
  ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  [ -n "$ver" ] || return 1
  if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    PYTHON_BIN="$cand"
    return 0
  fi
  if [ -z "$FOUND_OLD_PYTHON" ]; then
    FOUND_OLD_PYTHON="$ver ($cand)"
  fi
  return 1
}

find_python() {
  PYTHON_BIN=""
  FOUND_OLD_PYTHON=""
  local candidates=()
  local cmd_path

  if [ "$OS_NAME" = "Darwin" ]; then
    candidates+=(
      "/opt/homebrew/bin/python3"
      "/usr/local/bin/python3"
      "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
    )
    local fw
    for fw in /Library/Frameworks/Python.framework/Versions/3.*/bin/python3; do
      [ -x "$fw" ] && candidates+=("$fw")
    done
  fi

  local name
  for name in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python; do
    cmd_path="$(command -v "$name" 2>/dev/null || true)"
    [ -n "$cmd_path" ] && candidates+=("$cmd_path")
  done

  local cand
  for cand in "${candidates[@]}"; do
    if check_python_candidate "$cand"; then
      return 0
    fi
  done
  return 1
}

guide_and_install_python() {
  echo ""
  if [ -n "$FOUND_OLD_PYTHON" ]; then
    warn "检测到当前 Python 版本过低：$FOUND_OLD_PYTHON（本工具需要 Python 3.9 及以上版本）。"
  else
    warn "未检测到可用的 Python 3.9+ 环境（本工具基于纯 Python 标准库，无需额外第三方依赖）。"
  fi
  echo ""

  if [ "$OS_NAME" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      info "检测到本机已安装 Homebrew，可直接安装 Python 3.12："
      echo "  brew install python@3.12"
      echo ""
      if ask_yes_no "是否现在通过 Homebrew 自动安装 Python 3.12？" "Y"; then
        brew install python@3.12 || true
        if find_python; then
          ok "Python 安装成功：$PYTHON_BIN"
          return 0
        fi
      fi
    else
      info "macOS 安装 Python 3.9+（任选其一）："
      echo "  1. 推荐：前往官网下载 macOS 安装包，双击一路下一步即可："
      echo "     https://www.python.org/downloads/macos/"
      echo "  2. 或在终端运行以下命令安装 Apple 命令行工具（自带 Python 3.9+）："
      echo "     xcode-select --install"
      echo ""
      if ! is_macos_clt_installed && ask_yes_no "是否现在触发 xcode-select --install 弹窗？" "N"; then
        xcode-select --install || true
        info "请在弹出的系统窗口中点击「安装」，完成后重新运行本命令即可。"
        exit 1
      fi
    fi
    info "安装完成后，重新粘贴运行本命令即可开始测速。"
    exit 1
  fi

  # Linux
  local sudo_cmd=""
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    sudo_cmd="sudo"
  fi

  local install_cmd=""
  if command -v apt-get >/dev/null 2>&1; then
    install_cmd="${sudo_cmd:+$sudo_cmd }apt-get update && ${sudo_cmd:+$sudo_cmd }apt-get install -y python3"
  elif command -v dnf >/dev/null 2>&1; then
    install_cmd="${sudo_cmd:+$sudo_cmd }dnf install -y python3"
  elif command -v yum >/dev/null 2>&1; then
    install_cmd="${sudo_cmd:+$sudo_cmd }yum install -y python3"
  elif command -v pacman >/dev/null 2>&1; then
    install_cmd="${sudo_cmd:+$sudo_cmd }pacman -Sy --noconfirm python"
  elif command -v zypper >/dev/null 2>&1; then
    install_cmd="${sudo_cmd:+$sudo_cmd }zypper install -y python3"
  elif command -v apk >/dev/null 2>&1; then
    install_cmd="${sudo_cmd:+$sudo_cmd }apk add --no-cache python3"
  fi

  if [ -n "$install_cmd" ]; then
    info "检测到系统包管理器，可通过以下命令安装 Python 3："
    echo "  $install_cmd"
    echo ""
    if ask_yes_no "是否现在自动执行上述命令安装 Python 3？" "Y"; then
      if bash -c "$install_cmd"; then
        if find_python; then
          ok "Python 安装成功：$PYTHON_BIN"
          return 0
        fi
        warn "系统默认仓库安装的 Python 版本仍低于 3.9。"
      else
        warn "自动安装未能完成。"
      fi
    fi
  fi

  echo ""
  info "请手动安装 Python 3.9+ 后重新运行本命令："
  echo "  - 官网下载：https://www.python.org/downloads/"
  echo "  - 或使用 uv 快速获取：curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12"
  exit 1
}

download_file() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.tmp.$$"
  local ok_dl=0

  if command -v curl >/dev/null 2>&1; then
    if curl -fsSL --connect-timeout 15 --max-time 60 -o "$tmp" "$url"; then
      ok_dl=1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -q --timeout=15 -O "$tmp" "$url"; then
      ok_dl=1
    fi
  else
    err "未找到 curl 或 wget，无法自动下载测速脚本。"
    echo "请手动访问 https://github.com/oktothemoon/llm-speedtest 下载 speedtest.py 和 ui.py。" >&2
    exit 1
  fi

  if [ "$ok_dl" -ne 1 ] || [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    return 1
  fi
  mv -f "$tmp" "$dest"
  return 0
}

sync_files() {
  mkdir -p "$INSTALL_DIR"

  local git_bin=""
  git_bin="$(command -v git 2>/dev/null || true)"
  if [ "$OS_NAME" = "Darwin" ] && [ "$git_bin" = "/usr/bin/git" ] && ! is_macos_clt_installed; then
    git_bin=""
  fi

  if [ -d "$INSTALL_DIR/.git" ] && [ -n "$git_bin" ]; then
    info "检测到本地 Git 仓库（$INSTALL_DIR），正在更新..."
    if "$git_bin" -C "$INSTALL_DIR" pull --ff-only >/dev/null 2>&1; then
      if [ -f "$INSTALL_DIR/speedtest.py" ] && [ -f "$INSTALL_DIR/ui.py" ]; then
        ok "已通过 Git 更新到最新版本。"
        return 0
      fi
    else
      if [ -f "$INSTALL_DIR/speedtest.py" ] && [ -f "$INSTALL_DIR/ui.py" ]; then
        warn "Git 自动更新未完成（可能网络异常或存在本地修改），将继续使用本地文件。"
        return 0
      fi
    fi
  fi

  info "正在下载最新测速脚本到 $INSTALL_DIR ..."
  if ! download_file "$BASE_URL/speedtest.py" "$INSTALL_DIR/speedtest.py" || \
     ! download_file "$BASE_URL/ui.py" "$INSTALL_DIR/ui.py"; then
    if [ -f "$INSTALL_DIR/speedtest.py" ] && [ -f "$INSTALL_DIR/ui.py" ]; then
      warn "在线更新失败（可能是 GitHub 网络连接超时），将继续使用本地已有版本。"
      return 0
    fi
    err "下载失败：$BASE_URL"
    echo "" >&2
    echo "可能原因：当前网络无法直连 GitHub raw。" >&2
    echo "建议排查：" >&2
    echo "  1. 检查网络/代理设置后重试" >&2
    echo "  2. 或打开浏览器下载仓库源码压缩包：https://github.com/oktothemoon/llm-speedtest" >&2
    echo "  3. 若有镜像源，可设置环境变量 LLM_SPEEDTEST_BASE_URL 后重试" >&2
    exit 1
  fi
  chmod +x "$INSTALL_DIR/speedtest.py" 2>/dev/null || true
  ok "脚本已就绪。"
}

main() {
  if ! find_python; then
    guide_and_install_python
  fi

  sync_files

  cd "$INSTALL_DIR"
  if [ ! -t 0 ] && has_tty; then
    exec "$PYTHON_BIN" "$INSTALL_DIR/speedtest.py" "$@" </dev/tty
  else
    exec "$PYTHON_BIN" "$INSTALL_DIR/speedtest.py" "$@"
  fi
}

main "$@"
