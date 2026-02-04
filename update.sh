#!/usr/bin/env bash
#
# Lethe Update Script
# Checks for updates and applies them
#
# Usage: curl -fsSL https://lethe.gg/update | bash
#
# For container (safe) mode: tells user to restart container
# For native (unsafe) mode: restarts service automatically
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Config
REPO_OWNER="atemerev"
REPO_NAME="lethe"
INSTALL_DIR="${LETHE_INSTALL_DIR:-$HOME/.lethe}"
CONFIG_DIR="${LETHE_CONFIG_DIR:-$HOME/.config/lethe}"

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        Darwin*) echo "mac" ;;
        *) echo "unknown" ;;
    esac
}

detect_install_mode() {
    # Check if running in container mode
    if command -v podman &>/dev/null; then
        if podman ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^lethe$'; then
            echo "container-podman"
            return
        fi
    fi
    
    if command -v docker &>/dev/null; then
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^lethe$'; then
            echo "container-docker"
            return
        fi
    fi
    
    # Check for systemd service (Linux native)
    if [ -f "$HOME/.config/systemd/user/lethe.service" ]; then
        echo "native-systemd"
        return
    fi
    
    # Check for launchd service (Mac native)
    if [ -f "$HOME/Library/LaunchAgents/com.lethe.agent.plist" ]; then
        echo "native-launchd"
        return
    fi
    
    echo "unknown"
}

get_current_version() {
    if [ -d "$INSTALL_DIR/.git" ]; then
        cd "$INSTALL_DIR"
        # Try to get tag, fall back to commit hash
        local tag=$(git describe --tags --exact-match 2>/dev/null || echo "")
        if [ -n "$tag" ]; then
            echo "$tag"
        else
            git rev-parse --short HEAD 2>/dev/null || echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

get_latest_release() {
    local latest=$(curl -fsSL "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/releases/latest" 2>/dev/null | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    if [ -z "$latest" ]; then
        echo "main"
    else
        echo "$latest"
    fi
}

update_code() {
    local target_version="$1"
    
    if [ ! -d "$INSTALL_DIR" ]; then
        error "Installation directory not found: $INSTALL_DIR"
    fi
    
    cd "$INSTALL_DIR"
    
    info "Fetching updates..."
    git fetch origin --tags
    
    if [ "$target_version" != "main" ]; then
        git checkout "$target_version"
    else
        git checkout main
        git pull origin main
    fi
    
    # Update dependencies
    info "Updating dependencies..."
    if command -v uv &>/dev/null; then
        uv sync
    fi
    
    success "Code updated to $target_version"
}

rebuild_container() {
    local container_cmd="$1"
    
    info "Rebuilding container image..."
    cd "$INSTALL_DIR"
    $container_cmd build -t lethe:latest .
    success "Container image rebuilt"
}

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                   LETHE UPDATE                            ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

main() {
    print_header
    
    # Check installation exists
    if [ ! -d "$INSTALL_DIR" ]; then
        error "Lethe is not installed at $INSTALL_DIR"
    fi
    
    # Detect install mode
    local install_mode=$(detect_install_mode)
    local current_version=$(get_current_version)
    local latest_version=$(get_latest_release)
    
    echo "  Install mode:    $install_mode"
    echo "  Current version: $current_version"
    echo "  Latest version:  $latest_version"
    echo ""
    
    # Check if update needed
    if [ "$current_version" == "$latest_version" ]; then
        success "Already up to date!"
        exit 0
    fi
    
    info "Update available: $current_version → $latest_version"
    echo ""
    
    # Update code
    update_code "$latest_version"
    
    # Handle restart based on install mode
    case "$install_mode" in
        container-podman)
            rebuild_container "podman"
            echo ""
            warn "Container mode detected. To apply the update:"
            echo ""
            echo -e "  ${CYAN}podman stop lethe && podman rm lethe${NC}"
            echo ""
            echo "  Then run the same 'podman run' command you used to start it,"
            echo "  or re-run the installer to recreate the container."
            echo ""
            ;;
        container-docker)
            rebuild_container "docker"
            echo ""
            warn "Container mode detected. To apply the update:"
            echo ""
            echo -e "  ${CYAN}docker stop lethe && docker rm lethe${NC}"
            echo ""
            echo "  Then run the same 'docker run' command you used to start it,"
            echo "  or re-run the installer to recreate the container."
            echo ""
            ;;
        native-systemd)
            info "Restarting systemd service..."
            systemctl --user restart lethe
            success "Service restarted!"
            echo ""
            echo "  View logs: journalctl --user -u lethe -f"
            echo ""
            ;;
        native-launchd)
            info "Restarting launchd service..."
            launchctl unload "$HOME/Library/LaunchAgents/com.lethe.agent.plist" 2>/dev/null || true
            launchctl load "$HOME/Library/LaunchAgents/com.lethe.agent.plist"
            success "Service restarted!"
            echo ""
            echo "  View logs: tail -f ~/Library/Logs/lethe.log"
            echo ""
            ;;
        *)
            warn "Could not detect install mode."
            echo "  Please restart Lethe manually."
            echo ""
            ;;
    esac
    
    success "Update complete! ($current_version → $latest_version)"
}

main "$@"
