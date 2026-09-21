#!/bin/bash
# gloss-server launch-agent management for Glossematics embeddings serving.
#
# Installs/uninstalls a per-user launchd agent (~/Library/LaunchAgents) that runs the
# OpenAI-compatible /v1/embeddings daemon (gloss-server) against a local model bundle.
# The agent starts at login and is kept alive; it binds 127.0.0.1 only.
#
# usage:
#   gloss-server-launchagent.sh install  [--bundle DIR] [--port N] [--dimensions N]
#                                        [--model-name ID] [--label ID]
#   gloss-server-launchagent.sh uninstall [--label ID]
#   gloss-server-launchagent.sh restart   [--label ID]
#   gloss-server-launchagent.sh status    [--label ID]
#
# defaults: bundle=artifacts/JinaV5OmniSmall.w8a16.bundle (repo-relative), port=11435,
#           dimensions=1024, label=com.meridian.glossematics.embeddings

set -euo pipefail

LABEL="com.meridian.glossematics.embeddings"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$SERVER_DIR/.." && pwd)"

ACTION="${1:-}"
shift || true
[[ "$ACTION" =~ ^(install|uninstall|restart|status|-h|--help)$ ]] || {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle)     BUNDLE="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --dimensions) DIMENSIONS="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --label)      LABEL="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

PORT="${PORT:-11435}"
DIMENSIONS="${DIMENSIONS:-1024}"

PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
INSTALL_BIN="$HOME/Library/Application Support/Glossematics/bin/gloss-server"
LOG_DIR="$HOME/Library/Logs/Glossematics"
GUI_TARGET="gui/$(id -u)"

bootout() {
    launchctl bootout "$GUI_TARGET/$LABEL" >/dev/null 2>&1 || true
    # bootout is asynchronous; wait until the job is gone before bootstrapping again,
    # otherwise bootstrap can race the teardown and fail with error 5.
    for _ in $(seq 1 20); do
        launchctl print "$GUI_TARGET/$LABEL" >/dev/null 2>&1 || return 0
        sleep 0.5
    done
}

health() {
    curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null
}

case "$ACTION" in
    -h|--help)
        sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;

    install)
        BUNDLE="${BUNDLE:-$REPO_DIR/TestBundles/JinaV5OmniSmall.w8a16.dummy.bundle}"
        if [[ ! -d "$BUNDLE" ]]; then
            echo "error: bundle not found: $BUNDLE" >&2
            echo "pass --bundle /path/to/<model>.bundle (a native-v1 distribution bundle)" >&2
            exit 1
        fi
        BUNDLE="$(cd "$BUNDLE" && pwd)"

        echo "building gloss-server (release)..."
        swift build -c release --package-path "$SERVER_DIR" --product gloss-server
        BIN_PATH="$(swift build --package-path "$SERVER_DIR" -c release --show-bin-path)/gloss-server"
        [[ -x "$BIN_PATH" ]] || { echo "error: build did not produce gloss-server" >&2; exit 1; }

        mkdir -p "$(dirname "$INSTALL_BIN")" "$LOG_DIR"
        cp -f "$BIN_PATH" "$INSTALL_BIN"
        # Resource bundles (docs page, library mel filters, tokenizer assets) must sit next to
        # the installed binary for Bundle.module lookups to resolve.
        BIN_DIR="$(dirname "$BIN_PATH")"
        for bundle in "$BIN_DIR"/*.bundle; do
            [[ -e "$bundle" ]] && cp -R "$bundle" "$(dirname "$INSTALL_BIN")/"
        done

        PROGRAM_ARGS=(
            "  <string>$INSTALL_BIN</string>"
            "  <string>--bundle</string>"
            "  <string>$BUNDLE</string>"
            "  <string>--port</string>"
            "  <string>$PORT</string>"
            "  <string>--dimensions</string>"
            "  <string>$DIMENSIONS</string>"
        )
        if [[ -n "${MODEL_NAME:-}" ]]; then
            PROGRAM_ARGS+=("  <string>--model-name</string>" "  <string>$MODEL_NAME</string>")
        fi

        bootout
        cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
$(printf '%s\n' "${PROGRAM_ARGS[@]}")
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>LimitLoadToSessionType</key>
  <string>Aqua</string>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/embeddings-server.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/embeddings-server.log</string>
</dict>
</plist>
PLIST
        plutil -lint "$PLIST_PATH" >/dev/null

        echo "bootstrapping agent $LABEL ..."
        launchctl bootstrap "$GUI_TARGET" "$PLIST_PATH"

        echo -n "waiting for http://127.0.0.1:${PORT}/health"
        for _ in $(seq 1 120); do
            if HEALTH="$(health)"; then
                echo
                echo "agent is up:"
                echo "$HEALTH" | python3 -m json.tool
                echo
                echo "  label:  $LABEL"
                echo "  plist:  $PLIST_PATH"
                echo "  binary: $INSTALL_BIN"
                echo "  bundle: $BUNDLE"
                echo "  logs:   $LOG_DIR/embeddings-server.log"
                echo
                echo "try: curl -s http://127.0.0.1:${PORT}/v1/models"
                exit 0
            fi
            echo -n "."
            sleep 2
        done
        echo
        echo "error: agent did not become healthy within 240s; check $LOG_DIR/embeddings-server.log" >&2
        exit 1
        ;;

    uninstall)
        bootout
        rm -f "$PLIST_PATH" "$INSTALL_BIN"
        rmdir "$(dirname "$INSTALL_BIN")" 2>/dev/null || true
        echo "uninstalled $LABEL (logs kept in $LOG_DIR)"
        ;;

    restart)
        bootout
        [[ -f "$PLIST_PATH" ]] || { echo "error: not installed ($PLIST_PATH missing)" >&2; exit 1; }
        launchctl bootstrap "$GUI_TARGET" "$PLIST_PATH"
        echo "restarted $LABEL"
        ;;

    status)
        launchctl print "$GUI_TARGET/$LABEL" 2>/dev/null | grep -E "state|pid|last exit" | sed 's/^[[:space:]]*//' || true
        if HEALTH="$(health)"; then
            echo "$HEALTH" | python3 -m json.tool
        else
            echo "health: no response on 127.0.0.1:${PORT}"
        fi
        ;;
esac
