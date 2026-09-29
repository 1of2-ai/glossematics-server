#!/bin/bash
# Manage the per-user launchd agent for gloss-server.
#
# usage:
#   gloss-server-launchagent.sh install [server options] [--label ID]
#   gloss-server-launchagent.sh uninstall [--label ID]
#   gloss-server-launchagent.sh restart [--label ID]
#   gloss-server-launchagent.sh status [--label ID]
#
# Install is intentionally transactional: release-build and --check-config run while the old
# daemon is still healthy, files are staged before launchd is touched, and a failed bootstrap or
# readiness gate restores the previous binary/resource/plist snapshot when one exists.

set -euo pipefail
umask 077

LABEL="com.meridian.glossematics.embeddings"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ACTION="${1:-}"
shift || true
case "$ACTION" in
    install|uninstall|restart|status|-h|--help) ;;
    *)
        echo "usage: $0 install|uninstall|restart|status [options]" >&2
        exit 2
        ;;
esac

# Defaults mirror gloss-server. The server itself remains the final validator.
PORT="11435"
COMPUTE="ane"
MAX_BATCH="2048"
MAX_BODY_MB="64"
MAX_TOTAL_BODY_MB="256"
MAX_QUEUE_REQUESTS="128"
MAX_QUEUE_ITEMS="8192"
MAX_REQUEST_TOKENS="131072"
ANE_PROGRAM_BUDGET="64"
BATCH_WINDOW_MS="2.0"
KEEP_WARM_SECONDS="60"
MAX_CONNECTIONS="256"
IO_TIMEOUT_SECONDS="30"
SHUTDOWN_GRACE_SECONDS="15"
ACCESS_LOG="errors"
MODEL_NAME=""
BUNDLE=""
ALLOW_DUMMY_FLAG="0"
DIMENSIONS=""
COMPUTE_SET="0"
BUDGET_SET="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle)                 BUNDLE="${2:?missing --bundle value}"; shift 2 ;;
        --port)                   PORT="${2:?missing --port value}"; shift 2 ;;
        --compute)                COMPUTE="${2:?missing --compute value}"; COMPUTE_SET="1"; shift 2 ;;
        --dimensions)             DIMENSIONS="${2:?missing --dimensions value}"; shift 2 ;;
        --model-name)             MODEL_NAME="${2:?missing --model-name value}"; shift 2 ;;
        --label)                  LABEL="${2:?missing --label value}"; shift 2 ;;
        --max-batch)              MAX_BATCH="${2:?missing --max-batch value}"; shift 2 ;;
        --max-body-mb)            MAX_BODY_MB="${2:?missing --max-body-mb value}"; shift 2 ;;
        --max-total-body-mb)      MAX_TOTAL_BODY_MB="${2:?missing --max-total-body-mb value}"; shift 2 ;;
        --max-queue-requests)     MAX_QUEUE_REQUESTS="${2:?missing --max-queue-requests value}"; shift 2 ;;
        --max-queue-items)        MAX_QUEUE_ITEMS="${2:?missing --max-queue-items value}"; shift 2 ;;
        --max-request-tokens)     MAX_REQUEST_TOKENS="${2:?missing --max-request-tokens value}"; shift 2 ;;
        --ane-program-budget)     ANE_PROGRAM_BUDGET="${2:?missing --ane-program-budget value}"; BUDGET_SET="1"; shift 2 ;;
        --batch-window-ms)        BATCH_WINDOW_MS="${2:?missing --batch-window-ms value}"; shift 2 ;;
        --keep-warm-seconds)      KEEP_WARM_SECONDS="${2:?missing --keep-warm-seconds value}"; shift 2 ;;
        --max-connections)        MAX_CONNECTIONS="${2:?missing --max-connections value}"; shift 2 ;;
        --idle-timeout-seconds)   IO_TIMEOUT_SECONDS="${2:?missing --idle-timeout-seconds value}"; shift 2 ;;
        --shutdown-grace-seconds) SHUTDOWN_GRACE_SECONDS="${2:?missing --shutdown-grace-seconds value}"; shift 2 ;;
        --access-log)             ACCESS_LOG="${2:?missing --access-log value}"; shift 2 ;;
        *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "error: --label may contain only letters, numbers, '.', '_', and '-'" >&2
    exit 2
}

PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
INSTALL_DIR="$HOME/Library/Application Support/Glossematics/bin"
INSTALL_BIN="$INSTALL_DIR/gloss-server"
LOG_DIR="$HOME/Library/Logs/Glossematics"
LOG_FILE="$LOG_DIR/embeddings-server.log"
GUI_TARGET="gui/$(id -u)"

bootout() {
    launchctl bootout "$GUI_TARGET/$LABEL" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
        launchctl print "$GUI_TARGET/$LABEL" >/dev/null 2>&1 || return 0
        sleep 0.2
    done
    echo "error: launchd job $LABEL is still present after bootout" >&2
    return 1
}

job_running() {
    launchctl print "$GUI_TARGET/$LABEL" 2>/dev/null | grep -q 'state = running'
}

plist_port() {
    local plist="${1:-$PLIST_PATH}"
    [[ -f "$plist" ]] || { printf '%s\n' "$PORT"; return; }
    python3 - "$plist" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'rb') as f:
    args = plistlib.load(f).get('ProgramArguments', [])
try:
    print(args[args.index('--port') + 1])
except (ValueError, IndexError):
    print('11435')
PY
}

health_any() {
    curl -sS --connect-timeout 1 -m 3 "http://127.0.0.1:$1/health" 2>/dev/null || true
}

ready() {
    curl -fsS --connect-timeout 1 -m 3 "http://127.0.0.1:$1/ready" 2>/dev/null
}

health_failed() {
    python3 -c 'import json,sys
try:
    print("yes" if json.load(sys.stdin).get("status") == "failed" else "no")
except Exception:
    print("no")'
}

wait_ready() {
    local port="$1"
    local timeout_seconds="${2:-1800}"
    local deadline=$((SECONDS + timeout_seconds))
    local payload=""
    echo -n "waiting for http://127.0.0.1:${port}/ready" >&2
    while (( SECONDS < deadline )); do
        if job_running && payload="$(ready "$port")"; then
            echo >&2
            printf '%s\n' "$payload"
            return 0
        fi
        # Only trust health when launchd confirms *this* job is running. A different process on the
        # port must never be mistaken for this daemon's sticky startup failure.
        if job_running; then
            payload="$(health_any "$port")"
            if [[ -n "$payload" && "$(printf '%s' "$payload" | health_failed)" == "yes" ]]; then
                echo >&2
                echo "error: daemon reported a sticky startup failure:" >&2
                printf '%s\n' "$payload" | python3 -m json.tool >&2 || printf '%s\n' "$payload" >&2
                return 1
            fi
        fi
        echo -n "." >&2
        sleep 1
    done
    echo >&2
    return 1
}

rotate_log() {
    [[ -f "$LOG_FILE" ]] || return 0
    local bytes
    bytes="$(stat -f '%z' "$LOG_FILE" 2>/dev/null || echo 0)"
    (( bytes < 10485760 )) && return 0
    rm -f "$LOG_FILE.3"
    [[ -f "$LOG_FILE.2" ]] && mv "$LOG_FILE.2" "$LOG_FILE.3"
    [[ -f "$LOG_FILE.1" ]] && mv "$LOG_FILE.1" "$LOG_FILE.2"
    mv "$LOG_FILE" "$LOG_FILE.1"
}

server_args() {
    SERVER_ARGS=(
        --bundle "$BUNDLE"
        --port "$PORT"
        --max-batch "$MAX_BATCH"
        --max-body-mb "$MAX_BODY_MB"
        --max-total-body-mb "$MAX_TOTAL_BODY_MB"
        --max-queue-requests "$MAX_QUEUE_REQUESTS"
        --max-queue-items "$MAX_QUEUE_ITEMS"
        --max-request-tokens "$MAX_REQUEST_TOKENS"
        --batch-window-ms "$BATCH_WINDOW_MS"
        --keep-warm-seconds "$KEEP_WARM_SECONDS"
        --max-connections "$MAX_CONNECTIONS"
        --idle-timeout-seconds "$IO_TIMEOUT_SECONDS"
        --shutdown-grace-seconds "$SHUTDOWN_GRACE_SECONDS"
        --access-log "$ACCESS_LOG"
    )
    # Placement flags belong to BidirLM bundles; Jina bundles take a default Matryoshka size.
    case "$(bundle_family "$BUNDLE")" in
        bidirlm) SERVER_ARGS+=(--compute "$COMPUTE" --ane-program-budget "$ANE_PROGRAM_BUDGET") ;;
        jina)
            if [[ "$COMPUTE_SET" == "1" || "$BUDGET_SET" == "1" ]]; then
                echo "error: --compute and --ane-program-budget apply to BidirLM bundles only" >&2
                exit 2
            fi
            ;;
        *) echo "error: unsupported bundle (not BidirLM Omni or jina-embeddings-v5-omni-small)" >&2; exit 2 ;;
    esac
    [[ -n "$DIMENSIONS" ]] && SERVER_ARGS+=(--dimensions "$DIMENSIONS")
    [[ -n "$MODEL_NAME" ]] && SERVER_ARGS+=(--model-name "$MODEL_NAME")
    [[ "$ALLOW_DUMMY_FLAG" == "1" ]] && SERVER_ARGS+=(--allow-dummy)
}

bundle_family() {
    python3 - "$1/manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
if str(manifest.get("format", "")).startswith("bidirlm-omni"):
    print("bidirlm")
elif manifest.get("formatVersion") == 2 and manifest.get("modelID") == "jinaai/jina-embeddings-v5-omni-small":
    print("jina")
else:
    print("unknown")
PY
}

bundle_kind() {
    python3 - "$1/manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
if manifest.get("fixture") == "dummy-noop" or (manifest.get("converter") or {}).get("name") == "dummy-noop":
    print("dummy")
else:
    print("real")
PY
}

generate_plist() {
    local destination="$1"
    shift
    python3 - "$destination" "$LABEL" "$LOG_FILE" "$@" <<'PY'
import plistlib, sys
path, label, log, *args = sys.argv[1:]
plist = {
    'Label': label,
    'ProgramArguments': args,
    'RunAtLoad': True,
    'KeepAlive': True,
    'ThrottleInterval': 10,
    'LimitLoadToSessionType': 'Aqua',
    'StandardOutPath': log,
    'StandardErrorPath': log,
    'Umask': 0o077,
}
with open(path, 'wb') as f:
    plistlib.dump(plist, f, fmt=plistlib.FMT_XML, sort_keys=False)
PY
    plutil -lint "$destination" >/dev/null
}

show_tail() {
    [[ -f "$LOG_FILE" ]] || return 0
    echo "---- tail $LOG_FILE ----" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    echo "-------------------------" >&2
}

case "$ACTION" in
    -h|--help)
        echo "usage: $0 install|uninstall|restart|status [server options]"
        ;;

    install)
        [[ -n "$BUNDLE" ]] || { echo "error: install requires an explicit --bundle" >&2; exit 2; }
        [[ -d "$BUNDLE" ]] || { echo "error: bundle not found: $BUNDLE" >&2; exit 1; }
        case "$(bundle_kind "$BUNDLE")" in
            dummy)
                if [[ "${ALLOW_DUMMY:-0}" != "1" ]]; then
                    echo "error: refusing to install a Core ML dummy fixture (constant embeddings)" >&2
                    echo "set BUNDLE=<real bundle>, or ALLOW_DUMMY=1 to override" >&2
                    exit 1
                fi
                ALLOW_DUMMY_FLAG="1"
                ;;
        esac
        BUNDLE="$(cd "$BUNDLE" && pwd)"
        server_args

        echo "building gloss-server (release)..."
        swift build -c release --package-path "$SERVER_DIR" --product gloss-server
        BIN_DIR="$(swift build --package-path "$SERVER_DIR" -c release --show-bin-path)"
        BIN_PATH="$BIN_DIR/gloss-server"
        [[ -x "$BIN_PATH" ]] || { echo "error: build did not produce gloss-server" >&2; exit 1; }

        echo "preflighting full bundle + server configuration..."
        "$BIN_PATH" "${SERVER_ARGS[@]}" --check-config

        mkdir -p "$HOME/Library/LaunchAgents" "$INSTALL_DIR" "$LOG_DIR"
        chmod 700 "$INSTALL_DIR" "$LOG_DIR" 2>/dev/null || true
        TMP_PLIST="$(mktemp "${TMPDIR:-/tmp}/gloss-plist.XXXXXX")"
        STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gloss-stage.XXXXXX")"
        BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gloss-backup.XXXXXX")"
        trap 'rm -rf "$TMP_PLIST" "$STAGE_DIR" "$BACKUP_DIR"' EXIT

        install -m 755 "$BIN_PATH" "$STAGE_DIR/gloss-server"
        shopt -s nullglob
        RESOURCE_BUNDLES=("$BIN_DIR"/*.bundle)
        shopt -u nullglob
        [[ ${#RESOURCE_BUNDLES[@]} -gt 0 ]] || {
            echo "error: release build produced no SwiftPM resource bundle" >&2
            exit 1
        }
        for resource in "${RESOURCE_BUNDLES[@]}"; do cp -R "$resource" "$STAGE_DIR/"; done
        find "$STAGE_DIR" -type f -name docs.html -print -quit | grep -q . || {
            echo "error: staged SwiftPM resources do not contain docs.html" >&2
            exit 1
        }

        INSTALLED_ARGS=("$INSTALL_BIN" "${SERVER_ARGS[@]}")
        generate_plist "$TMP_PLIST" "${INSTALLED_ARGS[@]}"

        HAD_OLD=0
        if [[ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null || true)" ]]; then
            mkdir -p "$BACKUP_DIR/bin"
            cp -R "$INSTALL_DIR/." "$BACKUP_DIR/bin/"
            HAD_OLD=1
        fi
        [[ -f "$PLIST_PATH" ]] && cp "$PLIST_PATH" "$BACKUP_DIR/old.plist"

        restore_previous_install() {
            if ! bootout; then
                echo "error: cannot restore previous files while the failed launchd job is still present" >&2
                return 1
            fi
            rm -rf "$INSTALL_DIR"
            mkdir -p "$INSTALL_DIR"
            if (( HAD_OLD )); then cp -R "$BACKUP_DIR/bin/." "$INSTALL_DIR/"; fi
            if (( HAD_OLD )) && [[ -f "$BACKUP_DIR/old.plist" ]]; then
                cp "$BACKUP_DIR/old.plist" "$PLIST_PATH"
                if launchctl bootstrap "$GUI_TARGET" "$PLIST_PATH" >/dev/null 2>&1; then
                    local old_port
                    old_port="$(plist_port "$BACKUP_DIR/old.plist")"
                    if wait_ready "$old_port" 1800 >/dev/null; then
                        echo "previous install restored and ready on 127.0.0.1:${old_port}" >&2
                    else
                        echo "warning: previous install launched but did not become ready" >&2
                    fi
                else
                    echo "warning: previous install restored on disk but could not be restarted" >&2
                fi
            else
                rm -f "$PLIST_PATH"
            fi
        }

        # All build/preflight/staging work is complete before stopping the old process.
        rotate_log
        if ! bootout; then
            echo "error: refusing to replace files while the old launchd job is still present" >&2
            exit 1
        fi

        replace_installed_files() {
            rm -rf "$INSTALL_DIR" \
                && mkdir -p "$INSTALL_DIR" \
                && cp -R "$STAGE_DIR/." "$INSTALL_DIR/" \
                && chmod 755 "$INSTALL_BIN" \
                && cp "$TMP_PLIST" "$PLIST_PATH"
        }
        if ! replace_installed_files; then
            echo "error: install file replacement failed; restoring previous install" >&2
            restore_previous_install
            exit 1
        fi

        if ! launchctl bootstrap "$GUI_TARGET" "$PLIST_PATH"; then
            echo "error: launchctl bootstrap failed; restoring previous install" >&2
            restore_previous_install
            show_tail
            exit 1
        fi
        if HEALTH="$(wait_ready "$PORT" 1800)"; then
            printf '%s\n' "$HEALTH" | python3 -m json.tool
            echo "docs:    http://127.0.0.1:${PORT}/docs"
            echo "metrics: http://127.0.0.1:${PORT}/metrics"
        else
            echo "error: new daemon did not become ready; restoring previous install" >&2
            show_tail
            restore_previous_install
            exit 1
        fi
        ;;

    uninstall)
        bootout
        rm -f "$PLIST_PATH"
        rm -rf "$INSTALL_DIR"
        echo "uninstalled $LABEL (logs kept at $LOG_DIR)"
        ;;

    restart)
        [[ -f "$PLIST_PATH" ]] || { echo "error: not installed ($PLIST_PATH missing)" >&2; exit 1; }
        PORT="$(plist_port "$PLIST_PATH")"
        rotate_log
        bootout
        launchctl bootstrap "$GUI_TARGET" "$PLIST_PATH"
        HEALTH="$(wait_ready "$PORT" 1800)"
        printf '%s\n' "$HEALTH" | python3 -m json.tool
        ;;

    status)
        [[ -f "$PLIST_PATH" ]] || { echo "error: not installed ($PLIST_PATH missing)" >&2; exit 1; }
        PORT="$(plist_port "$PLIST_PATH")"
        if ! JOB_STATE="$(launchctl print "$GUI_TARGET/$LABEL" 2>/dev/null)"; then
            echo "error: plist exists but launchd job $LABEL is not loaded" >&2
            exit 1
        fi
        printf '%s\n' "$JOB_STATE" \
            | grep -E 'state =|pid =|last exit code|runs =' \
            | sed 's/^[[:space:]]*//' || true
        printf '%s\n' "$JOB_STATE" | grep -q 'state = running' || {
            echo "error: launchd job is loaded but not running" >&2
            exit 1
        }
        PAYLOAD="$(health_any "$PORT")"
        [[ -n "$PAYLOAD" ]] || { echo "error: no health response on port $PORT" >&2; exit 1; }
        printf '%s\n' "$PAYLOAD" | python3 -m json.tool
        printf '%s\n' "$PAYLOAD" \
            | python3 -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("ready") is True else 1)'
        ;;
esac
