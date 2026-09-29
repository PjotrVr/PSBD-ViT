#!/bin/bash
# Clone every reference implementation in third_party.lock into third_party/<name>
# at its pinned commit. Safe to rerun: a checkout already at its pin is left alone,
# and a directory at any other commit is reported and never touched, since it may
# hold someone's own reading notes or a deliberately different revision.
#
# Outbound network on Supek needs the proxy:
#   export http_proxy=http://10.150.1.1:3128 https_proxy=http://10.150.1.1:3128
#
# Usage: scripts/fetch_third_party.sh [name ...]   (no names fetches every row)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$REPO_ROOT/third_party.lock"
TARGET_ROOT="$REPO_ROOT/third_party"

wanted() {
    local name="$1"
    shift
    if [ "$#" -eq 0 ]; then
        return 0
    fi
    for requested in "$@"; do
        if [ "$requested" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

# A blobless clone downloads file contents only for the pinned tree, which keeps
# BackdoorBench and the others at a fraction of a full history clone.
clone_at_commit() {
    local name="$1" url="$2" commit="$3" sparse_paths="$4"
    local destination="$TARGET_ROOT/$name"

    if [ -d "$destination/.git" ]; then
        local current
        current="$(git -C "$destination" rev-parse HEAD)"
        if [ "$current" = "$commit" ]; then
            echo "ok       $name $commit"
            return 0
        fi
        echo "skipped  $name is at $current, the lock pins $commit, left untouched" >&2
        return 1
    fi
    if [ -e "$destination" ]; then
        echo "skipped  $destination exists and is not a git checkout, left untouched" >&2
        return 1
    fi

    if [ -n "$sparse_paths" ]; then
        git clone --quiet --filter=blob:none --no-checkout "$url" "$destination"
        git -C "$destination" sparse-checkout set --no-cone ${sparse_paths//,/ }
    else
        git clone --quiet --filter=blob:none --no-checkout "$url" "$destination"
    fi
    git -C "$destination" -c advice.detachedHead=false checkout --quiet "$commit"
    echo "cloned   $name $commit"
}

mkdir -p "$TARGET_ROOT"
failures=0
while read -r name url commit sparse_paths; do
    case "$name" in
        "" | "#"*) continue ;;
    esac
    if ! wanted "$name" "$@"; then
        continue
    fi
    if ! clone_at_commit "$name" "$url" "$commit" "${sparse_paths:-}"; then
        failures=$((failures + 1))
    fi
done < "$LOCK_FILE"

if [ "$failures" -gt 0 ]; then
    echo "$failures reference checkouts did not match the lock" >&2
    exit 1
fi
