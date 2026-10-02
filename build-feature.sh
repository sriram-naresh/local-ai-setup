#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

BUMP="${1:-patch}"
VERSION_FILE="$PROJECT_DIR/VERSION"
CURRENT_VERSION="$(tr -d '\r\n' < "$VERSION_FILE")"

if [[ ! "$CURRENT_VERSION" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  echo "VERSION must contain a semantic version such as 1.1.0; found: $CURRENT_VERSION" >&2
  exit 1
fi

major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"
patch="${BASH_REMATCH[3]}"

case "$BUMP" in
  patch) patch=$((patch + 1)) ;;
  minor) minor=$((minor + 1)); patch=0 ;;
  major) major=$((major + 1)); minor=0; patch=0 ;;
  *) echo "Usage: $0 [patch|minor|major]" >&2; exit 2 ;;
esac

NEW_VERSION="$major.$minor.$patch"
export APP_VERSION="$NEW_VERSION"

DOCKER_BIN="${DOCKER_BIN:-docker}"
if ! command -v "$DOCKER_BIN" >/dev/null 2>&1; then
  docker_candidate=""
  if [[ -n "${LOCALAPPDATA:-}" ]] && command -v cygpath >/dev/null 2>&1; then
    docker_candidate="$(cygpath -u "$LOCALAPPDATA")/Programs/DockerDesktop/resources/bin/docker.exe"
  elif [[ -n "${LOCALAPPDATA:-}" ]] && command -v wslpath >/dev/null 2>&1; then
    docker_candidate="$(wslpath -u "$LOCALAPPDATA")/Programs/DockerDesktop/resources/bin/docker.exe"
  fi

  if [[ -n "$docker_candidate" && -x "$docker_candidate" ]]; then
    DOCKER_BIN="$docker_candidate"
  else
    echo "Docker CLI '$DOCKER_BIN' was not found. Put docker on PATH or set DOCKER_BIN." >&2
    exit 1
  fi
fi

echo "Building local-rag:$NEW_VERSION..."
"$DOCKER_BIN" compose build app
printf '%s' "$NEW_VERSION" > "$VERSION_FILE"

legacy_stopped=false
legacy_container="$("$DOCKER_BIN" ps --filter 'name=^/local-rag$' --format '{{.Names}}')"
if [[ "$legacy_container" == "local-rag" ]]; then
  echo "Stopping the existing local-rag container to free port 8501..."
  "$DOCKER_BIN" stop local-rag
  legacy_stopped=true
fi

if ! "$DOCKER_BIN" compose up --detach --force-recreate app; then
  if [[ "$legacy_stopped" == true ]]; then
    "$DOCKER_BIN" start local-rag >/dev/null || true
  fi
  echo "Could not start local-rag:$NEW_VERSION; the previous local-rag container was restarted when available." >&2
  exit 1
fi

echo "Running local-rag:$NEW_VERSION on http://localhost:8501"
