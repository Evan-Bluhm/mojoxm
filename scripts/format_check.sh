#!/usr/bin/env bash
# format_check.sh -- non-mutating check that every .mojo file matches
# `mojo format` output.  Mirrors the pre-commit hook but covers the
# *whole working tree* (not just staged files), so it can be run
# standalone before pushing or as a CI gate.
#
# Exits 0 if every file is already formatted, 1 if any file would be
# rewritten (and prints the list).  Files are never modified.
#
# Honours the same MOJO override env var as the Makefile + pre-commit
# hook: set MOJO=<path> if `mojo` isn't on PATH.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

# Pick a mojo invocation: explicit MOJO override, then PATH, then pixi.
if [[ -n "${MOJO:-}" ]]; then
    # shellcheck disable=SC2206
    mojo_cmd=($MOJO)
elif command -v mojo >/dev/null 2>&1; then
    mojo_cmd=(mojo)
elif [[ -f pixi.toml || -f pyproject.toml ]] && command -v pixi >/dev/null 2>&1; then
    mojo_cmd=(pixi run mojo)
else
    echo "format_check.sh: 'mojo' not on PATH and pixi/pyproject.toml not found." >&2
    echo "                 Set MOJO=<path> or run 'pixi install'." >&2
    exit 1
fi

# Find every .mojo file in src/, benchmarks/, examples/, test/.
mapfile -t files < <(find src benchmarks examples test -name '*.mojo' | sort)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "format_check.sh: no .mojo files found" >&2
    exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

bad=()
for f in "${files[@]}"; do
    tmp="$tmpdir/$f"
    mkdir -p "$(dirname "$tmp")"
    cp "$f" "$tmp"
    "${mojo_cmd[@]}" format -q "$tmp"
    if ! diff -q "$f" "$tmp" >/dev/null 2>&1; then
        bad+=("$f")
    fi
done

if [[ ${#bad[@]} -gt 0 ]]; then
    echo "format_check: ${#bad[@]} file(s) need formatting:" >&2
    printf '  %s\n' "${bad[@]}" >&2
    echo >&2
    echo "Run 'make format' to fix." >&2
    exit 1
fi

echo "format_check: all ${#files[@]} .mojo files are formatter-clean."
