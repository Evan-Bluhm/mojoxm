#!/usr/bin/env bash
# format_check.sh [-q] -- non-mutating check that every .mojo file
# matches `mojo format` output.  Mirrors the pre-commit hook but
# covers the *whole working tree* (not just staged files), so it
# can be run standalone before pushing or as a CI gate.
#
# Exits 0 if every file is already formatted, 1 if any file would
# be rewritten (and prints the list on stderr).  Files are never
# modified.
#
# Flags:
#   -q, --quiet     Suppress the trailing success summary line.
#                   Errors still print on stderr.  Useful in CI
#                   scripts where "no output = OK" is the convention.
#
# Honours the same MOJO override env var as the Makefile + pre-
# commit hook: set MOJO=<path> if `mojo` isn't on PATH.

set -euo pipefail

quiet=0
case "${1:-}" in
    -q|--quiet) quiet=1 ;;
    -h|--help) sed -n '2,17p' "$0" | cut -c3-; exit 0 ;;
    "") ;;
    *) echo "format_check.sh: unknown flag '${1}' (try -h)" >&2; exit 2 ;;
esac

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

# Stage every file into a parallel tree under $tmpdir, run a single
# `mojo format` over the whole batch (one JIT-warmup amortised across
# all files -- ~25x faster than per-file invocations), then diff each
# pair.  In-place rewrite of the temp copies leaves the working tree
# untouched.
copies=()
for f in "${files[@]}"; do
    tmp="$tmpdir/$f"
    mkdir -p "$(dirname "$tmp")"
    cp "$f" "$tmp"
    copies+=("$tmp")
done

"${mojo_cmd[@]}" format -q "${copies[@]}"

bad=()
for f in "${files[@]}"; do
    if ! diff -q "$f" "$tmpdir/$f" >/dev/null 2>&1; then
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

if (( quiet == 0 )); then
    echo "format_check: all ${#files[@]} .mojo files are formatter-clean."
fi
