#!/usr/bin/env bash
# Two repository-wide rules, checked on every push:
#   1. no Chinese characters in tracked files;
#   2. no AI attribution in tracked files or commit messages.
# The attribution shapes live in attribution-patterns.txt, the one file the scan
# skips. Byte-level `grep -E` under LC_ALL=C on purpose: stock macOS grep has no
# -P and its bash 3.2 has no \u escapes, and both fail silently rather than loudly.
set -uo pipefail
export LC_ALL=C

fail=0

# CJK Unified Ideographs U+4E00-U+9FFF as UTF-8 byte ranges.
cjk=$'\xe4[\xb8-\xbf][\x80-\xbf]|[\xe5-\xe8][\x80-\xbf][\x80-\xbf]|\xe9[\x80-\xbf][\x80-\xbf]'
if git ls-files -z | xargs -0 grep -nE "$cjk" 2>/dev/null; then
  echo "hygiene: Chinese characters found in tracked files (see above)" >&2
  fail=1
fi

patterns_file="$(cd "$(dirname "$0")" && pwd)/attribution-patterns.txt"
[ -r "$patterns_file" ] || { echo "hygiene: cannot read $patterns_file" >&2; exit 1; }

patterns=()
while IFS= read -r line || [ -n "$line" ]; do
  [ -n "$line" ] && patterns+=("$line")
done <"$patterns_file"
# A truncated patterns file would otherwise scan nothing and still print "ok".
[ "${#patterns[@]}" -ge 5 ] || { echo "hygiene: expected >= 5 patterns, found ${#patterns[@]}" >&2; exit 1; }

for pattern in "${patterns[@]}"; do
  if git ls-files -z -- . ':(exclude)scripts/attribution-patterns.txt' \
      | xargs -0 grep -nEi "$pattern" 2>/dev/null; then
    echo "hygiene: AI attribution in tracked files: $pattern" >&2
    fail=1
  fi
  if git log --format=%B 2>/dev/null | grep -nEi "$pattern" >/dev/null; then
    echo "hygiene: AI attribution in a commit message: $pattern" >&2
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then echo "hygiene: ok"; fi
exit "$fail"
