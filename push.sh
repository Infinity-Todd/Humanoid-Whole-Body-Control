#!/usr/bin/env bash
set -e
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "❌  Not a git repo"; exit 1; fi
MSG=${1:-"update $(date +'%Y-%m-%d %H:%M:%S')"}
echo "📝  $MSG"
git add -A
git commit -m "$MSG" --allow-empty
CUR=$(git symbolic-ref --short HEAD)
echo "🚀  pushing to origin/$CUR ..."
git push origin "$CUR"
echo "✅  done"
