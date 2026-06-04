#!/usr/bin/env bash
#
# Remove all "Co-Authored-By: Claude ..." trailers from the git history.
#
# This rewrites every commit message that contains a Claude / anthropic
# co-author line. After the rewrite, the GitHub "Contributors" list will
# drop the synthetic claude contributor at the next push.
#
# Safe to run multiple times. Idempotent if the trailers are already gone.
#
# USAGE
#   1) Make sure your working tree is clean:
#        git status
#   2) Make a safety branch (so you can recover if something goes wrong):
#        git branch backup/before-claude-cleanup
#   3) Run this script from the repo root:
#        bash scripts/remove_claude_coauthors.sh
#   4) Inspect the rewritten history:
#        git log --all --pretty=format:"%h %an <%ae> | %s" | head
#        git log --all --pretty=format:"%H%n%B" | grep -i "co-authored-by" | head
#      (the second command should return nothing)
#   5) Force-push the rewrite to GitHub (destructive on remote):
#        git push --force-with-lease origin main
#        git push --force-with-lease --tags origin
#
# NOTES
#   - Force-pushing rewrites history that any collaborator already pulled.
#     For a solo-maintained repo this is fine. For a collaborated repo,
#     coordinate with collaborators first.
#   - This script does NOT modify the working tree -- only the commit
#     graph. Files stay identical.
#   - GitHub may take a few minutes to re-index the contributor list
#     after the push.

set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if ! git diff-index --quiet HEAD --; then
  echo "Error: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

echo ">> Rewriting commit messages to drop Co-Authored-By: Claude trailers..."

# Use filter-branch's --msg-filter to strip the lines. The sed expression
# deletes any line that matches Co-Authored-By with Claude or
# noreply@anthropic.com, plus any immediately preceding blank line that
# would now be a trailing blank.
git filter-branch --force --msg-filter '
  sed -E "/^Co-Authored-By: *Claude.*$/Id; /^Co-Authored-By:.*noreply@anthropic\.com.*$/Id" \
  | awk "
      # Strip trailing blank lines added by removed trailers
      { lines[NR]=\$0 }
      END {
        last = NR
        while (last > 0 && lines[last] ~ /^[[:space:]]*$/) last--
        for (i = 1; i <= last; i++) print lines[i]
      }
    "
' --tag-name-filter cat -- --all

echo ""
echo ">> Verifying no Co-Authored-By Claude lines remain..."
remaining=$(git log --all --pretty=format:"%B" | grep -ciE "co-authored-by.*(claude|anthropic)" || true)
if [ "$remaining" -gt 0 ]; then
  echo "WARNING: $remaining co-author lines still present. Inspect with:"
  echo "  git log --all --pretty=format:'%H%n%B' | grep -iE 'co-authored-by.*(claude|anthropic)'"
  exit 1
fi
echo ">> Clean. History rewritten."

echo ""
echo ">> Cleaning up filter-branch backup refs..."
git for-each-ref --format='%(refname)' refs/original/ \
  | xargs -r -n1 git update-ref -d
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo ">> Done. Next steps (run manually when ready):"
echo "     git log --all --pretty=format:'%h %an <%ae> | %s' | head"
echo "     git push --force-with-lease origin main"
echo "     git push --force-with-lease --tags origin"
