# Remove all "Co-Authored-By: Claude ..." trailers from the git history.
#
# PowerShell-native version (uses Python as the message filter to avoid
# bash/WSL dependencies). Run from the repository root in PowerShell.
#
# USAGE
#   1) Working tree must be clean (no uncommitted changes):
#        git status
#   2) Make a safety branch:
#        git branch backup/before-claude-cleanup
#   3) Run this script:
#        powershell -ExecutionPolicy Bypass -File scripts\remove_claude_coauthors.ps1
#   4) Verify clean history:
#        git log --all --pretty=format:"%h %an <%ae> | %s" | Select-Object -First 10
#   5) Force-push (destructive on remote):
#        git push --force-with-lease origin main
#        git push --force-with-lease --tags origin
#
# Idempotent: safe to re-run.

$ErrorActionPreference = "Stop"

# ---------- Sanity checks ----------

git rev-parse --is-inside-work-tree 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not inside a git repository."
    exit 1
}

$dirty = git status --porcelain
if ($dirty) {
    Write-Error "Working tree has uncommitted changes. Commit or stash first."
    Write-Host $dirty
    exit 1
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "python not found in PATH. Install Python 3 or add it to PATH."
    exit 1
}
$pythonExe = $pythonCmd.Source

# ---------- Write the Python message filter ----------

$filterScript = @'
import sys, re

msg = sys.stdin.read()

# Drop any line that is a Co-Authored-By trailer for Claude / anthropic
cleaned = re.sub(
    r"^Co-Authored-By:[^\r\n]*(?:Claude|anthropic)[^\r\n]*\r?\n?",
    "",
    msg,
    flags=re.MULTILINE | re.IGNORECASE,
)

# Collapse runs of trailing blank lines to a single newline
cleaned = re.sub(r"(\r?\n)+\s*$", "\n", cleaned)

sys.stdout.write(cleaned)
'@

$tempFilter = Join-Path $env:TEMP "remove_claude_coauthors_filter.py"
Set-Content -Path $tempFilter -Value $filterScript -Encoding UTF8

try {
    Write-Host ">> Rewriting commit messages to drop Co-Authored-By: Claude trailers..."

    # Suppress the filter-branch deprecation warning (filter-repo is nicer but
    # not always installed on Windows).
    $env:FILTER_BRANCH_SQUELCH_WARNING = "1"

    # The msg-filter shell command. Git invokes it through sh.exe on Windows,
    # which strips backslashes from quoted strings. Use forward slashes for
    # BOTH the python executable and the script path -- Git Bash on Windows
    # accepts forward-slash paths and passes them to Win32 APIs correctly.
    $tempFilterFwd = $tempFilter -replace '\\', '/'
    $pythonExeFwd = $pythonExe -replace '\\', '/'
    $filterCmd = "`"$pythonExeFwd`" `"$tempFilterFwd`""

    git filter-branch --force --msg-filter $filterCmd --tag-name-filter cat -- --all

    if ($LASTEXITCODE -ne 0) {
        Write-Error "git filter-branch failed."
        exit 1
    }

    Write-Host ""
    Write-Host ">> Verifying no Co-Authored-By Claude lines remain..."

    $allMessages = git log --all --pretty=format:"%B"
    $remaining = $allMessages | Select-String -Pattern "co-authored-by.*(claude|anthropic)" -CaseSensitive
    # Case-insensitive equivalent:
    $remainingCI = $allMessages -join "`n" | Select-String -Pattern "co-authored-by.*(claude|anthropic)" -CaseSensitive:$false -AllMatches
    if ($remainingCI.Matches.Count -gt 0) {
        Write-Warning "Some co-author lines still present:"
        $remainingCI.Matches | ForEach-Object { Write-Host $_.Value }
        exit 1
    }
    Write-Host ">> Clean. History rewritten."

    Write-Host ""
    Write-Host ">> Cleaning up filter-branch backup refs..."
    $backupRefs = @(git for-each-ref --format='%(refname)' refs/original/) | Where-Object { $_ }
    foreach ($ref in $backupRefs) {
        $ref = $ref.Trim()
        if ($ref) {
            git update-ref -d $ref 2>$null
        }
    }
    git reflog expire --expire=now --all 2>$null | Out-Null
    git gc --prune=now --aggressive 2>$null | Out-Null

    Write-Host ""
    Write-Host ">> Done. Next steps (run manually when ready):"
    Write-Host "     git log --all --pretty=format:'%h %an <%ae> | %s' | Select-Object -First 10"
    Write-Host "     git push --force-with-lease origin main"
    Write-Host "     git push --force-with-lease --tags origin"
}
finally {
    Remove-Item -Path $tempFilter -ErrorAction SilentlyContinue
}
