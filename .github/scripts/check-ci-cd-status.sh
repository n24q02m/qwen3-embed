#!/bin/bash
# Check CI/CD workflow status before promoting
# CI: strict (must pass). CD: lenient (warn if failed, block only if running).
set -euo pipefail

BRANCH="${BRANCH:-dev}"
CI_WORKFLOW="${CI_WORKFLOW:-ci.yml}"
CD_WORKFLOW="${CD_WORKFLOW:-cd.yml}"

for arg in "$@"; do
  case $arg in
    --branch=*) BRANCH="${arg#*=}" ;;
    --ci-workflow=*) CI_WORKFLOW="${arg#*=}" ;;
    --cd-workflow=*) CD_WORKFLOW="${arg#*=}" ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

echo "Checking CI/CD status on $BRANCH branch..."
FAILED=false

# --- CI: Strict check (must pass) ---
echo "--- Checking CI workflow ($CI_WORKFLOW) ---"
CI_STATUS=$(gh run list --workflow="$CI_WORKFLOW" --branch="$BRANCH" --limit=1 --json status,conclusion --jq '.[0]' 2>/dev/null || echo "")

if [ -z "$CI_STATUS" ] || [ "$CI_STATUS" = "null" ]; then
  echo "Warning: No CI runs found for $BRANCH branch"
else
  CI_RUN_STATUS=$(echo "$CI_STATUS" | jq -r '.status')
  CI_CONCLUSION=$(echo "$CI_STATUS" | jq -r '.conclusion')
  echo "CI - Status: $CI_RUN_STATUS, Conclusion: $CI_CONCLUSION"

  if [ "$CI_RUN_STATUS" != "completed" ]; then
    echo "Error: CI is still running on $BRANCH branch"
    FAILED=true
  elif [ "$CI_CONCLUSION" != "success" ]; then
    echo "Error: CI failed on $BRANCH branch"
    FAILED=true
  else
    echo "CI passed"
  fi
fi

# --- CD: Lenient check (block only if in-progress, warn if failed) ---
echo "--- Checking CD workflow ($CD_WORKFLOW) ---"
CD_STATUS=$(gh run list --workflow="$CD_WORKFLOW" --branch="$BRANCH" --event=push --limit=1 --json status,conclusion --jq '.[0]' 2>/dev/null || echo "")

if [ -z "$CD_STATUS" ] || [ "$CD_STATUS" = "null" ]; then
  echo "Warning: No CD runs found for $BRANCH branch"
else
  CD_RUN_STATUS=$(echo "$CD_STATUS" | jq -r '.status')
  CD_CONCLUSION=$(echo "$CD_STATUS" | jq -r '.conclusion')
  echo "CD - Status: $CD_RUN_STATUS, Conclusion: $CD_CONCLUSION"

  if [ "$CD_RUN_STATUS" != "completed" ]; then
    echo "Error: CD is still running on $BRANCH branch. Please wait for release to complete."
    FAILED=true
  elif [ "$CD_CONCLUSION" != "success" ]; then
    echo "Warning: CD had issues on $BRANCH branch (downstream job may have failed)."
    echo "Proceeding with promotion - publish/build failures don't block code promotion."
  else
    echo "CD passed"
  fi
fi

if [ "$FAILED" = true ]; then
  echo ""
  echo "Please fix the issues or wait for workflows to complete before promoting."
  exit 1
fi

echo ""
echo "All checks passed. Proceeding with promotion..."
