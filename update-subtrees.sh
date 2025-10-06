#!/bin/bash
set -e

# --- Update all git subtrees ---

echo "--- 🕸️ Updating 'portal' subtree..."
git subtree pull --prefix=portal/ https://github.com/manuelcheong/genai-learning-portal.git main --squash

echo "--- 🦍 Updating 'kit' subtree..."
git subtree pull --prefix=kit/ https://github.com/manuelcheong/genai-learning-kit.git main --squash

echo "--- 🦞 Updating 'api' subtree..."
git subtree pull --prefix=api/ https://github.com/manuelcheong/genai-learning-api.git main --squash

echo "--- 🐝 Updating 'agent' subtree..."
git subtree pull --prefix=agents/ https://github.com/manuelcheong/genai-learning-agent-base.git main --squash

echo "--- 🐝 Updating 'agent' subtree..."
git subtree pull --prefix=agents/ https://github.com/manuelcheong/genai-learning-summarizing-agent.git  main --squash -X theirs


echo "✅ All subtrees updated successfully!"