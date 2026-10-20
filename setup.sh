#!/usr/bin/env bash
set -euo pipefail

for command in git gh; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "Missing required command: $command" >&2
        exit 1
    }
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -f ordered_attributions.py ]]; then
    script="ordered_attributions.py"
elif [[ -f ordered_attribution.py ]]; then
    script="ordered_attribution.py"
else
    echo "No ordered_attributions.py or ordered_attribution.py found." >&2
    exit 1
fi

gh auth status
owner="$(gh api user --jq .login)"
repository="$owner/machine-learning"

if [[ ! -d .git ]]; then
    if git rev-parse --git-dir >/dev/null 2>&1; then
        echo "This folder is inside another Git repository. Stopping." >&2
        exit 1
    fi
    git init -b main
fi

if [[ ! -f LICENSE ]]; then
    cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2026 Flyxion

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
fi

if [[ ! -f README.md ]]; then
    cat > README.md <<EOF
# Machine Learning

Experimental machine-learning programs by Flyxion.

## Order-aware data attribution

\`$script\` is an exploratory scaffold for studying the contributions
of training-data chunks during sequential learning.

The synthetic task includes useful, duplicate, irrelevant, and corrupted
chunks. The experiment compares sampled training-order marginal contributions
with leave-one-out attribution and evaluates removal of low-scoring chunks.

The planted categories describe how examples are constructed; their actual
effects depend on the model, training procedure, and evaluation task.

This is work in progress. Preliminary smoke tests are development checks
and do not establish that one attribution method outperforms another.
GPU runtime and memory requirements have not yet been measured locally.

## Usage

Use Python with PyTorch and NumPy installed. Inspect the available options:

\`\`\`bash
python $script --help
\`\`\`

For GPU execution, use a PyTorch installation compatible with your NVIDIA
hardware and driver.

## License

MIT. See LICENSE.
EOF
fi

if [[ ! -f .gitignore ]]; then
    cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
.venv/
venv/
.env
.pytest_cache/
.ipynb_checkpoints/
outputs/
results/
runs/
checkpoints/
*.pt
*.pth
EOF
fi

# Commit only the intended project files.
# Stop if other files were already staged.
while IFS= read -r -d '' staged; do
    case "$staged" in
        "$script"|README.md|LICENSE|.gitignore|setup.sh) ;;
        *)
            echo "Unrelated file already staged: $staged" >&2
            echo "Commit or unstage it before rerunning setup.sh." >&2
            exit 1
            ;;
    esac
done < <(git diff --cached --name-only -z)

git add -- "$script" README.md LICENSE .gitignore setup.sh

if ! git diff --cached --quiet; then
    git commit -m "Add attribution scaffold, documentation, and MIT license"
fi

branch="$(git symbolic-ref --quiet --short HEAD)" || {
    echo "Detached HEAD: switch to a branch before publishing." >&2
    exit 1
}

# Avoid pushing into an unrelated existing remote.
if git remote get-url origin >/dev/null 2>&1; then
    origin="$(git remote get-url origin)"
    case "$origin" in
        "https://github.com/$repository" | \
        "https://github.com/$repository.git" | \
        "git@github.com:$repository" | \
        "git@github.com:$repository.git" | \
        "ssh://git@github.com/$repository" | \
        "ssh://git@github.com/$repository.git")
            ;;
        *)
            echo "Origin points elsewhere: $origin" >&2
            echo "Expected GitHub repository: $repository" >&2
            exit 1
            ;;
    esac
fi

if ! gh repo view "$repository" >/dev/null 2>&1; then
    gh repo create "$repository" \
        --public \
        --description "Machine-learning experiments by Flyxion"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "https://github.com/$repository.git"
fi

git push -u origin "$branch"

echo "Published: https://github.com/$repository"
