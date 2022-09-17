#!/bin/bash
set -e

sudo -u ec2-user -i <<'EOF'
source activate pytorch_p38
pip install dataclasses-json
conda deactivate
EOF
