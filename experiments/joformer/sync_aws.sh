#!/bin/bash
# Sync joformer to/from AWS server
# Usage:
#   ./sync_aws.sh push    — push local files to AWS (excludes checkpoints, __pycache__)
#   ./sync_aws.sh pull    — pull results from AWS (excludes __pycache__)
#   ./sync_aws.sh         — defaults to push

AWS_HOST="ubuntu@34.228.210.71"
AWS_KEY="$HOME/AWS/ML-server.pem"
LOCAL_DIR="$HOME/Dropbox/ACarrot/Papers/journey_groupoids_tmlr_v7/experiments/joformer/"
REMOTE_DIR="~/joformer/"

EXCLUDE_COMMON="--exclude=venv/ --exclude=__pycache__/ --exclude=.DS_Store"

ACTION="${1:-push}"

if [ "$ACTION" = "push" ]; then
    echo "Pushing local -> AWS..."
    rsync -avz --exclude=checkpoints/ $EXCLUDE_COMMON -e "ssh -i $AWS_KEY" "$LOCAL_DIR" "$AWS_HOST:$REMOTE_DIR"
elif [ "$ACTION" = "pull" ]; then
    echo "Pulling AWS -> local..."
    rsync -avz $EXCLUDE_COMMON -e "ssh -i $AWS_KEY" "$AWS_HOST:$REMOTE_DIR" "$LOCAL_DIR"
else
    echo "Usage: ./sync_aws.sh [push|pull]"
    exit 1
fi
