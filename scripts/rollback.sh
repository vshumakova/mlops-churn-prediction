#!/bin/bash
echo "Rolling back to previous model"

# Backup current model
if [ -f "models/model.pkl" ]; then
    cp models/model.pkl models/model.pkl.backup
    echo "Current model backed up"
fi

# Restore from previous model
if [ -f "models/previous_model.pkl" ]; then
    cp models/previous_model.pkl models/model.pkl
    echo "Previous model restored"
elif [ -f "api/models/model.pkl.backup" ]; then
    cp api/models/model.pkl.backup models/model.pkl
    echo "Restored from backup"
else
    echo "No backup found (first deployment?)"
fi

# Restart API
docker-compose restart api 2>/dev/null || echo "Docker not running"

echo "Rollback complete"
