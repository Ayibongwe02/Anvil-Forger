#!/bin/bash
# Push Anvil v2 to Docker Hub
# Run this script on your local machine

USERNAME="ayibongwe"
IMAGE_NAME="anvil-v2"
TOKEN="[REDACTED]"  # Your PAT

echo "🔐 Logging in to Docker Hub..."
echo "$TOKEN" | docker login -u "$USERNAME" --password-stdin

if [ $? -eq 0 ]; then
    echo "✅ Login successful!"
    
    echo "📤 Pushing image $USERNAME/$IMAGE_NAME:latest..."
    docker push "$USERNAME/$IMAGE_NAME:latest"
    
    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo "🎉 Image available at: https://hub.docker.com/r/$USERNAME/$IMAGE_NAME"
    else
        echo "❌ Push failed"
    fi
else
    echo "❌ Login failed - check username and token"
fi
