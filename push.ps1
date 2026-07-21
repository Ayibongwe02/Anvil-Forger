# Push Anvil v2 to Docker Hub
# Run this script on your local machine with PowerShell

$USERNAME = "ayibongwe"
$IMAGE_NAME = "anvil-v2"
$TOKEN = "[REDACTED]"  # Your PAT

Write-Host "🔐 Logging in to Docker Hub..." -ForegroundColor Cyan

$TOKEN | docker login -u $USERNAME --password-stdin

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Login successful!" -ForegroundColor Green
    
    Write-Host "📤 Pushing image $USERNAME/$IMAGE_NAME`:latest..." -ForegroundColor Cyan
    docker push "$USERNAME/$IMAGE_NAME:latest"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Push successful!" -ForegroundColor Green
        Write-Host "🎉 Image available at: https://hub.docker.com/r/$USERNAME/$IMAGE_NAME" -ForegroundColor Green
        Write-Host ""
        Write-Host "📝 Next steps:" -ForegroundColor Cyan
        Write-Host "  1. Update docker-compose.prod.yml:"
        Write-Host "     - Remove or comment out the 'build:' section"
        Write-Host "     - Set image: ayibongwe/$IMAGE_NAME:latest"
        Write-Host ""
        Write-Host "  2. Deploy: docker-compose -f docker-compose.prod.yml up -d"
    }
    else {
        Write-Host "❌ Push failed" -ForegroundColor Red
    }
}
else {
    Write-Host "❌ Login failed - check username and token" -ForegroundColor Red
}
