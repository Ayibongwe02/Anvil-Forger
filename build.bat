@echo off
setlocal enabledelayedexpansion

set REGISTRY=%REGISTRY%
if "!REGISTRY!"=="" set REGISTRY=docker.io

set IMAGE_NAME=%IMAGE_NAME%
if "!IMAGE_NAME!"=="" set IMAGE_NAME=anvil

set IMAGE_TAG=%IMAGE_TAG%
if "!IMAGE_TAG!"=="" set IMAGE_TAG=latest

set FULL_IMAGE=!REGISTRY!/!IMAGE_NAME!:!IMAGE_TAG!

if "%1"=="help" (
    echo Anvil Docker Makefile Alternative for Windows
    echo.
    echo Build commands:
    echo   .\build.bat build              Build image for development
    echo   .\build.bat build-prod         Build optimized production image
    echo.
    echo Development commands:
    echo   .\build.bat up-dev             Start development stack with hot reload
    echo   .\build.bat down               Stop all containers
    echo   .\build.bat logs-dev           View development logs
    echo   .\build.bat shell              Open shell in running container
    echo.
    echo Production commands:
    echo   .\build.bat up-prod            Start production stack
    echo   .\build.bat logs-prod          View production logs
    echo.
    echo Registry commands:
    echo   .\build.bat push               Push image to registry
    echo.
    echo Maintenance:
    echo   .\build.bat clean              Remove containers and images
    echo   .\build.bat test               Run container health check
    echo.
    echo Examples:
    echo   set REGISTRY=ghcr.io ^& set IMAGE_NAME=myorg/anvil ^& set IMAGE_TAG=v1.0 ^& .\build.bat push
    echo   .\build.bat up-dev
    echo   .\build.bat logs-dev
    goto :eof
)

if "%1"=="build" (
    docker build -t !IMAGE_NAME!:!IMAGE_TAG! .
    goto :eof
)

if "%1"=="build-prod" (
    docker build -t !FULL_IMAGE! .
    goto :eof
)

if "%1"=="up-dev" (
    call :build
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    goto :eof
)

if "%1"=="up-prod" (
    call :build-prod
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    goto :eof
)

if "%1"=="up" (
    call :up-dev
    goto :eof
)

if "%1"=="down" (
    docker compose -f docker-compose.yml -f docker-compose.dev.yml down
    docker compose -f docker-compose.yml -f docker-compose.prod.yml down
    goto :eof
)

if "%1"=="logs" (
    docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
    goto :eof
)

if "%1"=="logs-dev" (
    docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
    goto :eof
)

if "%1"=="logs-prod" (
    docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f
    goto :eof
)

if "%1"=="shell" (
    docker compose exec anvil bash
    goto :eof
)

if "%1"=="test" (
    docker compose -f docker-compose.yml -f docker-compose.dev.yml exec anvil curl -f http://localhost:5000/
    goto :eof
)

if "%1"=="clean" (
    docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
    docker compose -f docker-compose.yml -f docker-compose.prod.yml down -v
    docker rmi -f !IMAGE_NAME!:!IMAGE_TAG! !FULL_IMAGE! 2>nul
    goto :eof
)

if "%1"=="push" (
    call :build
    docker tag !IMAGE_NAME!:!IMAGE_TAG! !FULL_IMAGE!
    docker push !FULL_IMAGE!
    goto :eof
)

if "%1"=="push-prod" (
    call :build-prod
    docker push !FULL_IMAGE!
    goto :eof
)

if "%1"=="dev-start" (
    call :up-dev
    echo Development stack started. Access at http://localhost:5000
    echo Run 'build.bat logs-dev' to view logs
    echo Run 'build.bat shell' to open a shell in the container
    goto :eof
)

if "%1"=="dev-stop" (
    docker compose -f docker-compose.yml -f docker-compose.dev.yml down
    goto :eof
)

if "%1"=="prod-start" (
    call :up-prod
    echo Production stack started. Access at http://localhost:5000
    goto :eof
)

if "%1"=="prod-stop" (
    docker compose -f docker-compose.yml -f docker-compose.prod.yml down
    goto :eof
)

if "%1"=="" (
    call :help
    goto :eof
)

echo Unknown command: %1
call :help
goto :eof

:build
docker build -t !IMAGE_NAME!:!IMAGE_TAG! .
exit /b 0

:build-prod
docker build -t !FULL_IMAGE! .
exit /b 0

:up-dev
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
exit /b 0

:up-prod
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
exit /b 0

:help
echo Anvil Docker Helper
goto :eof
