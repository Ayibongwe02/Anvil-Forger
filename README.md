# Anvil Forger - ML Model Training Platform

A collaborative web platform for training ML models, built with Flask and containerized with Docker.

## Features

- **Team-based collaboration** — Create teams and share projects with teammates via invite codes
- **Tabular ML** — Train classification and regression models on CSV data (scikit-learn)
- **Image classification** — Train image classifiers on labeled image folders
- **Model export** — Export as pickle, ONNX, or universal (dependency-free) formats
- **Model serving** — REST API to make predictions on any model
- **ONNX import** — Bring your own ONNX models and use them in Anvil

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (Flask development server)
python app.py
# Access: http://localhost:5000
```

### Docker Development

```bash
# Build image
docker build -t anvil-forger .

# Run container
docker run -d -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -e ANVIL_SECRET_KEY=your-secret-key \
  anvil-forger
```

### Docker Compose

```bash
# Development (with hot-reload)
docker-compose up

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## CI/CD Workflows

This project includes three automated GitHub Actions workflows:

### 1. Docker Build & Push (Docker Hub)
**File:** `.github/workflows/docker-build-push.yml`

Triggers on:
- `push` to `main` branch
- `pull_request` to `main` branch
- Manual trigger (`workflow_dispatch`)

Actions:
- Builds multi-stage Docker image
- On PR: builds only (no push)
- On push to main: builds and pushes to Docker Hub with tags:
  - `latest` (default branch)
  - `main` (branch name)
  - Git SHA prefix
  - Semantic version tags (if tagged)

**Required Secrets:**
- `DOCKERHUB_USERNAME` — Your Docker Hub username
- `DOCKERHUB_TOKEN` — Your Docker Hub Personal Access Token (not password!)

### 2. GitHub Container Registry (GHCR)
**File:** `.github/workflows/ghcr-build-push.yml`

Triggers on:
- `push` to `main` branch
- Manual trigger (`workflow_dispatch`)

Actions:
- Builds and pushes to GitHub Container Registry (`ghcr.io`)
- Uses GitHub's built-in `GITHUB_TOKEN` (no additional secrets needed)

**Required Secrets:**
- None (uses `secrets.GITHUB_TOKEN` automatically)

### 3. Tests
**File:** `.github/workflows/tests.yml`

Triggers on:
- `push` to `main` or `develop`
- `pull_request` to `main` or `develop`
- Manual trigger (`workflow_dispatch`)

Actions:
- Python 3.12 test matrix
- Installs dependencies from `requirements.txt`
- Runs linting (flake8)
- Runs pytest with coverage
- Uploads coverage to Codecov
- Builds Docker image and runs health check
- Checks image size

**Required Secrets:**
- None (optional: Codecov token for coverage reports)

## Setting Up Secrets

### Docker Hub

1. Create a Docker Hub account: https://hub.docker.com/signup
2. Generate a Personal Access Token:
   - Log in to Docker Hub
   - Account Settings → Security → New Access Token
   - Copy the token immediately
3. Add to GitHub:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Add `DOCKERHUB_USERNAME` with your Docker Hub username
   - Add `DOCKERHUB_TOKEN` with the token you copied

### GitHub Container Registry (GHCR)

No additional setup needed! GitHub automatically provides `secrets.GITHUB_TOKEN`.

## Deployment

### Deploy Latest from Docker Hub

```bash
docker pull ayibongwe02/anvil-forger:latest
docker run -d -p 5000:5000 \
  -v anvil-data:/app/data \
  -e ANVIL_SECRET_KEY=your-secret \
  ayibongwe02/anvil-forger:latest
```

### Deploy from GHCR

```bash
docker pull ghcr.io/Ayibongwe02/Anvil-Forger:latest
docker run -d -p 5000:5000 \
  -v anvil-data:/app/data \
  -e ANVIL_SECRET_KEY=your-secret \
  ghcr.io/Ayibongwe02/Anvil-Forger:latest
```

## Project Structure

```
.
├── .github/workflows/               # GitHub Actions workflows
│   ├── docker-build-push.yml       # Docker Hub build & push
│   ├── ghcr-build-push.yml         # GHCR build & push
│   ├── tests.yml                   # Tests & linting
│   └── deploy-railway.yml          # Railway deployment
├── src/                             # Source modules
│   ├── db.py                        # SQLite database
│   ├── auth.py                      # Authentication & teams
│   ├── tabular_training.py          # Tabular ML training
│   ├── image_training.py            # Image classification
│   ├── export_bundle.py             # Model export
│   ├── onnx_import.py               # ONNX model import
│   └── api_serving.py               # REST API blueprint
├── templates/                       # Flask templates
├── static/                          # CSS, JavaScript
├── data/                            # Datasets & models (volume mount)
├── app.py                           # Flask entry point
├── Dockerfile                       # Production multi-stage build
├── docker-compose.yml               # Development compose config
├── docker-compose.prod.yml          # Production compose config
├── requirements.txt                 # Python dependencies
└── .dockerignore                    # Docker build optimization
```

## Environment Variables

### Development

```bash
ANVIL_SECRET_KEY=dev-secret-key-change-in-production
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### Production

See `.env.production.example`:

```bash
ANVIL_SECRET_KEY=<strong-random-secret>
FLASK_ENV=production
DEBUG=False
```

## Database

Anvil uses **SQLite** (single-writer constraint) for simplicity. Data persists in `/app/data/` (mounted as a volume).

For multi-server deployments, migrate to PostgreSQL (edit `src/db.py`).

## Security

- ✅ Non-root user (`anvil:anvil`)
- ✅ Multi-stage Docker build (no build tools in runtime)
- ✅ Secrets via environment variables (not baked into image)
- ✅ Health checks for automatic recovery
- ✅ Resource limits (configurable in compose files)

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Image Size | ~924 MB | Includes all ML dependencies |
| Gunicorn Workers | 1 | SQLite single-writer constraint |
| Memory Limit | 2 GB (prod) | Adjustable |
| CPU Limit | 2 cores (prod) | Adjustable |
| Health Check | 30s interval | Tunable |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

All PRs trigger:
- Linting (flake8)
- Tests (pytest)
- Docker build (no push)

Merge to `main` triggers:
- Build and push to Docker Hub
- Build and push to GHCR
- Deploy to Railway (if configured)

## License

MIT

## Support

For issues or questions, open a GitHub issue or check the documentation in this repository.
