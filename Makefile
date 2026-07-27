.PHONY: help build build-dev build-prod up up-dev up-prod down logs logs-dev logs-prod shell test lint clean push push-dev push-prod

REGISTRY ?= docker.io
IMAGE_NAME ?= anvil
IMAGE_TAG ?= latest
FULL_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

help:
	@echo "Anvil Docker Makefile"
	@echo ""
	@echo "Build commands:"
	@echo "  make build              Build image for development"
	@echo "  make build-prod         Build optimized production image"
	@echo ""
	@echo "Development commands:"
	@echo "  make up-dev             Start development stack with hot reload"
	@echo "  make down               Stop all containers"
	@echo "  make logs-dev           View development logs"
	@echo "  make shell              Open shell in running container"
	@echo ""
	@echo "Production commands:"
	@echo "  make up-prod            Start production stack"
	@echo "  make logs-prod          View production logs"
	@echo ""
	@echo "Registry commands:"
	@echo "  make push               Push image to registry (REGISTRY, IMAGE_NAME, IMAGE_TAG)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean              Remove containers, volumes, and images"
	@echo "  make test               Run container health check"
	@echo ""
	@echo "Examples:"
	@echo "  make REGISTRY=ghcr.io IMAGE_NAME=myorg/anvil IMAGE_TAG=v1.0 push"
	@echo "  make up-dev"
	@echo "  make logs-dev"

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

build-prod:
	docker build -t $(FULL_IMAGE) --target runtime .

up-dev: build
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

up-prod: build-prod
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

up: up-dev

down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down

logs:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

logs-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

logs-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

shell:
	docker compose exec anvil bash

test:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml exec anvil curl -f http://localhost:5000/ || exit 1

lint:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml exec anvil python -m py_compile app.py src/**/*.py

clean:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down -v
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down -v
	docker rmi -f $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE) 2>/dev/null || true

push: build
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE)
	docker push $(FULL_IMAGE)

push-prod: build-prod
	docker push $(FULL_IMAGE)

# Development workflow
dev-start: up-dev
	@echo "Development stack started. Access at http://localhost:5000"
	@echo "Run 'make logs-dev' to view logs"
	@echo "Run 'make shell' to open a shell in the container"

dev-stop:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Production workflow
prod-start: up-prod
	@echo "Production stack started. Access at http://localhost:5000"

prod-stop:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down

# CI/CD helpers
ci-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

ci-test: ci-build
	docker compose -f docker-compose.yml -f docker-compose.dev.yml exec anvil python -m py_compile app.py

ci-push: ci-build
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE)
	docker push $(FULL_IMAGE)
