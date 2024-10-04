SHELL := /bin/bash
DOCKER_COMPOSE := docker-compose -f ./stack/docker-compose.yml
SERVICE_NAME := sdn-ml-service

build: ## Build container
	@${DOCKER_COMPOSE} build

up: ## Start container
	@${DOCKER_COMPOSE} up -d

ps: ## List running containers
	@${DOCKER_COMPOSE} ps

down: ## Close running containers
	@${DOCKER_COMPOSE} down

logs: ## Show container's logs
	@${DOCKER_COMPOSE} logs ${SERVICE_NAME}

tail: ## Tail container's logs
	@${DOCKER_COMPOSE} logs -f ${SERVICE_NAME}

shell: ## Run shell in container
	@${DOCKER_COMPOSE} run --rm ${SERVICE_NAME} bash
