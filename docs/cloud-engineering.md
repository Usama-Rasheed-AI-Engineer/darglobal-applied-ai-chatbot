# Cloud Engineering Evidence

## Linux and automation
The application is Python-based and includes shell-compatible Docker, Compose, and CI commands. Scraping, testing, and container health checks are reproducible.

## Containers and orchestration
Docker packages the application and `/health` probe. Kubernetes manifests define two replicas, readiness/liveness probes, service networking, resources, and an OpenRouter secret.

## CI/CD
GitHub Actions install dependencies, run tests, set up Buildx, authenticate to GHCR, and publish the image. Pull requests build without pushing.

## Security
Secrets are injected at runtime and excluded from Git. Prompts are bounded and commercial facts are never invented. The public source corpus records source URL and collection date.

## Operations
Monitor container health, API latency, error rate, memory/CPU use, and model availability. Application logs should be centralized in a later production environment.

## Runbook
1. Check `/health`.
2. Inspect container or Kubernetes pod logs.
3. Verify the data source file and source URLs.
4. Verify the OpenRouter key and model status.
5. Restart the service only after dependency checks.
6. Record incident time, impact, root cause, correction, and follow-up.
