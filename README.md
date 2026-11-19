# DarGlobal & Wasalt Applied AI Chatbot

A public-data retrieval chatbot for the Junior AI Engineer assessment. It returns sourced answers about DarGlobal and Wasalt and clearly avoids inventing unavailable prices or listing status.

## Run locally

```bash
docker compose up --build
```

Open `http://localhost:8000`. The chat works in grounded demo mode without a model key. To use OpenRouter:

```bash
export OPENROUTER_API_KEY="your-key"
docker compose up --build
```

## Tests

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Data approach

The scraper checks public robots rules and uses a descriptive user agent. It preserves source URLs, source names, and collection dates. Live scraping failures leave the reviewed seed data intact. The application never sends scraped content or API keys to the browser.

The chatbot uses keyword retrieval and OpenRouter server-side. A demo answer is used when no key is configured or OpenRouter is unavailable.

## Deployment

The Docker image exposes port 8000 and `/health`. GitHub Actions run tests and build/push the image to GHCR. Deployment credentials and the public hosting target are environment-specific.

## Cloud evidence

The repository includes Kubernetes deployment/service manifests, health probes, resource requests/limits, secret injection, CI/CD automation, and Dockerized operations. Extend them with the target account's ingress, TLS, DNS, Terraform, and monitoring configuration.

## Security and safety

- API keys are server-side environment variables.
- `.env` files are ignored.
- User prompts are capped at 1000 characters.
- Answers are source-grounded and avoid invented commercial facts.
- Docker runs a health check and the chat has no database or file-write surface.
