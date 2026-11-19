# Public Data Scraping

The startup crawler:

1. Checks robots.txt for each source.
2. Loads `/sitemap.xml` and any referenced sitemap indexes.
3. Prioritizes project, property, apartment, villa, and listing pages.
4. Extracts visible text with Selectolax.
5. Parses sitemap XML with lxml.
6. Falls back to HTTPX, Jina Reader, and ScraperAPI.
7. Chunks text with overlap and preserves source URL and collection date.
8. Merges successful scrapes with reviewed seed data instead of replacing data with failed output.

Configure Render environment variables:

- `JINA_API_KEY`: Jina Reader bearer key
- `SCRAPERAPI_KEY`: ScraperAPI key
- `REFRESH_SOURCES_ON_START=true`
- `SCRAPE_MAX_DARGLOBAL=500`
- `SCRAPE_MAX_WASALT=1000`

Never commit API keys to source files or repository configuration. Store both keys as Render secret environment variables. Rotate a key if it was pasted into chat or another untrusted location.

Because a key can be exposed in a conversation, rotate both keys in their provider dashboards after deployment setup. Create separate keys for Jina and ScraperAPI, then replace the Render secret values.
