FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 kreguard
WORKDIR /app
COPY pyproject.toml README.md ./
COPY kreguard ./kreguard
RUN pip install --no-cache-dir .
COPY examples/kreguard.json examples/support_prompt.txt /config/

USER kreguard
EXPOSE 8787
# Mount your own policy over /config. A token is required to listen on 0.0.0.0.
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8787/healthz', timeout=2)"
CMD ["python", "-m", "kreguard", "serve", "--config", "/config/kreguard.json", "--host", "0.0.0.0", "--token-env", "KREGUARD_TOKEN"]
