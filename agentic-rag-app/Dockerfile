# Single-container image that runs both FastAPI backend and Streamlit UI behind nginx
# - FastAPI (uvicorn) on internal port 8001
# - Streamlit on internal port 8501
# - nginx listens on 80 and reverse-proxies /api to FastAPI and / to Streamlit

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps (nginx) and cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Write top-level nginx config (user / events / http wrapper that includes sites-enabled)
RUN cat > /etc/nginx/nginx.conf <<'NGINX_CONF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;
 events { worker_connections 1024; }

http {
    # map for websocket proxying (used by Streamlit)
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Include site configs placed into sites-enabled
    include /etc/nginx/sites-enabled/*;
}
NGINX_CONF
# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure shell scripts use LF line endings inside image (fix CRLF from Windows)
RUN if [ -f /app/start.sh ]; then sed -i 's/\r$//' /app/start.sh; fi

# Put nginx site config in place
RUN mkdir -p /etc/nginx/sites-enabled
COPY docker/nginx.conf /etc/nginx/sites-enabled/default

# Ensure start script is executable
RUN chmod +x /app/start.sh

# Expose the external port
EXPOSE 80

# Start both services via start.sh (starts uvicorn, streamlit, then nginx)
CMD ["/app/start.sh"]
