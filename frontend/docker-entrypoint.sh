#!/bin/sh
set -eu

API_URL="${API_BASE_URL:-http://localhost:8000}"
ESCAPED_URL=$(printf '%s' "$API_URL" | sed 's/\\/\\\\/g; s/"/\\"/g')
printf 'window.__ENV__ = { API_BASE_URL: "%s" };\n' "$ESCAPED_URL" > /usr/share/nginx/html/env.js

exec nginx -g "daemon off;"
