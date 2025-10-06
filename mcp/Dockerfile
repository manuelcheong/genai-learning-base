FROM ghcr.io/github/github-mcp-server:latest AS mcp-server

# Usa una imagen base con Node.js para instalar mcp-proxy
FROM node:20-slim

# Instala mcp-proxy
RUN npm install -g mcp-proxy

# Copia el binario del servidor MCP desde la primera etapa
COPY --from=mcp-server /server/github-mcp-server /usr/local/bin/github-mcp-server

# Arranque por defecto: servidor MCP envuelto con streamable-http en el puerto 8090
CMD ["mcp-proxy", "--transport=streamablehttp", "--port=8090", "github-mcp-server", "server"]
