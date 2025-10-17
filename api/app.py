import os
import uvicorn
from fastapi import FastAPI, APIRouter, Body, Query, Request, WebSocket, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from mapfre_agentkit.a2a.client.app_client import A2AGatewayClient
from mapfre_agentkit.a2a.interceptors.headers_propagation_interceptor import (
    HeadersPropagationInterceptor,
)

from a2a.types import TextPart, FilePart, FileWithUri, FileWithBytes
import yaml
from typing import List, Dict

import uuid
import mimetypes
from minio import Minio
from minio.error import S3Error
import base64
import io
import json
#from datetime import timedelta
import time
from google import genai

# --- Configuración ---
# URL por defecto del agente (se puede sobrescribir)
DEFAULT_AGENT_CARD_URL = os.environ.get("AGENT_CARD_URL", "http://image-analyzer-agent:8080")

# ----------------------------------------------------------------------------
# Agent Client Manager
# ----------------------------------------------------------------------------


class AgentClientManager:
    """
    Gestiona múltiples clientes A2A, permitiendo la conexión dinámica a diferentes agentes.
    Mantiene un caché de clientes para reutilizar conexiones.
    """

    def __init__(self):
        self._clients: Dict[str, A2AGatewayClient] = {}
        self._interceptors = [HeadersPropagationInterceptor()]

    async def get_or_create_client(self, agent_url: str) -> A2AGatewayClient:
        """
        Obtiene un cliente existente del caché o crea uno nuevo si no existe.

        Args:
            agent_url (str): La URL del Agent Card del agente.

        Returns:
            A2AGatewayClient: El cliente configurado para el agente.
        """
        if agent_url not in self._clients:
            print(f"Creating new client for agent: {agent_url}")
            try:
                client = await A2AGatewayClient.from_card_url(
                    agent_url, interceptors=self._interceptors
                )
                self._clients[agent_url] = client
                print(f"Client created successfully for: {agent_url}")
            except Exception as e:
                print(f"Error creating client for {agent_url}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to connect to agent at {agent_url}: {str(e)}",
                )
        return self._clients[agent_url]

    async def remove_client(self, agent_url: str) -> bool:
        """
        Elimina un cliente del caché y cierra su conexión.

        Args:
            agent_url (str): La URL del Agent Card del agente.

        Returns:
            bool: True si el cliente fue eliminado, False si no existía.
        """
        if agent_url in self._clients:
            print(f"Removing client for agent: {agent_url}")
            await self._clients[agent_url].aclose()
            del self._clients[agent_url]
            return True
        return False

    def list_agents(self) -> list[str]:
        """
        Retorna la lista de URLs de agentes actualmente conectados.

        Returns:
            list[str]: Lista de URLs de agentes.
        """
        return list(self._clients.keys())

    async def close_all(self):
        """Cierra todos los clientes y limpia el caché."""
        print("Closing all agent clients...")
        for url, client in self._clients.items():
            print(f"Closing client for: {url}")
            await client.aclose()
        self._clients.clear()
        print("All clients closed.")


# Variable global para el gestor de clientes
agent_manager: Optional[AgentClientManager] = None

# ----------------------------------------------------------------------------
# MinioUploader
# ----------------------------------------------------------------------------

class MinioUploader:
    """
    Gestiona la conexión y subida de archivos a un bucket de MinIO.
    """
    def __init__(self):
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        secure = False #os.getenv("MINIO_USE_SECURE", "False").lower() == "true"

        if not all([endpoint, access_key, secret_key]):
            raise ValueError("Las variables de entorno de MinIO no están configuradas.")

        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            print("✅ Conectado a MinIO exitosamente.")
        except Exception as e:
            print(f"❌ Error al conectar con MinIO: {e}")
            raise

        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "media1")
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Asegura que el bucket exista, si no, lo crea."""
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                }
            ]
        }
        
        try:
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                self.client.make_bucket(self.bucket_name)
                self.client.set_bucket_policy(self.bucket_name, json.dumps(bucket_policy))
                print(f"Bucket '{self.bucket_name}' creado.")
            else:
                print(f"Bucket '{self.bucket_name}' ya existe.")
        except S3Error as e:
            print(f"❌ Error al verificar/crear el bucket: {e}")
            raise

    def upload_file(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Sube un archivo en bytes a MinIO y retorna su URL pública.

        Args:
            file_bytes (bytes): El contenido del archivo.
            mime_type (str): El tipo MIME del archivo (ej: 'image/jpeg').

        Returns:
            str: La URL pública del archivo subido.
        """
        try:
            # Generar un nombre de objeto único para evitar colisiones
            extension = mimetypes.guess_extension(mime_type) or ''
            object_name = f"{uuid.uuid4()}{extension}"

            # Subir el objeto
            self.client.put_object(
                self.bucket_name,
                object_name,
                io.BytesIO(file_bytes),
                length=len(file_bytes),
                content_type=mime_type
            )
            print(f"✅ Archivo subido exitosamente a MinIO:")
            # Construir la URL pública
            file_url = (
                f"http://"
                f"minio:9000/"
                f"{self.bucket_name}/{object_name}"
            )
            
            print(f"🚀 Archivo subido exitosamente a: {file_url}")
            return file_url

        except S3Error as e:
            print(f"❌ Error al subir el archivo a MinIO: {e}")
            raise HTTPException(status_code=500, detail="Error al subir el archivo.")



# -----------------
    # Upload content
# -----------------

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def upload_file_google_file_api(data_file, mime_type):
    """Sube el video a Google File API y luego lo envía al agente."""

    print(f"📤 Subiendo video a Google File API")

    # Subir el archivo a Google
    try:
        uploaded_file = client.files.upload(file=data_file, config={"mime_type": mime_type})
    except Exception as e:
        print(f"❌ Error al subir el archivo a Google: {e}")
        raise HTTPException(status_code=500, detail="Error al subir el archivo.")


    print(f"✅ Video subido: {uploaded_file.name}")
    print(f"   URI: {uploaded_file.uri}")

    # Esperar a que el archivo esté ACTIVE
    print(f"⏳ Esperando a que el archivo esté listo...")
    while uploaded_file.state.name != "ACTIVE":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
        print(f"   Estado: {uploaded_file.state.name}")
    print(f"✅ Archivo listo para usar")

    print(f"✅ Completado")

    return uploaded_file.uri



# ----------------------------------------------------------------------------
# FastAPI router factory to expose the gateway as API endpoints
# ----------------------------------------------------------------------------


def build_router(manager: AgentClientManager) -> APIRouter:
    """
    Builds a FastAPI APIRouter to expose multi-agent capabilities via an HTTP API.

    The router exposes the following endpoints:
      - `GET /agents`: Lists all connected agents.
      - `GET /card`: Returns an agent's Agent Card (requires agent_url param).
      - `POST /messages`: Sends a message to an agent (requires agent_url param).
      - `POST /agents/connect`: Manually connects to a new agent.
      - `DELETE /agents/disconnect`: Disconnects from an agent.

    Args:
        manager (AgentClientManager): The agent client manager instance.

    Returns:
        APIRouter: A configured FastAPI router with the endpoints.
    """
    router = APIRouter()

    @router.get("/agents")
    async def list_agents() -> dict[str, Any]:
        """
        Lista todos los agentes actualmente conectados.

        Returns:
            dict[str, Any]: Diccionario con la lista de URLs de agentes.
        """
        return {"agents": manager.list_agents(), "count": len(manager.list_agents())}

    @router.get("/discover")
    async def discover_agents(
        compose_file: str = '/app/docker-compose.yml',
        label_key: str = 'agent.owner',
        label_value: str = 'mak'
    ) -> List[Dict]:
        """
        Lee docker-compose.yml y extrae servicios marcados con un label específico.
        
        Args:
            compose_file: Ruta al docker-compose.yml
            label_key: Clave del label a buscar (ej: 'app.type')
            label_value: Valor del label (ej: 'agent')
        
        Returns:
            Lista de diccionarios con información de cada agente descubierto
        """
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            discovered_agents = []
            
            for service_name, service_config in services.items():
                labels = service_config.get('labels', {})
                
                # Manejar tanto formato lista como diccionario
                label_dict = {}
                if isinstance(labels, list):
                    # Formato: ['app.type=agent', 'app.agent.name=hello']
                    for label in labels:
                        if '=' in label:
                            key, val = label.split('=', 1)
                            label_dict[key] = val
                elif isinstance(labels, dict):
                    # Formato: {'app.type': 'agent', 'app.agent.name': 'hello'}
                    label_dict = labels
                
                # Filtrar por label
                if label_dict.get(label_key) == label_value:
                    agent_info = {
                        'service_name': service_name,
                        'internal_url': f'http://{service_name}:{label_dict.get("app.agent.port", "8080")}',
                        'agent_name': label_dict.get('app.agent.name', service_name),
                        'port': label_dict.get('app.agent.port', '8080'),
                        'labels': label_dict
                    }
                    discovered_agents.append(agent_info)
                    print(f"✅ Discovered agent: {service_name} ({agent_info['agent_name']})")
            
            return discovered_agents
        
        except FileNotFoundError:
            print(f"❌ docker-compose.yml not found at {compose_file}")
            return []
        except Exception as e:
            print(f"❌ Error parsing docker-compose.yml: {e}")
            return []

    @router.post("/agents/connect")
    async def connect_agent(
        body: dict = Body(..., description="{ agent_url: str }")
    ) -> dict[str, Any]:
        """
        Conecta manualmente a un nuevo agente.

        Args:
            body (dict): Debe contener 'agent_url' con la URL del Agent Card.

        Returns:
            dict[str, Any]: Confirmación de la conexión.
        """
        agent_url = body.get("agent_url")
        if not agent_url:
            raise HTTPException(status_code=400, detail="agent_url is required")

        gateway = await manager.get_or_create_client(agent_url)
        return {
            "status": "connected",
            "agent_url": agent_url,
            "agent_name": (
                gateway._card.name if hasattr(gateway._card, "name") else "unknown"
            ),
        }

    @router.delete("/agents/disconnect")
    async def disconnect_agent(
        agent_url: str = Query(..., description="URL of the agent to disconnect")
    ) -> dict[str, Any]:
        """
        Desconecta de un agente y limpia su sesión.

        Args:
            agent_url (str): La URL del Agent Card del agente.

        Returns:
            dict[str, Any]: Confirmación de la desconexión.
        """
        removed = await manager.remove_client(agent_url)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Agent {agent_url} not found")
        return {"status": "disconnected", "agent_url": agent_url}

    @router.get("/card")
    async def get_card(
        agent_url: str = Query(
            DEFAULT_AGENT_CARD_URL,
            description="URL of the agent's Agent Card",
        )
    ) -> dict[str, Any]:
        """
        Obtiene el Agent Card de un agente específico.

        Args:
            agent_url (str): La URL del Agent Card del agente.

        Returns:
            dict[str, Any]: El Agent Card en formato diccionario.
        """
        gateway = await manager.get_or_create_client(agent_url)
        return gateway._card.model_dump(exclude_none=True)

    @router.post("/messages")
    async def post_message(
        body: dict = Body(
            ...,
            description="{ message: str, contextId?, metadata?, agent_url? }",
        ),
        stream: bool = Query(False, description="Enable streaming if supported"),
        agent_url: str = Query(
            DEFAULT_AGENT_CARD_URL,
            description="URL of the agent to send the message to",
        ),
        request: Request = None,
    ):
        """
        Endpoint to send a message to a specific agent and receive a response.

        Permite seleccionar dinámicamente el agente mediante el parámetro agent_url.
        Cada agente mantiene su propia sesión mediante el contextId.

        Supports two modes:
        1.  **Non-streaming** (default): Returns a single JSON response with the final message.
        2.  **Streaming** (if `stream=true` and supported by the agent): Returns a
            `text/event-stream` response (Server-Sent Events) with intermediate
            events and the final message.

        Args:
            body (dict): The request body, containing the message to send.
                        Opcionalmente puede incluir 'agent_url' para sobrescribir el query param.
            stream (bool): A query parameter to enable streaming mode.
            agent_url (str): Query param con la URL del agente (puede ser sobrescrito en body).
            request (Request): The FastAPI request object.

        Returns:
            Union[JSONResponse, StreamingResponse]: A JSON response or a streaming SSE response.
        """
        # Permitir que agent_url venga en el body o en query params
        target_agent_url = body.get("agent_url", agent_url)

        # Obtener o crear el cliente para este agente
        gateway = await manager.get_or_create_client(target_agent_url)

        text = str(body.get("messages", ""))
        kind = str(body.get("kind", "text"))
        attachment = body.get("attachment", "")
        context_id = body.get("context_id")
        metadata = body.get("metadata") or {}

        content = []
        if text:
            content.append(TextPart(text=text))
        if attachment:
            mime_type = attachment.split(':')[1].split(';')[0]
            attachment_bytes = attachment.split(',')[1]
            content.append(FilePart(file=FileWithBytes(bytes=attachment_bytes, mime_type=mime_type )))

        # Agregar información del agente al metadata
        metadata["agent_url"] = target_agent_url

        #headers = dict(request.headers)
        headers = {"x-mapfre-session-id": "pruebaaaaa"}
        call_context = gateway.build_propagation_context(headers)
        print(f"Context: {call_context}")

        if stream and gateway.supports_streaming():

            async def event_gen() -> AsyncGenerator[bytes, None]:
                async for evt in gateway.send_message(
                    content=content,
                    context_id=context_id,
                    metadata=metadata,
                    context=call_context,
                ):
                    # SSE framing: event + data
                    event_name = evt.get("event", "message")
                    payload = evt.get("data", {})
                    yield f"event: {event_name}\n".encode()
                    yield f"data: {JSONResponse(content=payload).body.decode()}\n\n".encode()

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        # Non-streaming path
        last_payload: Optional[dict[str, Any]] = None
        async for evt in gateway.send_message(
            content=content,
            context_id=context_id,
            metadata=metadata,
            context=call_context,
        ):
            if evt.get("event") == "message":
                last_payload = evt.get("data")
        return JSONResponse(content=last_payload or {})

    return router


# --- Lógica de la aplicación ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida del AgentClientManager.
    Lo crea al iniciar la app, opcionalmente conecta al agente por defecto,
    monta el router y cierra todas las conexiones al terminar.
    """
    global agent_manager
    print("Initializing Agent Client Manager...")

    agent_manager = AgentClientManager()

    # Inicializa el uploader de MinIO y lo adjunta al estado de la app
    try:
        app.state.minio_uploader = MinioUploader()
    except Exception as e:
        print(f"FATAL: No se pudo inicializar MinIO Uploader. La subida de archivos no funcionará. Error: {e}")
        app.state.minio_uploader = None # Asegura que el atributo exista aunque sea nulo
    

    # Conectar opcionalmente al agente por defecto si está configurado
    if DEFAULT_AGENT_CARD_URL:
        try:
            print(f"Connecting to default agent at: {DEFAULT_AGENT_CARD_URL}")
            await agent_manager.get_or_create_client(DEFAULT_AGENT_CARD_URL)
            print("Default agent connected successfully.")
        except Exception as e:
            print(f"Warning: Could not connect to default agent: {e}")
            print(
                "The API will still work, but you'll need to specify agent_url in requests."
            )

    # Construye el router y lo añade a la aplicación
    api_router = build_router(agent_manager)
    app.include_router(api_router, prefix="/agent")
    print("Multi-agent API router included.")

    yield  # La aplicación se ejecuta aquí

    print("Shutting down Agent Client Manager...")
    if agent_manager:
        await agent_manager.close_all()
    print("All agent connections closed.")




# --- Creación de la App FastAPI ---

app = FastAPI(
    title="Multi-Agent API Gateway",
    description="Exposes multiple MAPFRE AgentKit agents via a RESTful API with dynamic agent selection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", summary="Root endpoint for health checks")
async def read_root():
    """Endpoint raíz para verificar que el servicio está activo."""
    connected_agents = agent_manager.list_agents() if agent_manager else []
    return {
        "status": "ok",
        "message": "Multi-Agent API Gateway is running",
        "connected_agents": connected_agents,
        "agent_count": len(connected_agents),
    }

# ----------------------------------------------------------------------------
### - WebSocket ### 
# ----------------------------------------------------------------------------

@app.websocket("/ws/message")
async def websocket_message_endpoint(websocket: WebSocket):
    """
    Gestiona una conexión WebSocket para la comunicación bidireccional con agentes.
    - Cada conexión WebSocket crea su propio `AgentClientManager` para aislar las sesiones.
    - Escucha mensajes JSON del cliente, los reenvía al agente especificado y
      devuelve las respuestas del agente a través del WebSocket.
    """
    await websocket.accept()
    #print("WebSocket connection accepted.")
    #print(f"Client connected from {websocket.client}")
    # Crea un gestor de clientes específico para esta sesión de WebSocket.
    ws_agent_manager = AgentClientManager()
    print("WebSocket connection accepted. A new agent manager has been created for this session.")

    # Accede al uploader de MinIO desde el estado de la aplicación
    minio_uploader: MinioUploader = websocket.app.state.minio_uploader
    print(f"MinIO Uploader: {minio_uploader}")
    if not minio_uploader:
        print("WARNING: MinIO Uploader no está disponible. La subida de archivos fallará.")

    try:
        while True:
            # Espera a recibir un mensaje JSON del cliente (frontend)
            data = await websocket.receive_json()

            #print(f"Received message from client: {data}")
            
            # Extrae los datos necesarios del mensaje
            target_agent_url = data.get("agent_url", DEFAULT_AGENT_CARD_URL)
            if not target_agent_url:
                await websocket.send_json({"error": "agent_url is required"})
                continue
            
            text = str(data.get("messages", ""))
            attachment = data.get("attachment", "")
            context_id = data.get("context_id")
            metadata = data.get("metadata", {})
            kind = str(data.get("kind", "text"))

            
            #print(f"WS message received for agent: {data}")

            try:
                # Obtiene o crea un cliente A2A para el agente solicitado
                gateway = await ws_agent_manager.get_or_create_client(target_agent_url)

                # Construye el contenido del mensaje (texto y/o archivo)
                content = []
                if text:
                    content.append(TextPart(text=text))
                if attachment:
                    #print(f"Attachment: {attachment}")
                    # 1. Extraer mime_type y datos en base64
                    header, encoded = attachment.split(',', 1)
                    mime_type = header.split(':')[1].split(';')[0]
                    print(f"Mime type: {mime_type}")
                    # 2. Decodificar de base64 a bytes
                    file_bytes = base64.b64decode(encoded)
                    
                    decoded_bytes = io.BytesIO(file_bytes)

                    #print(f"File bytes: {file_bytes}")
                    
                    # 3. Subir el archivo a MinIO y obtener la URL
                    #file_url = minio_uploader.upload_file(file_bytes, mime_type)
                    file_url = upload_file_google_file_api(decoded_bytes, mime_type)

                    """ presigned_url = minio_uploader.client.get_presigned_url(
                        "GET",
                        minio_uploader.bucket_name,
                        file_url.split("/")[-1],
                        expires=timedelta(days=1),
                    ) """

                    print(f"Presigned URL: {file_url}")
                    
                    # 4. Enviar la URL al agente en lugar de los bytes
                    content.append(FilePart(file=FileWithUri(uri=f"{file_url}")))
                    #content.append(FilePart(file=FileWithBytes(bytes=encoded, mime_type=mime_type)))

                    print(f"Content: {content}")

                    """ mime_type = attachment.split(':')[1].split(';')[0]
                    attachment_bytes = attachment.split(',')[1]
                    content.append(FilePart(file=FileWithBytes(bytes=attachment_bytes, mime_type=mime_type))) """

                headers = {"x-mapfre-session-id": context_id}
                call_context = gateway.build_propagation_context(headers)
                #print(f"Context: {call_context}")
                #print(f"Content: {content}")

                # Envía el mensaje al agente y retransmite cada evento de vuelta al cliente WS
                async for evt in gateway.send_message(
                    content=content,
                    context_id=context_id,
                    metadata=metadata,
                    context=call_context,  # No hay headers HTTP para propagar en WS
                ):
                    await websocket.send_json(evt)

            except ConnectionError as e:
                # Si falla la conexión con el agente, informa al cliente WS
                print(f"Error connecting to agent via WebSocket: {e}")
                await websocket.send_json({"error": str(e), "event": "error"})
            
            except Exception as e:
                # Captura otros posibles errores durante el envío
                print(f"An unexpected error occurred in WebSocket: {e}")
                await websocket.send_json({"error": f"An unexpected error occurred: {e}", "event": "error"})


    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    
    finally:
        # Al desconectar, cierra todas las conexiones de agentes creadas en esta sesión
        print("Closing agent clients for the disconnected WebSocket session...")
        await ws_agent_manager.close_all()
        print("WebSocket session cleanup complete.")

# ----------------------------------------------------------------------------

# --- Punto de entrada para ejecutar el servidor ---

if __name__ == "__main__":
    # Este servidor correría en otro puerto, por ejemplo, 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)
