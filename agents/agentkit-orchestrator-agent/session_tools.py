import asyncio
import json
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import datetime
import os

# Librería asíncrona para Redis
import redis.asyncio as aioredis 
from redis.asyncio import Redis 

# --- 1. Definición del Modelo de Sesión ---
# Usar Pydantic para validar y serializar/deserializar datos de forma eficiente
class AgentSession(BaseModel):
    user_id: str
    last_interaction: str = Field(default_factory=lambda: "No date")
    history: List[Dict[str, str]] = Field(default_factory=list)
    state: Dict = Field(default_factory=dict)

# --- 2. Clase de Gestor de Sesiones Asíncrono ---
class RedisSessionManager:
    """Gestiona la conexión y las operaciones CRUD asíncronas con Redis."""
    
    # TTL (Time To Live) de la sesión en segundos (ej: 2 horas)
    TTL_SECONDS = 7200
    
    def __init__(self, host: str, port: int, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self._redis_pool: Optional[Redis] = None

    async def initialize(self):
        """Inicializa la conexión asíncrona (pool)."""
        if self._redis_pool is None:
            self._redis_pool = aioredis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True # Decodifica automáticamente a strings de Python
            )
            # Prueba de conexión asíncrona
            await self._redis_pool.ping()
            print("✅ Conexión asíncrona a Redis establecida.")

    async def close(self):
        """Cierra la conexión al finalizar la aplicación."""
        if self._redis_pool:
            await self._redis_pool.close()
            print("❌ Conexión a Redis cerrada.")

    async def save_session(self, session_id: str, session_data: AgentSession) -> bool:
        """Guarda una sesión."""
        
        # Serializa el objeto Pydantic a una cadena JSON
        json_string = session_data.model_dump_json()
        
        # Operación SET con expiración (EX)
        result = await self._redis_pool.set(
            session_id, 
            json_string, 
            ex=self.TTL_SECONDS
        )
        return result

    async def load_session(self, session_id: str) -> Optional[AgentSession]:
        """Carga una sesión y la deserializa, o devuelve None si no existe/expiró."""
        
        # Operación GET asíncrona
        json_string = await self._redis_pool.get(session_id)
        
        if json_string:
            # Deserializa la cadena JSON de vuelta al modelo Pydantic
            data = json.loads(json_string)
            return AgentSession(**data)
        
        return None

    async def get_history_by_user_id(self, user_id: str) -> List[Dict[str, str]]:
        """
        Busca todas las sesiones de un usuario y devuelve su historial combinado.

        Este método escanea las claves que coinciden con el patrón de sesión,
        carga cada sesión y, si el user_id coincide, agrega su historial
        a una lista consolidada.
        """
        user_history = []
        # Escanea todas las claves que podrían ser sesiones de agente
        async for key in self._redis_pool.scan_iter("agent:session:*"):
            session_data = await self.load_session(key)
            # Comprueba si la sesión pertenece al usuario solicitado
            if session_data and session_data.user_id == user_id:
                user_history.extend(session_data.history)
        
        return user_history

manager = RedisSessionManager(
    host='redis', # Cambiar a 'redis' si se ejecuta dentro de la red Docker
    port=6379,
    password=os.getenv('REDIS_PASS', '') 
)


async def save_session_user(session_id: str, session_data: AgentSession) -> bool:
    
    print(f"\nGuardando sesión: {session_id}")
    print(f"Datos de sesión: {session_data}")  
    try:
        await manager.initialize()

        # Ejemplo de datos de sesión
        new_session = AgentSession(
            user_id=session_id,
            last_interaction=datetime.datetime.utcnow().isoformat(),
            history=[{"role": "user", "text": session_data}]
        )

        # 1. Guardar la sesión
        saved = await manager.save_session(session_id, new_session)
        #print(f"\nEstado de guardado: {saved}")

    except Exception as e:
        print(f"\nOcurrió un error: {e}")
    finally:
        await manager.close()

async def load_session_user(user_id: str) -> Optional[AgentSession]:
    """Carga una sesión y la deserializa, o devuelve None si no existe/expiró."""

    try:
        await manager.initialize()

        # 2. Cargar la sesión
        loaded_sessions = await manager.get_history_by_user_id(user_id)

        return loaded_sessions

    except Exception as e:
        print(f"\nOcurrió un error: {e}")
    finally:
        await manager.close()
        return loaded_session.history