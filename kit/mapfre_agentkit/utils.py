import asyncio
import threading
import os
import yaml
import logging
import importlib

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    pass


def safe_async_run(coro):
    """Simple wrapper to safely run async code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():

            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    result = asyncio.run(coro)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def expand_env_vars(env_dict):
    """
    Expande los valores del diccionario usando variables de entorno solo si el valor es una clave de entorno existente.
    Si la variable no existe en el entorno, deja el valor literal.
    """
    result = {}
    for k, v in env_dict.items():
        if isinstance(v, str) and v in os.environ:
            result[k] = os.getenv(v)
        else:
            logger.warning(f"Environment variable {v} not found")
    return result


def get_protocol():
    """
    Returns the default protocol ("http" or "https") depending on the environment.

    If the environment variable IS_LOCAL is set to a truthy value ("1", "true", "yes"; case-insensitive),
    the protocol will be "http". Otherwise, it will be "https".

    Returns:
        str: "http" if running locally, otherwise "https".
    """
    is_local = os.getenv("IS_LOCAL", "false").lower() in ("1", "true", "yes")
    if is_local:
        protocol = "http"
    else:
        protocol = "https"
    return protocol


def load_config() -> dict:
    """Load configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    path = os.getenv("CONFIG_PATH", "agent_config.yaml")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
