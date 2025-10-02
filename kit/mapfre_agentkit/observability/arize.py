import os
import logging
from importlib.metadata import entry_points
from arize.otel import register
from mapfre_agentkit.utils import Singleton

logger = logging.getLogger(__name__)


class Arize(Singleton):
    """Singleton class to configure and manage the Arize observability tracer.

    This class initializes the Arize tracer provider and automatically discovers
    and applies available OpenInference instrumentors.
    """

    def __init__(self, config):
        """Initialize the Arize tracer and start the auto-instrumentation process.

        Args:
            config (dict): A dictionary containing the Arize-specific configuration,
                           such as 'arize_space_id' and 'arize_project_name'.
        """
        self.tracer_provider = register(
            space_id=config.get("arize_space_id"),
            api_key=os.getenv("ARIZE_API_KEY"),
            project_name=config.get("arize_project_name", "default"),
        )
        self._auto_instrument()

    def _auto_instrument(self):
        """Discover and apply available OpenInference instrumentors using entry points."""
        logger.info("Arize: Starting auto-instrumentation process...")
        openinference_entry_points = entry_points(group="openinference_instrumentor")

        if not openinference_entry_points:
            logger.warning(
                "No OpenInference instrumentors found. Skipping auto-instrumentation for Arize."
            )
            return

        for entry_point in openinference_entry_points:
            try:
                instrumentor_cls = entry_point.load()
                instrumentor = instrumentor_cls()
                instrumentor.instrument(tracer_provider=self.tracer_provider)
                logger.info(
                    f"Auto-instrumentation of '{entry_point.name}' for Arize enabled."
                )
            except Exception as e:
                logger.warning(
                    f"Auto-instrumentation of '{entry_point.name}' for Arize failed: {e}"
                )

    def get_tracer_provider(self):
        return self.tracer_provider
