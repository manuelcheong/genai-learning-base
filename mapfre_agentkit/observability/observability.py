import os
from arize.otel import register
from mapfre_agentkit.utils import load_config, Singleton
from mapfre_agentkit.observability.arize import Arize
from mapfre_agentkit.observability.phoenix import Phoenix


class Observability(Singleton):

    def __init__(self):
        self.config = load_config()
        self.phoenix = None
        self.arize = None
        monitoring = self.config.get("monitoring")
        if monitoring:
            phoenix_config = monitoring.get("phoenix")
            arize_config = monitoring.get("arize")

            if phoenix_config:
                self.phoenix = Phoenix(config=phoenix_config)
            if arize_config:
                self.arize = Arize(config=arize_config)

    def get_tracers_provider(self):
        tracers = []
        if self.phoenix:
            tracers.append(self.phoenix.get_tracer_provider())
        if self.arize:
            tracers.append(self.arize.get_tracer_provider())
        return tracers
