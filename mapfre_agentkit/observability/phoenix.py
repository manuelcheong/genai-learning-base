from phoenix.otel import register
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from mapfre_agentkit.utils import Singleton
from mapfre_agentkit.observability.custom_processor import ForceCustomAttributes


class Phoenix(Singleton):

    def __init__(self, config: dict):
        set_global_textmap(
            CompositePropagator(
                [
                    TraceContextTextMapPropagator(),
                    W3CBaggagePropagator(),
                ]
            )
        )
        self.tracer_provider = register(
            endpoint=f"http://{config.get("host")}:{config.get("port")}/v1/traces",
            project_name=config.get("project_name", "default"),
            auto_instrument=True,
            set_global_tracer_provider=False,
        )
        self.tracer_provider.add_span_processor(
            ForceCustomAttributes(
                endpoint=f"http://{config.get("host")}:{config.get("port")}/v1/traces"
            )
        )

    def get_tracer_provider(self):
        return self.tracer_provider
