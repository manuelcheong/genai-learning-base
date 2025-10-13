import os
import uvicorn
from mapfre_agentkit.agents.generators.agent_a2a_generator import AgentA2AFactory
from mapfre_agentkit.observability.observability import (
    Observability,
)
from mapfre_agentkit.observability.middleware import ObservabilityMiddleware
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from opentelemetry.instrumentation.starlette import StarletteInstrumentor

# from arize.otel import register

# Import and configure the automatic instrumentor from OpenInference
# from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from dotenv import load_dotenv


load_dotenv()

# tracer_provider = register(
#    space_id=os.getenv("ARIZE_SPACE_ID"),
#    api_key=os.getenv("ARIZE_API_KEY"),
#    project_name="CHEF",
#    batch=False,
# )
#
#
## Finish automatic instrumentation
# GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)


def main():
    try:
        Observability()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "agent_config.yaml")

        factory = AgentA2AFactory(config_path=config_path)

        agent = factory.build_agent()
        agent_card = factory.build_agent_card()
        executor = factory.build_executor(agent, agent_card)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        app = server.build()
        app.add_middleware(
            ObservabilityMiddleware,
            public_paths=["/.well-known/agent.json", "/.well-known/agent-card.json"],
        )

        # StarletteInstrumentor().instrument_app(app)

        print("🚀 Starting agent ...")
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Application stopped by user.")
