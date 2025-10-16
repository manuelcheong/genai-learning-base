import os
import yaml
from typing import List, Dict

# --- Punto de entrada para ejecutar el servidor ---

def discover_agents_by_label(
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

if __name__ == "__main__":
    AGENT_SERVICES = discover_agents_by_label()
    print(f"Discovered agents: {AGENT_SERVICES}")
