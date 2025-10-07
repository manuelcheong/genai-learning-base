# my_client.py
from fastmcp import Client
import asyncio
import json

ACCESS_TOKEN=""

print("--- Creating Client ---")



async def search_repository(repositorio: str) -> str:
    print(f"Searching for repository: {repositorio}")
    config = {
        "mcpServers": {
            "protected_api": {
                "transport": "http",  # Or "http" depending on the server
                "url": "https://api.githubcopilot.com/mcp/",
                    # CRITICAL: Define the 'Authorization' header here
                "headers": {
                    "Authorization": f"Bearer {ACCESS_TOKEN}"
                }
            }
        }
    }

    client = Client(config)

    async with client:

        print(f"Client configured to connect to: {client}")
        search_result = await client.call_tool(
            "search_repositories", {
                "minimal_output": True,
                "query": repositorio
            }
        ) 
                
        print(f"Resultado de search repositories: {search_result}")
        return search_result

async def get_content(repositorio: str, path: str) -> str:
    print(f"Searching for repository: {repositorio}")
    config = {
        "mcpServers": {
            "protected_api": {
                "transport": "http",  # Or "http" depending on the server
                "url": "https://api.githubcopilot.com/mcp/",
                    # CRITICAL: Define the 'Authorization' header here
                "headers": {
                    "Authorization": f"Bearer {ACCESS_TOKEN}"
                }
            }
        }
    }

    client = Client(config)

    async with client:

        print(f"Client configured to connect to: {client}")
        get_content = await client.call_tool(
            "get_file_contents", {
                "owner": "manuelcheong",
                "path": f"/{path}",
                "ref": "",
                "repo": repositorio
            }
        ) 
                
        print(f"Resultado de search repositories: {get_content}")
        return get_content    
