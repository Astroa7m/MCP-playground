import asyncio
import sys
from contextlib import AsyncExitStack
from typing import Optional

import groq
import openai
from mcp import ClientSession, StdioServerParameters, stdio_client


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # todo: use .env instead
        # self.client = groq.Groq(api_key=sys.argv[2])
        # or
        self.client = openai.OpenAI(
            api_key=sys.argv[2],
            base_url="https://api.groq.com/openai/v1"
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        self.sdtio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.sdtio, self.write))

        await self.session.initialize()

        # listing available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Processes query using ChatGroq"""

        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]

        response = await self.session.list_tools()

        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties":{
                        key: {
                            "type": value["type"],
                            "description": value.get("title", "")
                        }
                        for key, value in tool.inputSchema['properties'].items()
                    },
                    "required": list(tool.inputSchema['properties'].keys())
                }
            }
        } for tool in response.tools]

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            tools=available_tools,
            messages=messages,
            # stream=True
        )
        print(response)
        return "You said you were testing"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        pass
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
