import asyncio
import sys
from contextlib import AsyncExitStack
from typing import Optional

import groq
import openai
from mcp import ClientSession, StdioServerParameters, stdio_client


class MCPClient:
    def __init__(self):
        # managing the connection of the client
        self.session: Optional[ClientSession] = None
        # ensures resources are properly closed when not needed in async context
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

        # preparing the mcp client to run server's script
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # stdio client: launches the server script then opens a communication via stdio channel
        # passing server_params to stdio client here tells it what server commands to run in order to launch it
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        # stdio client returns a (reader, writer)
        # which let's the client read/write to the channel/server
        self.sdtio, self.write = stdio_transport

        # wraps the io streams (read/write) into mcp session object to handle tool invocation and lifecycle

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.sdtio, self.write))

        # initializes and starts the session
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

        return f"""
        Reasoning: {response.choices[0].message.reasoning}
        Response: {response.choices[0].message.content}
        """

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
