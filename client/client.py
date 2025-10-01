import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from string import Template
from typing import Optional, Dict

import groq
import openai
from mcp import ClientSession, StdioServerParameters, stdio_client

continue_calling_prompt = Template("""
Right now you are a planning agent, your job is to decide whether the current tool result fully resolves the user’s query or not from previous interactions.

INSTRUCTION: You are NOT calling a tool. You are ONLY returning a decision in JSON format.
Do NOT wrap your response in a tool call. Do NOT use a tool name like "JSON".
Just return a plain JSON object as described below.

If it doesn't resolve the user's question reply strictly with the following JSON format, the arguments are provided to you witin the tools
replace the below templates with the correct potential_next_tool name, args, and their values:
{
    "continue": True,
    "potential_next_tool": "here tool name string",
    "function": {
        "arguments": {
            "arg1":"arg1 value",
            "arg2": 0,
            "arg3": null
        }
    }
}

Otherwise reply:
{
    "continue": False,
    "potential_next_tool": null
    "function": null
}
Here is the current tool result:
${tool_results}
""")

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

        command = "uv" if is_python else "node"

        full_dir = os.path.dirname(server_script_path)  # "/path/tp/server/"
        server_file = os.path.basename(server_script_path)  # e.g. "server.py"

        # preparing the mcp client to run server's script
        server_params = StdioServerParameters(
            command=command,
            args=[
                "--directory",
                full_dir,
                "run",
                server_file
            ]
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

    async def get_and_format_tools(self):
        response = await self.session.list_tools()

        available_tools = []
        for tool in response.tools:
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema  # already JSON Schema
                }
            })
        return available_tools

    async def prompt_llm(self, messages, model="openai/gpt-oss-120b"):
        return self.client.chat.completions.create(
            model=model,
            tools=await self.get_and_format_tools(),
            messages=messages,
        )

    async def call_function(self, tool_name: str, tool_args: Dict, session: ClientSession) -> Dict:
        """Calls a tool and returns its result as a dictionary."""
        tool_name = tool_name
        tool_args = tool_args
        # tool call
        print(f"Executing: {tool_name}")
        result = await session.call_tool(tool_name, tool_args)
        tool_results = {"call": tool_name, "result": result}
        print(f"Done executing: {tool_name}")
        return tool_results

    async def process_query(self, query: str, messages) -> str:
        """Processes query using ChatGroq"""

        messages.append({
            "role": "user",
            "content": query,
        })

        response = await self.prompt_llm(messages)

        # Process response and handle tool calls
        final_text = []
        # getting the response (only one no need for a loop)
        choice = response.choices[0]
        if choice.finish_reason == 'stop':  # text
            final_text.append(choice.message.content)
        elif choice.finish_reason == 'tool_calls':
            # calling the tool and getting the results
            tool_name = choice.message.tool_calls[0].function.name
            tool_args = json.loads(choice.message.tool_calls[0].function.arguments or "{}")

            tool_results = await self.call_function(tool_name,tool_args, self.session)
            print("first tool")
            print(tool_results["result"].content[0].text, '\n')
            messages.append(
                {"role": "system",
                 "content": f"Called function: {tool_name}, with args: {tool_args}, and result:\n{json.dumps(tool_results["result"].content[0].text)}"}
            )
            # before continuing, we check if we need to call another tool
            # creating a copy as we want the decision to be internal (not within message history)
            messages_copied = messages[::]
            messages_copied.append(
                {"role": "system", "content": continue_calling_prompt.substitute(tool_results=tool_results)}
            )
            response = await self.prompt_llm(messages_copied)
            ## the LLM will return either json response as described in continue_calling_prompt above
            should_call_next_tool_obj = json.loads(response.choices[0].message.content)
            if should_call_next_tool_obj['continue']:
                tool_name = should_call_next_tool_obj["potential_next_tool"]
                tool_args = should_call_next_tool_obj['function']['arguments'] or "{}"
                # should call next tool
                tool_results = await self.call_function(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    session=self.session
                )
                messages.append(
                    {"role": "system",
                     "content": f"Called function: {tool_name}, with args: {tool_args}, and result:\n{json.dumps(tool_results["result"].content[0].text)}"}
                )
            # getting the next response
            response = await self.prompt_llm(messages)
            # # adding last assistant to the list
            last_message = response.choices[0].message.content
            messages.append({
                "role": "assistant",
                "content": last_message
            })
            final_text.append(last_message)
        # return "\n".join(final_text)
        return str(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        messages = []

        while True:

            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query, messages)
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
