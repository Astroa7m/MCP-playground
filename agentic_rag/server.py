from typing import List
import os

import chromadb
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from vector_db_setup import collection_name, client_file_name

mcp = FastMCP("aou_faq_collection", host="localhost", port="8080")


embedding = HuggingFaceEndpointEmbeddings()
client = chromadb.PersistentClient(path=client_file_name)
collection = client.get_collection(collection_name)

@mcp.tool()
def aou_retrieval_tool(query: str):
    """
    Retrieves the most relevant information about the AOU (Arab Open University)
    It returns answer from FAQ collection, so use it when you encounter a related question
    :param query: user query to retrieve the most relevant document
    :return: most relevant documents retrieved from vector db
    """
    # make sure we run vector_db_setup before
    query_embedding = embedding.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    docs = results["documents"][0]

    return docs

@mcp.tool()
def firecrawl_web_search_tool(query: str) -> List[str]:
    """
    Search for information related to the user query, using afirecrawl tool.
    You can use this tool to scrape the internet, or get information about something not within your
    knowledge, to further help the user.

    :param query: user query
    :return: list of strings of most relevant web searches
    """

    url = "https://api.firecrawl.dev/v2/search"

    payload = {
        "query": query,
        "timeout": 60000
    }

    load_dotenv()

    headers = {
        "Authorization": f"Bearer {os.getenv('firecrawl_api')}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text


if __name__ == "__main__":
    # run the server
    print("Starting server")
    mcp.run(transport='stdio')

