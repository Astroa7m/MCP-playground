import json
import os
from typing import List

import chromadb
from fastmcp import FastMCP
from data_setup import collection_name, client_file_name, SCHMEA_SUMMARY_FILE
mcp = FastMCP("aware_agentic_rag")


client = chromadb.PersistentClient(path=client_file_name)
collection = client.get_collection(collection_name)

@mcp.tool()
def aou_retrieval_tool(query: str, source_files: list[str] = ('FAQ.csv', 'FAQ2.csv'), n_result: int = 6):
    """
    Retrieves relevant information from ChromaDB based on a user query and optional source file filter.

    This tool is designed to answer questions related to the Arab Open University (AOU),
    including topics such as tutors, modules, faculty, and FAQs.

    Default behavior:
      - If no specific source files are provided, the tool defaults to FAQ sources:
        ['FAQ.csv', 'FAQ2.csv']
      - This ensures general questions are answered even when the query is ambiguous.

    Usage:
      - Before calling this tool, use `get_csv_schema_summary()` to inspect available CSV sources.
      - Based on the query, select the most relevant files and pass their names via `source_files`.

    Filtering:
      - The query will be restricted to documents originating from the specified files.
      - If multiple files are relevant, pass them as a list:
          source_files = ["tutors.csv", "faculty.csv"]

    Fallback behavior:
      - If the query requires broader context or spans multiple domains, pass None to source_files.
      - This will trigger a full search across all indexed documents without filtering by file.
      - Use this when the model is unsure which file is best or needs additional information.

    Notes:
      - if source_files, then it must contain filenames exactly as used during ingestion. Do not include other metadata keys such as 'columns' or nested structures.
      - The number of results returned (`n_result`) can be adjusted manually.

    Example:
      aou_retrieval_tool("Who is Alaa?", source_files=["tutors.csv"])
      aou_retrieval_tool("What is the grading policy?", source_files=None)
    """

    where_clause = None
    if source_files:
        where_clause = {"source_file": {"$in": source_files}}


    results = collection.query(
        # query_embeddings=[query_embedding],
        query_texts=query,
        n_results=n_result,
        where=where_clause,
        include=["documents"] # for debugging I could append ["metadatas", "data"]
    )
    # docs = results["documents"][0]

    return results

@mcp.tool()
def get_csv_schema_summary():
    """
    Retrieves a summary of all available CSV files in the knowledge corpus.

    Each entry in the returned list contains:
      - 'file': the filename (e.g., 'faculty_profiles.csv')
      - 'columns': a list of column headers present in that file

    Format:
      [
        {'file': 'filename.csv', 'columns': ['name', 'role', 'bio']},
        {'file': 'survey_data.csv', 'columns': ['respondent_id', 'rating', 'feedback']},
        ...
      ]

    Use this tool before answering any knowledge-based query.
    It helps determine which file is most relevant, and provides metadata
    for filtering when calling `aou_retrieval_tool(query, metadata)`.
    """
    with open(SCHMEA_SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

#
# @mcp.tool()
# def firecrawl_web_search_tool(query: str) -> List[str]:
#     """
#     Search for information related to the user query, using afirecrawl tool.
#     You can use this tool to scrape the internet, or get information about something not within your
#     knowledge, to further help the user.
#
#     :param query: user query
#     :return: list of strings of most relevant web searches
#     """
#
#     url = "https://api.firecrawl.dev/v2/search"
#
#     payload = {
#         "query": query,
#         "timeout": 60000
#     }
#
#     load_dotenv()
#
#     headers = {
#         "Authorization": f"Bearer {os.getenv('firecrawl_api')}",
#         "Content-Type": "application/json"
#     }
#
#     response = requests.request("POST", url, json=payload, headers=headers)
#
#     return response.text



if __name__ == "__main__":
    # run the server
    mcp.run(transport='stdio')