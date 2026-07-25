import sys
import os
import asyncio
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent / "PageIndex"))

from agents import Agent, Runner, function_tool, set_tracing_disabled
from pageindex import PageIndexClient

PDF_URL = "https://arxiv.org/pdf/2603.15031"
WORKSPACE = Path(__file__).parent / "workspace"
PDF_PATH = WORKSPACE / "attention-residuals.pdf"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="X-Y") with tight ranges.
Answer based only on tool output. Be concise.
"""

def query_agent(client, doc_id, prompt):
    @function_tool
    def get_document() -> str:
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_page_content],
        model=client.retrieve_model
    )
    
    async def _run():
        streamed_run = Runner.run_streamed(agent, prompt)
        async for event in streamed_run.stream_events():
            pass
        return str(streamed_run.final_output)

    return asyncio.run(_run())

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    
    set_tracing_disabled(True)
    
    # We map OpenAI to Ollama locally via litellm/openai client standard env vars
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"
    
    if not PDF_PATH.exists():
        print(f"[PageIndex Worker] Downloading {PDF_URL} ...")
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        r = requests.get(PDF_URL)
        with open(PDF_PATH, "wb") as f:
            f.write(r.content)
            
    client = PageIndexClient(workspace=WORKSPACE, model="ollama/qwen2.5:7b", retrieve_model="ollama/qwen2.5:7b")
    
    doc_id = next((did for did, doc in client.documents.items() if doc.get('doc_name') == PDF_PATH.name), None)
    if not doc_id:
        print("[PageIndex Worker] Indexing PDF for the first time... this may take a moment.")
        doc_id = client.index(PDF_PATH)
        
    print(f"[PageIndex Worker] Using Document ID: {doc_id}")
    print(f"[PageIndex Worker] Asking LLM Agent to reason over tree structure...")
    
    answer = query_agent(client, doc_id, query)
    print("\n[PageIndex Agent Answer]:")
    print(answer)

if __name__ == "__main__":
    main()
