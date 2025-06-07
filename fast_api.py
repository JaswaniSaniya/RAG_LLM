
import uvicorn
from fastapi import FastAPI
from rag_agent import doc_analyze

app = FastAPI()


@app.post("/ask")
async def analyze_document(query: str):
    """
    Endpoint to interact with the RAG Based APP.
    """
    try:
        # Format the message in the structure agent expects
        analyzer = doc_analyze()
        response = analyzer.get_response(query)
        return {"response": str(response)}
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app,  port=8000)