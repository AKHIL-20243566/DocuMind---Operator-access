"""
DocuMind RAG-LLM Assistant — Main entry point.
Run with: uvicorn main:app --reload --port 8000
"""
import logging
import uvicorn
from app.api.routes import app
from app.pipeline.rag_pipeline import RAGPipeline
from config.settings import API_HOST, API_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG pipeline on startup
pipeline = RAGPipeline()
pipeline.initialize()

if __name__ == "__main__":
    logger.info(f"Starting DocuMind API on {API_HOST}:{API_PORT}")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)