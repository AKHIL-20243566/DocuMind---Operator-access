"""Script to build/rebuild the vector database from raw documents."""
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.pipeline.rag_pipeline import RAGPipeline
from config.settings import DATA_RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Building vector database...")
    logger.info(f"Raw documents directory: {DATA_RAW_DIR}")

    pipeline = RAGPipeline()
    pipeline.initialize(doc_dir=DATA_RAW_DIR)

    logger.info(f"Vector database built with {len(pipeline.documents)} documents")
    logger.info("Done!")


if __name__ == "__main__":
    main()