from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core import Settings
from typing import List
import os
from rag_pipeline.data_loader import Cocktail

from .custom_embedder import CustomHFEmbedder


class CocktailVectorStore:
    """
    Handles indexing and retrieval of cocktail data using LlamaIndex.
    """

    def __init__(self, documents: List[Cocktail], persist_dir: str = "storage"):
        """
        Initialize the vector store. Loads existing index if present, otherwise builds a new one.

        :param documents: List of Cocktail objects to index.
        :param persist_dir: Directory to store or load the vector index.
        """
        self.persist_dir = persist_dir

        Settings.embed_model = CustomHFEmbedder(model_name="intfloat/e5-small-v2")

        if os.path.exists(persist_dir):
            # Load existing index from disk
            self.index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))
        else:
            # Build new index from cocktail data
            llama_docs = self._convert_to_documents(documents)
            self.index = VectorStoreIndex.from_documents(llama_docs)
            self.index.storage_context.persist(persist_dir)

    def _convert_to_documents(self, cocktails: List[Cocktail]) -> List[Document]:
        """
        Convert Cocktail objects into LlamaIndex-compatible Document objects.

        :param cocktails: List of Cocktail objects.
        :return: List of Document objects.
        """
        docs = []
        for cocktail in cocktails:
            text = self._format_cocktail(cocktail)
            doc = Document(text=text, metadata={"name": cocktail.name})
            docs.append(doc)
        return docs

    def _format_cocktail(self, cocktail: Cocktail) -> str:
        """
        Convert a single cocktail into a clean text block for LLM context.

        :param cocktail: A Cocktail instance.
        :return: Text description of the cocktail.
        """
        ingredients = ", ".join([
            f"{(m or '').strip()} {(i or '').strip()}"
            for i, m in zip(cocktail.ingredients, cocktail.measures)
        ]) if cocktail.ingredients and cocktail.measures else ", ".join(cocktail.ingredients or [])

        return (
            f"Name: {cocktail.name}\n"
            f"Alcoholic: {cocktail.alcoholic}\n"
            f"Category: {cocktail.category}\n"
            f"Glass: {cocktail.glass}\n"
            f"Ingredients: {ingredients}\n"
            f"Instructions: {cocktail.instructions}\n"
        )

    def query(self, question: str, top_k: int = 3) -> str:
        """
        Search for relevant cocktails based on user question.

        :param question: The natural language question.
        :param top_k: Number of top documents to retrieve.
        :return: Focused text context to pass into LLM.
        """
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)
        return "\n\n".join([n.get_content() for n in nodes])
