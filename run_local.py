from rag_pipeline.data_loader import CocktailLoader
from rag_pipeline.vector_store import CocktailVectorStore
from rag_pipeline.llm_interface import LocalLLM
from rag_pipeline.user_memory import UserMemory
from rag_pipeline.rag_engine import RAGEngine


def main():
    # Load and parse cocktails from CSV
    print("Loading cocktail data...")
    loader = CocktailLoader("data/cocktails.csv")
    cocktails = loader.load()

    # Initialize vector index
    print("Initializing vector store...")
    vector_store = CocktailVectorStore(cocktails)

    # Load LLM (path to downloaded .gguf model)
    print("Loading local LLM...")
    llm = LocalLLM("models/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.gguf")

    # User memory for ingredients
    memory = UserMemory()

    # Connect components via RAG engine
    engine = RAGEngine(vector_store, llm, memory)

    print("\nReady! Type your cocktail questions. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = engine.run(user_input)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
