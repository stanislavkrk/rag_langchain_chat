from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate


class LocalLLM:
    """
    LocalLLM wraps a llama-cpp model using Langchain and handles prompt formatting and response generation.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, temperature: float = 0.5):
        """
        Initialize the local LLM model via llama-cpp and Langchain.

        :param model_path: Path to the .gguf model file.
        :param n_ctx: Max context size (tokens).
        :param temperature: Sampling temperature (0 = deterministic).
        """
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            temperature=temperature,
            verbose=False,
        )

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant that recommends real cocktails based on ingredients.\n"
                "Use only the information provided in the context below.\n"
                "Do not invent any ingredients or cocktails.\n\n"
                "Context:\n{context}\n\n"
                "User Question:\n{question}\n\n"
                "Answer:"
            )
        )

    def ask(self, context: str, question: str) -> str:
        """
        Generate an answer using the model based on given context and question.

        :param context: Retrieved information (e.g., similar cocktails).
        :param question: User query.
        :return: Model's answer as string.
        """
        prompt = self.prompt_template.format(context=context, question=question)
        return self.llm.invoke(prompt)

