from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr


class CustomHFEmbedder(BaseEmbedding):
    """
    HuggingFace embedding without sentence-transformers, Pydantic-compatible
    """

    _device: str = PrivateAttr()
    _tokenizer: PreTrainedTokenizer = PrivateAttr()
    _model: PreTrainedModel = PrivateAttr()

    def __init__(self, model_name: str = "intfloat/e5-small-v2", device: str = None):
        super().__init__()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self._device)


    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            model_output = self._model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy().tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([f"passage: {text}"])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([f"query: {query}"])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
