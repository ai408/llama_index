"""Init file."""

from llama_index.embeddings.adapter import (
    AdapterEmbeddingModel,
    LinearAdapterEmbeddingModel,
)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.base import BaseEmbedding, SimilarityMode
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.clarifai import ClarifaiEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.embeddings.elasticsearch import (
    ElasticsearchEmbedding,
    ElasticsearchEmbeddings,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.embeddings.google_palm import GooglePaLMEmbedding
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
    HuggingFaceInferenceAPIEmbedding,
    HuggingFaceInferenceAPIEmbeddings,
)
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.huggingface_utils import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.llm_rails import LLMRailsEmbedding, LLMRailsEmbeddings
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.ollama_embedding import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.pooling import Pooling
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.embeddings.voyageai import VoyageEmbedding

__all__ = [
    "AdapterEmbeddingModel",  # Adapter嵌入模型
    "BedrockEmbedding",  # Bedrock嵌入
    "ClarifaiEmbedding",  # Clarifai嵌入
    "ClipEmbedding",  # Clip嵌入
    "CohereEmbedding",  # Cohere嵌入
    "BaseEmbedding",  # Base嵌入
    "DEFAULT_HUGGINGFACE_EMBEDDING_MODEL",  # 默认的HuggingFace嵌入模型
    "ElasticsearchEmbedding",  # Elasticsearch嵌入
    "FastEmbedEmbedding",  # FastEmbed嵌入
    "GoogleUnivSentEncoderEmbedding",  # GoogleUnivSentEncoder嵌入
    "GradientEmbedding",  # Gradient嵌入
    "HuggingFaceInferenceAPIEmbedding",  # HuggingFaceInferenceAPI嵌入
    "HuggingFaceEmbedding",  # HuggingFace嵌入
    "InstructorEmbedding",  # Instructor嵌入
    "LangchainEmbedding",  # Langchain嵌入
    "LinearAdapterEmbeddingModel",  # 线性Adapter嵌入模型
    "LLMRailsEmbedding",  # LLMRails嵌入
    "OpenAIEmbedding",  # OpenAI嵌入
    "AzureOpenAIEmbedding",  # AzureOpenAI嵌入
    "OptimumEmbedding",  # Optimum嵌入
    "Pooling",  # Pooling嵌入
    "GooglePaLMEmbedding",  # GooglePaLM嵌入
    "SimilarityMode",  # 相似度模式
    "TextEmbeddingsInference",  # 文本嵌入推理
    "resolve_embed_model",  # 解析嵌入模型
    # Deprecated, kept for backwards compatibility  # 弃用，为了向后兼容
    "LLMRailsEmbeddings",  # LLMRails嵌入
    "ElasticsearchEmbeddings",  # Elasticsearch嵌入
    "HuggingFaceInferenceAPIEmbeddings",  # HuggingFaceInferenceAPI嵌入
    "VoyageEmbedding",  # Voyage嵌入
    "OllamaEmbedding",  # Ollama嵌入
]
