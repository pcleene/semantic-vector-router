"""Embedding provider implementations."""

from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.embedders.cohere import CohereEmbedder
from semantic_vector_router.embedders.huggingface import HuggingFaceEmbedder
from semantic_vector_router.embedders.openai import OpenAIEmbedder
from semantic_vector_router.embedders.voyage import VoyageEmbedder

__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
    "VoyageEmbedder",
    "CohereEmbedder",
    "HuggingFaceEmbedder",
]
