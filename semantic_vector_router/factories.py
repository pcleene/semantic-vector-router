"""Factory functions for creating embedders and rerankers."""

from typing import Literal, Optional, cast

from semantic_vector_router.config import get_api_key
from semantic_vector_router.embedders.base import BaseEmbedder
from semantic_vector_router.embedders.cohere import CohereEmbedder
from semantic_vector_router.embedders.huggingface import HuggingFaceEmbedder
from semantic_vector_router.embedders.openai import OpenAIEmbedder
from semantic_vector_router.embedders.voyage import VoyageEmbedder
from semantic_vector_router.exceptions import ConfigurationError
from semantic_vector_router.models import (
    EmbeddingProvider,
    RerankerProvider,
    SVRConfig,
)
from semantic_vector_router.rerankers.base import BaseReranker
from semantic_vector_router.rerankers.cohere import CohereReranker
from semantic_vector_router.rerankers.voyage import VoyageReranker
from semantic_vector_router.utils.rate_limiter import RateLimiterRegistry


def create_embedder(
    config: SVRConfig,
    rate_limiter_registry: RateLimiterRegistry,
) -> BaseEmbedder:
    """Create the appropriate embedder based on config."""
    provider = config.embedding.provider
    model = config.embedding.model
    api_key_env = config.embedding.api_key_env
    timeout_s = config.resilience.embedding_timeout_ms / 1000.0

    if provider == EmbeddingProvider.OPENAI:
        api_key = get_api_key(api_key_env, "OpenAI")
        assert api_key is not None
        embedder = OpenAIEmbedder(
            model=model,
            api_key=api_key,
            dimensions=config.embedding.dimensions,
            timeout=timeout_s,
        )
        embedder.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return embedder

    elif provider == EmbeddingProvider.VOYAGE:
        api_key = get_api_key(api_key_env, "Voyage AI")
        assert api_key is not None
        output_dimension = cast(
            Optional[Literal[256, 512, 1024, 2048]],
            config.embedding.voyage_output_dimension,
        )
        output_dtype: Literal[
            "float", "int8", "uint8", "binary", "ubinary"
        ] = config.embedding.voyage_quantization.value  # type: ignore[assignment]

        voyage_embedder = VoyageEmbedder(
            model=model,
            api_key=api_key,
            input_type="query",
            output_dimension=output_dimension,
            output_dtype=output_dtype,
            timeout=timeout_s,
        )
        voyage_embedder.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return voyage_embedder

    elif provider == EmbeddingProvider.COHERE:
        api_key = get_api_key(api_key_env, "Cohere")
        assert api_key is not None
        cohere_embedder = CohereEmbedder(
            model=model,
            api_key=api_key,
            input_type=cast(
                Literal[
                    "search_query", "search_document",
                    "classification", "clustering",
                ],
                config.embedding.input_type or "search_query",
            ),
            timeout=timeout_s,
        )
        cohere_embedder.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return cohere_embedder

    elif provider == EmbeddingProvider.HUGGINGFACE:
        hf_embedder = HuggingFaceEmbedder(
            model=model,
            device=config.embedding.device or "cpu",
        )
        hf_embedder.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return hf_embedder

    else:
        raise ConfigurationError(f"Unknown embedding provider: {provider}")


def create_reranker(
    config: SVRConfig,
    rate_limiter_registry: RateLimiterRegistry,
) -> BaseReranker:
    """Create the appropriate reranker based on config."""
    provider = config.reranking.provider
    model = config.reranking.model
    api_key_env = config.reranking.api_key_env
    timeout_s = config.resilience.reranking_timeout_ms / 1000.0

    if provider == RerankerProvider.VOYAGE:
        api_key = get_api_key(api_key_env, "Voyage AI")
        voyage_reranker = VoyageReranker(model=model, api_key=api_key, timeout=timeout_s)
        voyage_reranker.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return voyage_reranker

    elif provider == RerankerProvider.COHERE:
        api_key = get_api_key(api_key_env, "Cohere")
        cohere_reranker = CohereReranker(model=model, api_key=api_key, timeout=timeout_s)
        cohere_reranker.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return cohere_reranker

    else:
        raise ConfigurationError(f"Unknown reranker provider: {provider}")


def create_document_embedder(
    config: SVRConfig,
    rate_limiter_registry: RateLimiterRegistry,
) -> BaseEmbedder:
    """Create an embedder configured for document embedding.

    For asymmetric embeddings (Voyage 4), uses the document model.
    For Cohere, uses input_type="search_document".
    For other providers, reuses the standard embedder config.
    """
    provider = config.embedding.provider
    timeout_s = config.resilience.embedding_timeout_ms / 1000.0
    api_key_env = config.embedding.api_key_env

    if provider == EmbeddingProvider.VOYAGE:
        api_key = get_api_key(api_key_env, "Voyage AI")
        assert api_key is not None
        doc_model = config.embedding.effective_document_model
        output_dimension = cast(
            Optional[Literal[256, 512, 1024, 2048]],
            config.embedding.voyage_output_dimension,
        )
        output_dtype: Literal[
            "float", "int8", "uint8", "binary", "ubinary"
        ] = config.embedding.voyage_quantization.value  # type: ignore[assignment]

        doc_voyage_embedder = VoyageEmbedder(
            model=doc_model,
            api_key=api_key,
            input_type="document",
            output_dimension=output_dimension,
            output_dtype=output_dtype,
            timeout=timeout_s,
        )
        doc_voyage_embedder.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return doc_voyage_embedder

    elif provider == EmbeddingProvider.COHERE:
        api_key = get_api_key(api_key_env, "Cohere")
        assert api_key is not None
        doc_cohere_embedder = CohereEmbedder(
            model=config.embedding.model,
            api_key=api_key,
            input_type="search_document",
            timeout=timeout_s,
        )
        doc_cohere_embedder.set_rate_limiter(
            rate_limiter_registry.get(provider.value)
        )
        return doc_cohere_embedder

    else:
        # For OpenAI, HuggingFace, etc. — same embedder for query and document
        return create_embedder(config, rate_limiter_registry)
