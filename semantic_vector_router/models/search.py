"""Search result models."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    """A single search result hit."""

    id: str
    score: float
    rerank_score: Optional[float] = None
    partition: str
    document: dict[str, Any]

    def __lt__(self, other: "SearchHit") -> bool:
        """Compare by score for sorting."""
        return (self.rerank_score or self.score) < (other.rerank_score or other.score)


class SearchResult(BaseModel):
    """Complete search result."""

    hits: list[SearchHit]
    query: str
    partitions_searched: list[str]
    total_candidates: int
    reranked: bool
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
