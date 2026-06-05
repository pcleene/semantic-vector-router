# Semantic Vector Router - Architecture & Design Documentation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Project Structure](#project-structure)
5. [Core Components Deep Dive](#core-components-deep-dive)
6. [Data Flow](#data-flow)
7. [Design Decisions & Rationale](#design-decisions--rationale)
8. [Configuration System](#configuration-system)
9. [Embedding System](#embedding-system)
10. [Routing & Merging](#routing--merging)
11. [Lifecycle Management](#lifecycle-management)
12. [Error Handling](#error-handling)
13. [Resilience & Retry](#resilience--retry)
14. [CLI Architecture](#cli-architecture)
15. [Metadata Store & Distributed State](#metadata-store--distributed-state)
16. [Detection Pipeline](#detection-pipeline)
17. [Repartition Engine](#repartition-engine)
18. [Observability](#observability)
19. [Testing Strategy](#testing-strategy)
20. [Future Considerations](#future-considerations)

---

## Executive Summary

**Semantic Vector Router (SVR)** is a Python SDK that solves the problem of degraded search quality in large, heterogeneous vector indexes. It automatically:

1. **Partitions** vector collections based on metadata fields (views or source collection with pre-filtering)
2. **Routes** queries to relevant partitions in parallel
3. **Merges** results with score normalization
4. **Reranks** using cross-encoder models for unified relevance

### Key Features

| Feature | Description |
|---------|-------------|
| **Three Index Modes** | VIEWS mode (index per partition), SOURCE mode (single index with filtering), FIELDS mode (per-partition embedding fields) |
| **Voyage 4 Support** | Shared embedding space with asymmetric embeddings for optimal cost/latency |
| **Multiple Embedders** | OpenAI, Voyage AI, Cohere, HuggingFace (local) |
| **Cross-Encoder Reranking** | Voyage AI and Cohere rerankers for unified relevance |
| **Lifecycle Management** | Auto-provisioning, change streams, health monitoring, auto-split |
| **Retry & Resilience** | `@with_retry` decorator with exponential backoff, `ResilienceConfig` for all timeouts, connection health checks |
| **Full CLI** | 11 command groups with subcommands: init, partitions, search, analyze, watch, split, config, index, monitor, repartition, cache |
| **Metadata Store** | `svr_metadata` collection for partition state, operations, and distributed locks |
| **Detection Pipeline** | 5 signal types (threshold breach, approaching, skew, underpopulated, stale) with COLLECT-STORE-ANALYZE-DECIDE workflow |
| **Repartition Engine** | 5-step zero-downtime repartitioning with resume and rollback |
| **Structured Logging** | JSON formatter (`SVRLogFormatter`), `ContextVar` correlation IDs, `log_operation` timing decorator |
| **Metrics Hooks** | `MetricsHandler` protocol, `MetricsCollector` with 12 metric types, pluggable backends |
| **Embedding Cache** | Thread-safe LRU + TTL cache (`EmbeddingCache`), `CacheKey` keyed by text/model/dimensions/input_type |
| **Pool Tuning** | Configurable `maxPoolSize`, `minPoolSize`, `maxIdleTimeMS`, `waitQueueTimeoutMS` |
| **Time-based Splitting** | Split by time buckets (yearly, monthly, weekly) in addition to secondary field and hash strategies |

The SDK is designed to be **transparent** to application code - developers specify which partitions to search, and the SDK handles all the complexity of parallel execution, result merging, and optional reranking.

---

## Problem Statement

Vector search quality degrades under two distinct (but often co-occurring) conditions. Understanding which problem you're facing helps determine the right solution.

---

### Problem 1: Scale -- Very Large Collections

**When it occurs:** Vector collections exceed ~10 million vectors, regardless of content homogeneity.

**Why it happens:**

| Issue | Explanation |
|-------|-------------|
| **ANN algorithm limitations** | Approximate Nearest Neighbor algorithms (HNSW, IVF) trParts Distributor accuracy for speed. As the search space grows, the approximation error increases--the algorithm may miss truly relevant vectors that exist in distant graph regions. |
| **Quantization error accumulation** | Binary and scalar quantization (used to reduce memory) introduce small errors per vector. With millions of vectors, these errors compound, making similarity scores less reliable for distinguishing "close" from "very close" matches. |
| **Index fragmentation** | Large indexes may span multiple shards or memory segments, introducing latency variance and potential inconsistencies in scoring across segments. |

**Symptoms:**
- Recall@k metrics degrParts Distributor over time without code changes
- p95 latency increases disproportionately to data growth
- Relevant results appear inconsistently (sometimes found, sometimes missed)

**The scale threshold isn't fixed**--it depends on your accuracy requirements, hardware, and index configuration. A 50M vector index might work fine for "good enough" recommendations but fail for precision-critical legal document retrieval.

---

### Problem 2: Heterogeneity -- Semantically Diverse Content

**When it occurs:** A single collection contains fundamentally different types of content (e.g., product descriptions, support tickets, legal contracts, marketing copy all indexed together).

**Why it happens:**

| Issue | Explanation |
|-------|-------------|
| **Semantic space pollution** | Embedding models map text to a continuous vector space where similar meanings cluster together. When you mix unrelated domains, the space becomes crowded with irrelevant neighbors. A query for "court filing deadline" might return results about "basketball court reservations" because both contain "court." |
| **Score distribution mismatch** | Different content types produce different similarity score distributions. Technical documentation might cluster tightly (scores 0.85-0.95), while creative marketing copy spreads widely (0.60-0.90). Comparing scores across these distributions is meaningless. |
| **Embedding model bias** | Models trained on general web text may not represent all domains equally well. Legal terminology, medical jargon, or code snippets might cluster poorly compared to conversational text. |

**Symptoms:**
- Irrelevant results appear in top-k that clearly don't belong (e.g., support ticket when searching products)
- Search quality varies dramatically depending on query type
- Users report "search is getting worse" even with same data volume
- High similarity scores (0.8+) for clearly irrelevant content

**Heterogeneity can hurt at any scale**--even a 100k vector index mixing legal documents with product descriptions will show degraded quality compared to separate indexes.

---

### The Compound Problem

In practice, **both problems often occur together**. An e-commerce company might have:
- 50M product embeddings (scale problem)
- Mixed with 5M support tickets (heterogeneity problem)
- Mixed with 2M blog posts (more heterogeneity)
- All in one index for "unified search"

The result: poor relevance, frustrated users, and engineers unsure whether to blame the embedding model, the index configuration, or the data itself.

---

### Pre/Post Filtering: Not a Solution

A common response is to use metadata filtering, but both approaches have fundamental tradeoffs:

| Approach | How it works | Problem |
|----------|--------------|---------|
| **Pre-filtering** | Filter documents BEFORE ANN search (e.g., `category = "electronics"`) | Disrupts ANN algorithm. HNSW graph was built on ALL vectors--filtering first means traversing a broken graph with missing edges. Can miss relevant results entirely. |
| **Post-filtering** | Run ANN on full index, THEN filter results | If only 1% of your index matches the filter, ANN might return 100 candidates with only 1 relevant result. You asked for 10 results but only got 1. |

**The right solution is partitioning**: separate indexes for semantically coherent subsets, with intelligent routing at query time.

---

### The Standard Advice

MongoDB and other vector database vendors recommend manual sharding/partitioning, but this requires developers to:

- Decide partition boundaries manually
- Create and manage multiple indexes
- Write application-level routing logic
- Handle result merging across partitions

### What SVR Provides

SVR automates all of this, providing:

- **Automatic partition management** based on user-defined metadata fields
- **Transparent query routing** without application logic changes
- **Result merging with reranking** for unified relevance scores
- **Lifecycle management** for detecting new partition values and auto-provisioning

---

## Solution Architecture

### High-Level Architecture

```
+---------------------------------------------------------------------+
|                        Application Code                              |
|                                                                      |
|   results = await svr.search(query, partitions=["electronics"])     |
+-------------------------------+-------------------------------------+
                                |
                                v
+---------------------------------------------------------------------+
|                          SVRClient                                   |
|  +--------------+  +--------------+  +--------------+  +-----------+ |
|  |  Embedder    |  |  Resolver    |  |   Backend    |  | Reranker  | |
|  |  (BYOM)      |  | (Routing)    |  |  (MongoDB)   |  | (Optional)| |
|  +------+-------+  +------+-------+  +------+-------+  +-----+-----+ |
+--------+---------------+----------------+------------------+---------+
         |                |                |                  |
         v                v                v                  v
+---------------------------------------------------------------------+
|                        MongoDB Atlas                                 |
|                                                                      |
|   +--------------+  +--------------+  +--------------+              |
|   | View:        |  | View:        |  | View:        |              |
|   | electronics  |  | furniture    |  | clothing     |              |
|   | + Vector Idx |  | + Vector Idx |  | + Vector Idx |              |
|   +--------------+  +--------------+  +--------------+              |
|                              |                                       |
|                              v                                       |
|                    +------------------+                              |
|                    | Source Collection |                              |
|                    |    (products)     |                              |
|                    +------------------+                              |
+---------------------------------------------------------------------+
```

### Key Architectural Principles

1. **Flexible index strategies**: Choose VIEWS mode (index per partition), SOURCE mode (single index with filtering), or FIELDS mode (per-partition embedding fields with dedicated indexes).

2. **Async-first**: All I/O operations are async using PyMongo's native async support. Parallel fan-out to multiple partitions.

3. **Config-driven**: All behavior controlled by a configuration file. No code changes needed to adjust partitioning.

4. **Pluggable components**: Abstract interfaces for backends, embedders, and rerankers allow easy extension.

5. **Transparent routing**: Application code doesn't need to know about partition implementation details.

6. **Resilient by default**: Exponential backoff retries, configurable timeouts, connection health checks, and watcher auto-reconnect.

7. **Observable**: Structured logging with correlation IDs, pluggable metrics hooks, and operation timing.

---

## Project Structure

```
semantic_vector_router/
├── __init__.py                 # Package exports (SVRClient, models, exceptions)
├── client.py                   # Main SVRClient class - primary user interface
├── config.py                   # Configuration loading/saving/validation
├── models.py                   # Pydantic models for config and data structures
├── exceptions.py               # Custom exception hierarchy
│
├── backends/                   # Database backend implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract BaseBackend interface
│   ├── mongodb.py              # MongoDB Atlas implementation
│   └── metadata.py             # MetadataStore (svr_metadata collection) — partition/operation/lock CRUD
│
├── embedders/                  # Embedding provider implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract BaseEmbedder interface
│   ├── openai.py               # OpenAI embeddings
│   ├── voyage.py               # Voyage AI embeddings (including Voyage 4 asymmetric)
│   ├── cohere.py               # Cohere embeddings
│   └── huggingface.py          # HuggingFace sentence-transformers (local)
│
├── routing/                    # Query routing and result handling
│   ├── __init__.py
│   ├── resolver.py             # Partition name → PartitionInfo resolution (async, dual-source)
│   └── merger.py               # Result merging and score normalization
│
├── rerankers/                  # Cross-encoder reranking implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract BaseReranker interface
│   ├── voyage.py               # Voyage AI reranker
│   └── cohere.py               # Cohere reranker
│
├── lifecycle/                  # Partition lifecycle management
│   ├── __init__.py
│   ├── scanner.py              # Scans collection for partition values
│   ├── provisioner.py          # Creates views, indexes, and partitions (with rollback)
│   ├── watcher.py              # Change stream monitoring (with auto-reconnect)
│   ├── monitor.py              # Health and threshold monitoring
│   ├── splitter.py             # Auto-split for overgrown partitions
│   ├── detector.py             # Detection pipeline (5 signals, distributed lock)
│   └── repartition.py          # Repartition engine (5-step workflow, resume, rollback)
│
├── utils/                      # Cross-cutting utilities
│   ├── __init__.py
│   ├── retry.py                # @with_retry decorator (exponential backoff + jitter)
│   ├── logging.py              # Structured logging (JSON formatter, correlation IDs, log_operation)
│   ├── metrics.py              # Metrics hooks (MetricsHandler protocol, MetricsCollector, 12 metric types)
│   ├── cache.py                # Embedding cache (LRU + TTL, thread-safe)
│   └── field_analyzer.py       # Field analysis for filter-suitable fields (SOURCE mode)
│
└── cli/                        # Command-line interface (Click)
    ├── __init__.py             # CLI entry point — registers 11 command groups
    ├── helpers.py              # Shared utilities (_run_async, _get_backend, handle_config_error)
    ├── init.py                 # Interactive + non-interactive setup wizard
    ├── partitions.py           # Partition management (list, status, create, delete, refresh, scan, provision)
    ├── search.py               # Vector search from CLI
    ├── analyze.py              # Collection field analysis
    ├── watch.py                # Change stream monitoring (start, status, confirm, reject, confirm-all)
    ├── split.py                # Partition splitting (check, execute)
    ├── config_cmd.py           # Config management (show, validate, set, path)
    ├── index.py                # Index management (status, rebuild, wait)
    ├── monitor.py              # Detection pipeline CLI (check, history)
    ├── repartition.py          # Repartition workflow CLI (pending, execute, status, rollback)
    └── cache.py                # Embedding cache management (stats)
```

### File Purposes

| File | Purpose |
|------|---------|
| `client.py` | Main entry point. Orchestrates all components. Users interact with `SVRClient`. |
| `models.py` | Pydantic models for all config sections and data structures. Type safety and validation. |
| `config.py` | Configuration loading from file/dict, saving, validation, env var resolution. |
| `exceptions.py` | Custom exception hierarchy with 16 specific error types. |
| `backends/mongodb.py` | MongoDB implementation: views, indexes, vector search, change streams, connection pool tuning. |
| `backends/metadata.py` | MetadataStore: partition/operation/lock CRUD on `svr_metadata` collection. |
| `embedders/*.py` | Embedding providers with consistent interface. |
| `routing/resolver.py` | Resolves partition specs to actual PartitionInfo objects. Async, dual-source (metadata or config fallback). |
| `routing/merger.py` | Score normalization, deduplication, result merging. |
| `rerankers/*.py` | Cross-encoder reranking via external APIs. |
| `lifecycle/scanner.py` | Discovers partition values in the source collection. |
| `lifecycle/provisioner.py` | Creates and manages partitions (with rollback on failure). |
| `lifecycle/watcher.py` | Change stream monitoring with auto-reconnect and state persistence. |
| `lifecycle/monitor.py` | Health and threshold monitoring. |
| `lifecycle/splitter.py` | Auto-split for overgrown partitions (secondary field, hash, time strategies). |
| `lifecycle/detector.py` | Detection pipeline with 5 signals, distributed lock-based coordination. |
| `lifecycle/repartition.py` | Repartition engine: 5-step workflow with resume and rollback. |
| `utils/retry.py` | `@with_retry` decorator: exponential backoff, jitter, retryable exception filtering. |
| `utils/logging.py` | `SVRLogFormatter` (JSON), `correlation_id_var` (ContextVar), `log_operation` decorator. |
| `utils/metrics.py` | `MetricsHandler` protocol, `MetricsCollector`, 12 `MetricType` values. |
| `utils/cache.py` | `EmbeddingCache` (LRU + TTL), `CacheKey`, thread-safe with statistics. |
| `utils/field_analyzer.py` | Analyzes collection fields for filter suitability (cardinality, coverage). |
| `cli/__init__.py` | CLI entry point, registers 11 command groups. |
| `cli/helpers.py` | Shared CLI utilities: `_run_async()`, `_get_backend()`, `handle_config_error`. |
| `cli/init.py` | Interactive setup wizard using Rich for terminal UI; also supports non-interactive mode. |

---

## Core Components Deep Dive

### 1. SVRClient (`client.py`)

The main user-facing class that orchestrates all operations.

```python
class SVRClient:
    def __init__(self, config_path=None, config=None, auto_connect=True):
        # Load config from file, dict, or SVRConfig object
        # Initialize components lazily

    async def connect(self):
        # Connect to MongoDB
        # Initialize embedder (if BYOM mode)
        # Initialize reranker (if enabled)
        # Initialize resolver and merger

    async def search(self, query, partitions=None, limit=10, ...):
        # 1. Resolve partitions (names -> PartitionInfo objects)
        # 2. Embed query (if BYOM mode)
        # 3. Fan out searches to all partitions in parallel
        # 4. Merge results with score normalization
        # 5. Optionally rerank
        # 6. Return SearchResult
```

**Key Design Decisions:**

- **Lazy initialization**: Components like scanner, provisioner, watcher are only created when first used
- **Context manager support**: Can use `async with SVRClient() as svr:` for automatic cleanup
- **Auto-connect option**: For convenience, but can be disabled for more control

### 2. MongoDB Backend (`backends/mongodb.py`)

Handles all MongoDB operations using PyMongo's native async support.

```python
class MongoDBBackend(BaseBackend):
    # Connection management
    async def connect(self) -> None
    async def disconnect(self) -> None
    async def is_connected(self) -> bool

    # View management
    async def create_partition_view(self, partition_name, filter_value, ...) -> str
    async def delete_partition_view(self, view_name) -> None

    # Index management
    async def create_vector_search_index(self, collection_name, index_name, ...) -> None
    async def delete_vector_search_index(self, collection_name, index_name) -> None

    # Search operations
    async def execute_search(self, partition, query, query_vector, ...) -> List[Dict]
    async def search_partitions(self, partitions, ...) -> List[Dict]  # Parallel

    # Collection operations
    async def get_distinct_values(self, field) -> List[Any]
    async def count_documents(self, collection_name, filter) -> int

    # Change streams
    async def watch_collection(self, pipeline) -> AsyncIterator[Dict]
```

**Key Implementation Details:**

1. **View Pipeline Construction** (`_build_partition_view_pipeline`):
   ```python
   pipeline = [
       {"$match": {partition_field: filter_value}},  # Filter for partition
       {"$addFields": {computed_field: concat_expr}}, # Optional: concatenate fields
       # Optional: $lookup for separate embeddings collection
   ]
   ```

2. **Vector Search Pipeline** (`_build_vector_search_pipeline`):
   ```python
   pipeline = [
       {"$vectorSearch": {
           "index": index_name,
           "path": "embedding",
           "queryVector": [...] or "queryString": "...",
           "numCandidates": 100,
           "limit": 20,
           "filter": {...}  # Optional additional filters
       }},
       {"$addFields": {
           "_svr_partition": partition_name,
           "_svr_score": {"$meta": "vectorSearchScore"}
       }}
   ]
   ```

3. **Parallel Search Execution**:
   ```python
   async def search_partitions(self, partitions, ...):
       tasks = [self.execute_search(p, ...) for p in partitions]
       results = await asyncio.gather(*tasks, return_exceptions=True)
       # Handle errors, combine results
       return combined_results
   ```

4. **Connection Pool Tuning**:
   ```python
   AsyncMongoClient(
       connection_string,
       maxPoolSize=config.database.max_pool_size,        # default: 100
       minPoolSize=config.database.min_pool_size,        # default: 0
       maxIdleTimeMS=config.database.max_idle_time_ms,   # default: 0 (unlimited)
       waitQueueTimeoutMS=config.database.wait_queue_timeout_ms,  # default: 0
   )
   ```

### 3. Embedders (`embedders/`)

Abstract base class defines the interface:

```python
class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> List[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...
```

**Provider Implementations:**

| Provider | Models | Notes |
|----------|--------|-------|
| OpenAI | text-embedding-3-small, text-embedding-3-large | Supports dimension reduction |
| Voyage AI | voyage-4-large, voyage-4, voyage-4-lite, voyage-4-nano | **Voyage 4: Shared embedding space, asymmetric embeddings** |
| Voyage AI (legacy) | voyage-3-large, voyage-code-3 | input_type: query vs document |
| Cohere | embed-english-v3.0 | input_type: search_query vs search_document |
| HuggingFace | Any sentence-transformers model | Runs locally, async via thread pool |

#### Voyage 4 Shared Embedding Space

Voyage 4 introduces an industry-first capability: **all four models produce compatible embeddings**. This enables:

**Asymmetric Embeddings** - Use different models for queries vs documents:
```python
# Index documents once with the most accurate model
doc_embedder = VoyageEmbedder(
    model="voyage-4-large",
    api_key=key,
    input_type="document",
)

# Query with the fastest model - no re-indexing needed!
query_embedder = VoyageEmbedder(
    model="voyage-4-lite",
    api_key=key,
    input_type="query",
)
```

**Benefits:**
- Optimize document embeddings for accuracy (one-time or infrequent cost)
- Optimize query embeddings for latency (real-time)
- Change query model without re-indexing documents
- voyage-4-large uses MoE architecture with 40% lower serving costs

**Model Comparison:**
| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| voyage-4-large | Slowest | Highest | Document indexing |
| voyage-4 | Medium | High | Balanced workloads |
| voyage-4-lite | Fast | Good | Real-time queries |
| voyage-4-nano | Fastest | Good | Local dev, prototyping (open-weight) |

**Dimension Options (Matryoshka Learning):**
- 2048, 1024 (default), 512, 256
- Quantization: float, int8, uint8, binary, ubinary

**Why separate embedders?**
- Each provider has different API formats, authentication, and capabilities
- Abstract interface allows adding new providers without changing client code
- Factory pattern (`create_embedder`) selects implementation based on config

### 4. Routing (`routing/`)

#### PartitionResolver (`resolver.py`)

Converts partition specifications to actual `PartitionInfo` objects:

```python
class PartitionResolver:
    async def resolve(self, partitions) -> List[PartitionInfo]:
        # Handle: None -> default, "all" -> all active, ["a", "b"] -> specific
        # Expand split partitions to their children
        # Respect max_partitions_per_query limit
        # Dual-source: metadata store first, config file fallback
```

**Split Partition Handling:**

When a partition has been split (status=SPLIT), queries to the parent automatically fan out to children:

```python
# User queries: ["electronics"]
# If electronics is split into electronics_laptops, electronics_phones:
# Resolver returns: [electronics_laptops, electronics_phones]
```

#### ResultMerger (`merger.py`)

Combines and normalizes results from multiple partitions:

```python
class ResultMerger:
    def merge(self, results, limit) -> List[SearchHit]:
        # 1. Convert raw dicts to SearchHit objects
        # 2. Deduplicate by document ID (keep highest score)
        # 3. Normalize scores (within partition or global)
        # 4. Sort by score descending
        # 5. Limit results
```

**Score Normalization Methods:**

1. **partition_minmax** (default): Normalize within each partition to 0-1. Prevents one partition from dominating.
2. **global_minmax**: Normalize across all results. Better when partitions have similar score distributions.
3. **none**: No normalization. Use raw scores.

### 5. Rerankers (`rerankers/`)

Cross-encoder models that re-score query-document pairs:

```python
class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: List[str], top_k: int) -> List[float]: ...

    async def rerank_hits(self, query, hits, text_field, top_k) -> List[SearchHit]:
        # Extract text from documents
        # Call rerank API
        # Apply scores to hits
        # Sort by rerank score
```

**When is reranking used?**
- Multi-partition queries (scores from different partitions aren't directly comparable)
- When enabled in config and `rerank=True` (or not explicitly `False`)
- When reranker is configured with valid API key

### 6. Lifecycle Management (`lifecycle/`)

#### Scanner (`scanner.py`)

Discovers partition values in the source collection:

```python
class PartitionScanner:
    async def scan_partition_values(self, limit=None) -> Dict[str, int]:
        # Returns: {"electronics": 150000, "furniture": 85000, ...}

    async def get_new_partition_values(self) -> List[Any]:
        # Returns values that exist in data but not in registry
```

#### Provisioner (`provisioner.py`)

Creates and manages partitions. **Supports three index location modes:**

##### Index Location Modes

| Mode | Config | What gets created | Search behavior |
|------|--------|-------------------|-----------------|
| **VIEWS** | `index_on: "views"` | One view + one index per partition | Searches execute on partition views |
| **SOURCE** | `index_on: "source"` | One shared index on source collection + optional views | Searches execute on source with pre-filtering |
| **FIELDS** | `index_on: "fields"` | Per-partition embedding field + per-partition index on source collection | Searches execute on source, each with its own HNSW graph |

**VIEWS Mode** (default):
```
[Source Collection]
        |
        +-- [View: electronics] --> [Index: svr_vector_idx_electronics]
        +-- [View: furniture]   --> [Index: svr_vector_idx_furniture]
        +-- [View: clothing]    --> [Index: svr_vector_idx_clothing]
```
- **Pros**: Complete isolation between partitions, each index is optimized for its subset
- **Cons**: More indexes to manage, index build time per partition
- **Best for**: Moderate number of partitions (<50), when isolation is important

**SOURCE Mode**:
```
[Source Collection] --> [Index: svr_vector_idx_source]
        |                    (with partition field as filter)
        |
        +-- [View: electronics] (optional, for browsing)
        +-- [View: furniture]   (optional, for browsing)
        +-- [View: clothing]    (optional, for browsing)
```
- **Pros**: Single index to manage, faster partition creation, works well with many partitions
- **Cons**: All partitions share one ANN graph, pre-filtering overhead
- **Best for**: Many partitions (>50), simpler operational overhead

**FIELDS Mode**:
```
[Source Collection]
  field: embedding_electronics --> [Index: svr_vector_idx_electronics]
  field: embedding_furniture   --> [Index: svr_vector_idx_furniture]
  field: embedding_clothing    --> [Index: svr_vector_idx_clothing]
```
- **Pros**: Dedicated HNSW graph per partition, no views needed, documents only have vectors in their partition's field
- **Cons**: Limited to 50 partitions (Atlas 64-index limit minus headroom), requires per-partition embedding fields
- **Best for**: When you need HNSW isolation without the overhead of views

```python
class PartitionProvisioner:
    async def ensure_source_index(self) -> str:
        # For SOURCE mode: Create single index on source collection
        # with partition field as filter field

    async def create_partition(self, name, filter_value) -> PartitionInfo:
        # VIEWS mode:
        #   1. Create MongoDB view with filter
        #   2. Create vector search index on view
        #
        # SOURCE mode:
        #   1. Ensure source index exists (once)
        #   2. Optionally create view for browsing
        #   3. Partition searches use pre-filter
        #
        # FIELDS mode:
        #   1. Create partition-specific embedding field
        #   2. Create dedicated vector search index on source for that field
        #
        # All modes:
        #   3. Get document count
        #   4. Register in config with index_location setting
        #   5. Save config
        # On failure: rollback (delete view/index created so far)

    async def delete_partition(self, name) -> None:
        # Delete view (if exists), remove from registry
        # In VIEWS mode: also delete the partition-specific index
```

#### Watcher (`watcher.py`)

Monitors for new partition values via change streams:

```python
class PartitionWatcher:
    async def start(self) -> None:
        # Watch source collection for inserts/updates
        # Detect new values for partition field
        # Auto-provision or queue for confirmation
        # Auto-reconnect on connection loss (configurable retries)

    async def stop(self) -> None:
        # Stop watching
```

**Change Stream Pipeline:**
```python
pipeline = [
    {"$match": {
        "operationType": {"$in": ["insert", "update", "replace"]},
        f"fullDocument.{partition_field}": {"$exists": True}
    }}
]
```

#### Monitor (`monitor.py`)

Tracks partition health and thresholds:

```python
class PartitionMonitor:
    async def check_partition_health(self, name) -> PartitionHealthStatus:
        # Get document count
        # Compare to threshold
        # Return: healthy, warning (>80%), or critical (>100%)

    async def get_partition_summary(self) -> Dict:
        # Overall health across all partitions
```

#### Splitter (`splitter.py`)

Handles splitting overgrown partitions:

```python
class PartitionSplitter:
    async def execute_split(self, partition_name) -> List[str]:
        # Based on strategy:
        # - secondary_field: Split by another field's values
        # - hash: Split into N shards by ID hash
        # - time: Split by time buckets (yearly, monthly, weekly)
        # Returns list of child partition names
```

---

## Data Flow

### Search Flow (BYOM Mode)

```
User: svr.search("wireless headphones", partitions=["electronics", "furniture"])

1. RESOLVE PARTITIONS
   PartitionResolver.resolve(["electronics", "furniture"])
   -> [PartitionInfo(electronics), PartitionInfo(furniture)]

2. EMBED QUERY
   OpenAIEmbedder.embed("wireless headphones")
   -> [0.12, -0.34, 0.56, ...] (1536 floats)

3. FAN OUT SEARCH (parallel)
   +-- Backend.execute_search(electronics, query_vector) -> [doc1, doc2, ...]
   +-- Backend.execute_search(furniture, query_vector)   -> [doc3, doc4, ...]

4. MERGE RESULTS
   ResultMerger.merge([doc1, doc2, doc3, doc4, ...])
   -> Deduplicate, normalize scores, sort
   -> [SearchHit, SearchHit, ...]

5. RERANK (if enabled and multi-partition)
   VoyageReranker.rerank_hits(query, hits)
   -> Reorder by cross-encoder scores

6. RETURN
   SearchResult(hits=[...], partitions_searched=["electronics", "furniture"], ...)
```

### Partition Creation Flow

```
CLI: svr init

1. SCAN COLLECTION
   Scanner.scan_partition_values()
   -> {"electronics": 150000, "furniture": 85000, ...}

2. USER CONFIRMS
   (Interactive prompt to proceed)

3. CREATE PARTITIONS (for each value)
   Provisioner.create_partition("electronics", filter_value="electronics")
   |
   +-- Backend.create_partition_view("electronics", filter)
   |  -> CREATE VIEW svr_partition_electronics AS
   |    SELECT * FROM products WHERE category = "electronics"
   |
   +-- Backend.create_vector_search_index(view, index_name, ...)
   |  -> CREATE SEARCH INDEX ON svr_partition_electronics
   |
   +-- Backend.count_documents(view)
   |  -> 150000
   |
   +-- Config.partitions.registry["electronics"] = PartitionInfo(...)

4. SAVE CONFIG
   save_config(config, ".svr/config.json")
```

---

## Design Decisions & Rationale

### 1. Views Over Collections

**Decision:** Use MongoDB views with filters, not separate collections.

**Rationale:**
- No data duplication (views are logical, not physical)
- Source data stays in one place, easier to manage
- Updates to source automatically reflected in views
- MongoDB 8+ supports vector search indexes on views

**Tradeoff:**
- Views have some performance overhead for complex pipelines
- Can't shard views independently (but fine for most use cases)

### 2. PyMongo Async Over Motor

**Decision:** Use PyMongo's native async support (4.5+) instead of Motor.

**Rationale:**
- Motor is effectively deprecated (maintainer recommends PyMongo async)
- PyMongo async is the official path forward
- Better maintained, fewer bugs, more features
- Same API patterns, easy migration

### 3. BYOM as Primary Mode

**Decision:** "Bring Your Own Model" is the default, auto-embedding is optional.

**Rationale:**
- Auto-embedding is still in Public Preview on MongoDB Atlas
- Many users want control over their embedding model
- BYOM works with any MongoDB deployment (including Community Edition)
- Auto-embedding is trivial to enable when available

### 4. Explicit Partitions First

**Decision:** V1 requires explicit partition selection (`partitions=["electronics"]`).

**Rationale:**
- Simple, predictable behavior
- No magic inference that might be wrong
- Clear to users which data is being searched
- Inference-based routing can be added in V2

**Future:** Could add LLM-based partition inference:
```python
# V2 future concept
results = await svr.search("laptop for programming", infer_partitions=True)
# LLM determines: ["electronics", "computers"]
```

### 5. Config-Driven Architecture

**Decision:** All behavior controlled by `.svr/config.json`, not code.

**Rationale:**
- Change partitioning strategy without code changes
- Environment-specific configs (dev, staging, prod)
- Version control for configuration
- Interactive CLI generates correct config

### 6. Abstract Interfaces for Extensibility

**Decision:** Use abstract base classes for Backend, Embedder, Reranker.

**Rationale:**
- Easy to add new providers without changing core code
- Consistent interface for testing (mocks)
- Clear contracts for each component
- Factory pattern for instantiation

### 7. Score Normalization by Partition

**Decision:** Default normalization is within each partition (partition_minmax).

**Rationale:**
- Different partitions may have different score distributions
- Prevents one "noisy" partition from dominating results
- Cross-encoder reranking provides final unified ranking

### 8. Lazy Component Initialization

**Decision:** Lifecycle components (scanner, provisioner, etc.) are created on first use.

**Rationale:**
- Faster client initialization for simple search use cases
- No wasted resources for components that aren't used
- Clean separation of concerns

### 9. Async Context Manager Support

**Decision:** Support `async with SVRClient() as svr:` pattern.

**Rationale:**
- Ensures proper cleanup (disconnect) even on errors
- Familiar Python pattern
- No resource leaks

### 10. Rich CLI with Interactive Prompts

**Decision:** Use Rich library for terminal UI, interactive prompts.

**Rationale:**
- Better user experience than raw input()
- Progress spinners, colored output, tables
- Familiar from tools like npm, yarn, etc.

---

## Configuration System

### Configuration File Structure

```json
{
  "version": "1.0",

  "database": {
    "backend": "mongodb",
    "connection_string_env": "MONGODB_URI",
    "database": "my_database",
    "source_collection": "products",
    "max_pool_size": 100,
    "min_pool_size": 0,
    "max_idle_time_ms": 0,
    "wait_queue_timeout_ms": 0
  },

  "partitioning": {
    "field": "category",
    "strategy": "exact_match",
    "view_prefix": "svr_partition_",
    "index_name_prefix": "svr_vector_idx_"
  },

  "vector_storage": {
    "mode": "embedded",
    "index_on": "views"
  },

  "vector_search": {
    "embedding_field": "embedding",
    "dimensions": 1536,
    "similarity": "cosine"
  },

  "embedding": {
    "mode": "byom",
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key_env": "OPENAI_API_KEY"
  },

  "routing": {
    "mode": "explicit",
    "default_partitions": "all",
    "max_partitions_per_query": 5
  },

  "reranking": {
    "enabled": true,
    "provider": "voyage",
    "model": "rerank-2",
    "api_key_env": "VOYAGE_API_KEY"
  },

  "resilience": {
    "max_retry_attempts": 3,
    "retry_base_delay": 0.5,
    "retry_max_delay": 30.0,
    "connection_timeout_ms": 10000,
    "server_selection_timeout_ms": 30000,
    "search_timeout_ms": 30000,
    "embedding_timeout_ms": 60000,
    "reranking_timeout_ms": 60000,
    "health_check_interval_s": 30,
    "watcher_max_retries": 10,
    "watcher_base_delay": 1.0,
    "watcher_max_delay": 60.0
  },

  "lifecycle": {
    "auto_provision": true,
    "change_stream_enabled": true,
    "metadata": {
      "connection_string_env": null,
      "database": null,
      "collection": "svr_metadata"
    },
    "detection": {
      "enabled": true,
      "interval": "1h",
      "threshold_vectors": 10000000,
      "min_threshold_vectors": 1000,
      "skew_ratio": 5.0,
      "trend_window_days": 30
    },
    "repartition": {
      "index_wait_timeout_s": 1800,
      "index_poll_interval_s": 10,
      "auto_cleanup_retired": true
    }
  },

  "logging": {
    "level": "INFO",
    "json_format": false,
    "log_query_text": false,
    "log_embeddings": false
  },

  "metrics": {
    "enabled": true,
    "include_partition_tags": true,
    "include_query_tags": false
  },

  "cache": {
    "enabled": true,
    "max_size": 10000,
    "ttl_seconds": 3600
  },

  "partitions": {
    "registry": {
      "electronics": {
        "name": "electronics",
        "view_name": "svr_partition_electronics",
        "index_name": "svr_vector_idx_electronics",
        "filter_value": "electronics",
        "document_count": 150000,
        "status": "active",
        "search_collection": "svr_partition_electronics",
        "index_location": "views"
      }
    }
  }
}
```

### Configuration Loading Priority

1. `config` parameter (dict or SVRConfig object) - highest priority
2. `config_path` parameter (explicit path)
3. Auto-discovery in order:
   - `.svr/config.json`
   - `svr.config.json`
   - `.svr.json`

### Environment Variable Resolution

API keys and connection strings use environment variable names in config:

```json
{
  "embedding": {
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

At runtime, `os.environ.get("OPENAI_API_KEY")` is called. This:
- Keeps secrets out of config files
- Allows different values per environment
- Follows 12-factor app principles

---

## Embedding System

### Embedding Modes

#### 1. BYOM (Bring Your Own Model)

```
Query: "wireless headphones"
        |
        v
   +--------------+
   |  SVRClient   |
   +------+-------+
          | embed(query)
          v
   +--------------+
   |  Embedder    |---- API Call -----> OpenAI/Voyage/Cohere
   |  (Provider)  |<-- [0.1, 0.2, ...] --+
   +------+-------+
          | query_vector
          v
   +--------------+
   |  MongoDB     |  $vectorSearch: { queryVector: [...] }
   |              |
   +--------------+
```

**Config:**
```json
{
  "embedding": {
    "mode": "byom",
    "provider": "openai",
    "model": "text-embedding-3-small"
  }
}
```

#### 2. Auto-embedding (MongoDB Atlas)

```
Query: "wireless headphones"
        |
        v
   +--------------+
   |  SVRClient   |  (no embedding needed)
   +------+-------+
          | raw query string
          v
   +--------------+
   |  MongoDB     |  $vectorSearch: { queryString: "..." }
   |  (Atlas)     |  --- internally embeds ---> Voyage AI
   +--------------+
```

**Config:**
```json
{
  "embedding": {
    "mode": "auto",
    "provider": "atlas_voyage"
  }
}
```

---

### Voyage 4: Shared Embedding Space

Voyage 4 is a major advancement in embedding technology, introducing **industry-first shared embedding spaces**. All Voyage 4 models produce compatible embeddings, enabling powerful optimization strategies.

#### Model Family

| Model | Parameters | Speed | Accuracy | Best Use Case |
|-------|------------|-------|----------|---------------|
| **voyage-4-large** | Largest (MoE) | Slowest | Highest | Document indexing |
| **voyage-4** | Medium | Balanced | High | General purpose |
| **voyage-4-lite** | Smaller | Fast | Good | Real-time queries |
| **voyage-4-nano** | Smallest | Fastest | Good | Local dev (open-weight, Apache 2.0) |

#### Asymmetric Embeddings

The killer feature: **use different models for queries vs documents** without re-indexing:

```
Document Indexing (one-time):            Query Time (real-time):
        |                                        |
        v                                        v
   voyage-4-large                           voyage-4-lite
   (highest accuracy)                       (lowest latency)
        |                                        |
        v                                        v
   +--------------+                         +--------------+
   | [0.1, 0.2,   |    COMPATIBLE!          | [0.1, 0.2,   |
   |  ...]        |<----------------------> |  ...]        |
   +--------------+    Same embedding       +--------------+
                         space
```

**Benefits:**
- **Cost optimization**: Expensive model for documents (run once), cheap model for queries (run often)
- **Latency optimization**: Fast query embedding without sacrificing document quality
- **No re-indexing**: Change query model anytime without touching your vector index
- **40% cost reduction**: voyage-4-large uses MoE architecture with lower inference costs

#### Configuration Examples

**Standard (same model for everything):**
```json
{
  "embedding": {
    "provider": "voyage",
    "model": "voyage-4",
    "dimensions": 1024
  }
}
```

**Asymmetric (recommended for production):**
```json
{
  "embedding": {
    "provider": "voyage",
    "model": "voyage-4-lite",
    "document_model": "voyage-4-large",
    "voyage_output_dimension": 1024,
    "voyage_quantization": "float"
  }
}
```

#### Dimension Options (Matryoshka Learning)

Voyage 4 supports flexible output dimensions without retraining:

| Dimensions | Storage | Accuracy | Use Case |
|------------|---------|----------|----------|
| 2048 | Largest | Highest | Maximum precision |
| 1024 | Default | High | General purpose |
| 512 | Medium | Good | Balanced |
| 256 | Smallest | Acceptable | Cost-sensitive |

#### Voyage API Quantization Options

Voyage can return vectors in different formats:

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| `float` | 4 bytes/dim | Best | Default |
| `int8` | 1 byte/dim | Very good | Production |
| `uint8` | 1 byte/dim | Very good | Production |
| `binary` | 1 bit/dim | Good | Large scale |
| `ubinary` | 1 bit/dim | Good | Large scale |

---

### MongoDB Quantization & Vector Storage

MongoDB Atlas Vector Search provides **two independent mechanisms** for optimizing vector storage and retrieval. Understanding when to use each is critical for optimal performance.

#### The Two Approaches

| Approach | Where | When | Best For |
|----------|-------|------|----------|
| **Automatic Quantization** | MongoDB index layer | At index time | Float vectors from any provider |
| **Pre-quantized Ingestion** | Application layer | At storage time | Pre-quantized vectors from Voyage |

#### 1. Vector Storage Format (Application Layer)

Controls how vectors are stored as BSON documents:

```python
class VectorStorageFormat(str, Enum):
    """How vectors are stored in MongoDB."""
    ARRAY = "array"              # Standard array<double> - 8 bytes per dimension
    BINDATA_FLOAT32 = "bindata_float32"  # BinData float32 - 4 bytes per dimension
    BINDATA_INT8 = "bindata_int8"        # BinData int8 - 1 byte per dimension
    BINDATA_PACKED_BIT = "bindata_packed_bit"  # BinData packed_bit - 1 bit per dimension
```

**Storage comparison (1024-dimension vector):**

| Format | Storage Size | Source |
|--------|-------------|--------|
| `array` | 8 KB | Native float64 |
| `bindata_float32` | 4 KB | Float32 packed |
| `bindata_int8` | 1 KB | Pre-quantized from Voyage |
| `bindata_packed_bit` | 128 bytes | Binary from Voyage |

#### 2. Index Quantization (MongoDB Index Layer)

Controls **automatic** quantization applied at the index level:

```python
class MongoDBIndexQuantization(str, Enum):
    """MongoDB Atlas Vector Search index quantization mode."""
    NONE = "none"    # No automatic quantization
    SCALAR = "scalar"  # Auto-quantize to int8 (~75% RAM reduction)
    BINARY = "binary"  # Auto-quantize to 1-bit (~97% RAM reduction)
```

Index definition with quantization:
```json
{
  "type": "vectorSearch",
  "fields": [{
    "path": "embedding",
    "type": "vector",
    "numDimensions": 1024,
    "similarity": "cosine",
    "quantization": "scalar"
  }]
}
```

#### Compatibility Rules

**CRITICAL**: Storage format and index quantization must be compatible:

| Storage Format | Compatible Index Quantization | Notes |
|---------------|------------------------------|-------|
| `array` | none, scalar, binary | Full flexibility |
| `bindata_float32` | none, scalar, binary | Recommended default |
| `bindata_int8` | **none ONLY** | Already quantized! |
| `bindata_packed_bit` | **none ONLY** | Already quantized! |

If you store pre-quantized vectors (int8/packed_bit), you MUST use `quantization: none` in the index definition. Double-quantization produces garbage results.

#### Recommended Configurations

**1. Starting fresh, cost-conscious (RECOMMENDED):**
```json
{
  "vector_storage": {
    "storage_format": "bindata_float32",
    "index_quantization": "scalar"
  }
}
```
- 4 bytes/dim storage (3x savings vs array)
- Index RAM reduced by 75%
- Great balance of cost and quality

**2. Maximum compression:**
```json
{
  "vector_storage": {
    "storage_format": "bindata_float32",
    "index_quantization": "binary"
  }
}
```
- 97% index RAM reduction
- Some quality loss - test for your use case

**3. Using Voyage 4 pre-quantized (int8):**
```json
{
  "embedding": {
    "provider": "voyage",
    "model": "voyage-4-large",
    "voyage_quantization": "int8"
  },
  "vector_storage": {
    "storage_format": "bindata_int8",
    "index_quantization": "none"
  }
}
```
- Voyage handles quantization with better quality
- 1 byte/dim storage
- No double-quantization

**4. Legacy/simple (no optimization):**
```json
{
  "vector_storage": {
    "storage_format": "array",
    "index_quantization": "none"
  }
}
```
- Highest storage cost (8 bytes/dim)
- Compatible with all tools

#### Decision Flowchart

```
Do you need maximum quality?
    |
    Yes --> Use `storage_format: bindata_float32`, `index_quantization: none`
    |
    No --> Are you using Voyage 4?
              |
              Yes --> Use Voyage int8/binary output
              |       with `storage_format: bindata_int8/packed_bit`
              |       and `index_quantization: none`
              |
              No --> Use `storage_format: bindata_float32`
                     with `index_quantization: scalar` (or binary for max compression)
```

#### Python API

```python
from semantic_vector_router.embedders.voyage import VoyageEmbedder

# Create asymmetric embedder pair
query_embedder, doc_embedder = VoyageEmbedder.create_asymmetric_pair(
    api_key=api_key,
    query_model="voyage-4-lite",      # Fast
    document_model="voyage-4-large",  # Accurate
    output_dimension=1024,
)

# Or configure individually
embedder = VoyageEmbedder(
    model="voyage-4",
    api_key=api_key,
    input_type="query",
    output_dimension=512,  # Reduced dimensions
    output_dtype="int8",   # Quantized
)

# Switch models easily
doc_embedder = embedder.for_documents(model="voyage-4-large")
query_embedder = embedder.for_queries(model="voyage-4-lite")
```

### Multi-Field Embedding

When documents have multiple relevant text fields, they can be concatenated:

**Config:**
```json
{
  "embedding": {
    "source_fields": ["title", "description", "specs"],
    "separator": " | ",
    "computed_field": "embedding_text"
  }
}
```

**View Pipeline:**
```javascript
{
  "$addFields": {
    "embedding_text": {
      "$concat": [
        {"$ifNull": ["$title", ""]},
        " | ",
        {"$ifNull": ["$description", ""]},
        " | ",
        {"$ifNull": ["$specs", ""]}
      ]
    }
  }
}
```

---

## Routing & Merging

### Partition Resolution Process

```
Input: partitions=["electronics", "furniture"]

1. Lookup in registry:
   electronics -> PartitionInfo(name="electronics", status=ACTIVE, ...)
   furniture -> PartitionInfo(name="furniture", status=ACTIVE, ...)

2. Handle split partitions:
   If electronics.status == SPLIT:
     Replace with electronics.child_partitions
     -> ["electronics_laptops", "electronics_phones"]

3. Apply max limit:
   If len(partitions) > config.routing.max_partitions_per_query:
     partitions = partitions[:max_limit]

4. Return: [PartitionInfo, PartitionInfo, ...]
```

### Result Merging Process

```
Input: Raw results from multiple partitions

1. Convert to SearchHit objects:
   {_id: "doc1", name: "...", _svr_score: 0.95, _svr_partition: "electronics"}
   -> SearchHit(id="doc1", score=0.95, partition="electronics", document={...})

2. Deduplicate (if enabled):
   Same document in multiple partitions -> keep highest score

3. Normalize scores:
   partition_minmax: Within each partition, scale to 0-1

   Electronics: [0.95, 0.85] -> [1.0, 0.0]  (normalized)
   Furniture:   [0.72, 0.70] -> [1.0, 0.0]  (normalized)

4. Sort by score (descending)

5. Limit to requested count
```

### Reranking Process

```
Input: Merged SearchHits, Query

1. Extract text from each document:
   hit.document.get("text") or hit.document.get("description") or ...

2. Call reranker API:
   POST /rerank {query: "...", documents: ["text1", "text2", ...]}
   Response: [{index: 0, relevance_score: 0.95}, ...]

3. Apply rerank scores:
   hit.rerank_score = score

4. Sort by rerank_score (descending)

5. Return top_k
```

---

## Lifecycle Management

### Auto-Provisioning Flow

```
Document inserted: {category: "automotive", name: "Car Parts", ...}
                          |
                          v
+------------------------------------------+
|            Change Stream                 |
|  Watching for: category field changes    |
+------------------+-----------------------+
                   |
                   v
        Is "automotive" known?
               |
        No ----+---- Yes (ignore)
        |
        v
  auto_provision: true?
        |
  Yes --+-- No (add to pending list)
  |
  v
  confirmation_required: true?
        |
  Yes --+-- No (create immediately)
  |         |
  v         v
(pending)  Provisioner.create_partition("automotive")
                      |
                      +-- Create view
                      +-- Create index
                      +-- Update config
```

### Auto-Split Process

```
Periodic check: Monitor.check_all_partitions()
                          |
                          v
        electronics: 12M vectors (threshold: 10M)
                          |
                          v
               STATUS: CRITICAL
                          |
                          v
        Is current time within schedule?
        (e.g., weekends 2am-6am)
               |
        Yes ---+--- No (mark as PENDING_SPLIT)
        |
        v
  Execute split based on strategy:

  secondary_field:
    electronics -> electronics__laptops
                   electronics__phones
                   electronics__accessories

  hash:
    electronics -> electronics__shard_0
                   electronics__shard_1
                   electronics__shard_2
                   electronics__shard_3

  time:
    electronics -> electronics__2024
                   electronics__2025
```

---

## Error Handling

### Exception Hierarchy

```
SVRException (base)
+-- ConfigurationError         # Invalid/missing config
+-- ConnectionError            # Database connection failed
+-- PartitionNotFoundError     # Requested partition doesn't exist
+-- PartitionAlreadyExistsError
+-- PartitionProvisioningError
+-- SearchError                # Vector search failed
+-- EmbeddingError             # Embedding generation failed
+-- RerankingError             # Reranking API failed
+-- IndexCreationError
+-- ViewCreationError
+-- ChangeStreamError
+-- SplitError
+-- MonitoringError
+-- ValidationError            # Input validation failed
+-- APIKeyError                # API key is missing or invalid
+-- ScanError                  # Partition scan operation failed
+-- MetadataError              # Metadata store operation failed
+-- DetectionError             # Detection pipeline operation failed
+-- RepartitionError           # Repartition operation failed
```

All exceptions inherit from `SVRException`, which supports an optional `details` dict for structured error context:

```python
class SVRException(Exception):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
```

### Error Handling Patterns

**1. Graceful degradation in parallel searches:**
```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"Search failed on {partitions[i]}: {result}")
        continue  # Skip failed partition, use other results
    combined.extend(result)
```

**2. API key validation:**
```python
def get_api_key(env_var_name, provider):
    value = os.environ.get(env_var_name)
    if value is None:
        raise ConfigurationError(
            f"API key for {provider} not found in {env_var_name}"
        )
    return value
```

**3. Retry with exponential backoff (implemented in `utils/retry.py`):**
```python
@with_retry(max_attempts=3, base_delay=0.5, max_delay=30.0)
async def connect(self):
    # Retries on transient failures with backoff
    ...
```

**4. Provisioner rollback on failure:**
```python
async def create_partition(self, name, filter_value):
    created_view = None
    created_index = None
    try:
        created_view = await self.backend.create_partition_view(...)
        created_index = await self.backend.create_vector_search_index(...)
    except Exception:
        # Rollback: delete anything created
        if created_index:
            await self.backend.delete_vector_search_index(...)
        if created_view:
            await self.backend.delete_partition_view(...)
        raise
```

---

## Resilience & Retry

SVR provides comprehensive resilience through the `utils/retry.py` module and the `ResilienceConfig` model.

### `@with_retry` Decorator

The `with_retry` decorator provides exponential backoff with jitter for both async and sync functions:

```python
from semantic_vector_router.utils.retry import with_retry

@with_retry(
    max_attempts=3,         # Total attempts (0 = no retry, just run once)
    base_delay=0.5,         # Base delay in seconds
    max_delay=30.0,         # Maximum delay cap in seconds
    retryable_exceptions=(AutoReconnect, NetworkTimeout),  # Which errors to retry
)
async def my_operation():
    ...
```

**Key behaviors:**
- **Exponential backoff**: Delay doubles each attempt: 0.5s, 1s, 2s, 4s, ...
- **Jitter**: Random +/-25% to prevent thundering herd
- **HTTP 429 Retry-After**: Respects `Retry-After` header from rate-limited APIs
- **Retryable HTTP status codes**: Only retries on 429, 500, 502, 503, 504
- **Sync/async auto-detection**: Uses `inspect.iscoroutinefunction` to wrap appropriately

**Programmatic use:**
```python
from semantic_vector_router.utils.retry import async_retry

result = await async_retry(
    func=some_async_function,
    args=(arg1,),
    max_attempts=5,
)
```

### `ResilienceConfig`

Centralizes all timeout and retry parameters in the configuration:

```python
class ResilienceConfig(BaseModel):
    # Retry settings
    max_retry_attempts: int = 3
    retry_base_delay: float = 0.5       # seconds
    retry_max_delay: float = 30.0       # seconds

    # MongoDB timeouts
    connection_timeout_ms: int = 10_000
    server_selection_timeout_ms: int = 30_000
    search_timeout_ms: int = 30_000      # maxTimeMS on $vectorSearch

    # API timeouts
    embedding_timeout_ms: int = 60_000
    reranking_timeout_ms: int = 60_000

    # Health check
    health_check_interval_s: int = 30    # min seconds between ping checks

    # Watcher reconnection
    watcher_max_retries: int = 10
    watcher_base_delay: float = 1.0
    watcher_max_delay: float = 60.0
```

### Connection Health Check

The MongoDB backend tracks connection staleness and provides a health check mechanism:

- Minimum interval between pings (`health_check_interval_s`) prevents excessive checking
- Used by watcher reconnection to validate the connection before resuming

### Provisioner Rollback on Failure

When partition creation fails partway through (e.g., view created but index creation fails), the provisioner automatically rolls back any resources that were already created, preventing orphaned views or indexes.

### Watcher State Persistence and Auto-Reconnect

The `PartitionWatcher` persists its resume token so that on reconnection it can resume from where it left off, without missing events. Reconnection uses configurable retry parameters (`watcher_max_retries`, `watcher_base_delay`, `watcher_max_delay`).

---

## CLI Architecture

### Entry Point

The CLI is built with [Click](https://click.palletsprojects.com/) and uses [Rich](https://github.com/Textualize/rich) for terminal UI. The entry point is `cli/__init__.py`, which registers all 11 command groups:

```python
# cli/__init__.py
@click.group()
@click.version_option(version="0.1.0", prog_name="svr")
def main() -> None:
    """Semantic Vector Router - Automatic vector index partitioning and query routing."""
    pass

main.add_command(init_command, name="init")
main.add_command(partitions_group, name="partitions")
main.add_command(search_command, name="search")
main.add_command(analyze_command, name="analyze")
main.add_command(watch_group, name="watch")
main.add_command(split_group, name="split")
main.add_command(config_group, name="config")
main.add_command(index_group, name="index")
main.add_command(monitor_group, name="monitor")
main.add_command(repartition_group, name="repartition")
main.add_command(cache_group, name="cache")
```

### Shared Utilities (`cli/helpers.py`)

All CLI modules share three helper functions:

```python
def _run_async(coro):
    """Run async code from Click (sync) commands."""
    return asyncio.run(coro)

async def _get_backend(config_path=None):
    """Load config and connect to MongoDB. Caller must disconnect."""
    config = load_config(config_path=config_path)
    backend = MongoDBBackend(config)
    await backend.connect()
    return config, backend

def handle_config_error(func):
    """Decorator to handle ConfigurationError with a user-friendly message."""
    # Catches ConfigurationError and shows: "Run 'svr init' to create a configuration file."
```

### Command Groups and Subcommands

| Command Group | Subcommands | Description |
|---------------|-------------|-------------|
| `svr init` | (standalone) | Interactive or non-interactive setup wizard |
| `svr partitions` | `list`, `status`, `create`, `delete`, `refresh`, `scan`, `provision` | Partition management |
| `svr search` | (standalone) | Execute vector search from CLI |
| `svr analyze` | (standalone) | Analyze collection fields for filter suitability |
| `svr watch` | `start`, `status`, `confirm`, `reject`, `confirm-all` | Change stream monitoring |
| `svr split` | `check`, `execute` | Partition splitting |
| `svr config` | `show`, `validate`, `set`, `path` | Configuration management |
| `svr index` | `status`, `rebuild`, `wait` | Vector search index management |
| `svr monitor` | `check`, `history` | Detection pipeline (health checks, signal history) |
| `svr repartition` | `pending`, `execute`, `status`, `rollback` | Repartition workflow management |
| `svr cache` | `stats` | Embedding cache statistics |

### Non-Interactive Init Mode

The `svr init` command supports both interactive mode (with Rich prompts) and non-interactive mode for CI/CD pipelines:

```bash
svr init --non-interactive \
  --database my_db \
  --collection products \
  --field category \
  --embedding-provider voyage \
  --index-on source
```

---

## Metadata Store & Distributed State

### Overview

The `MetadataStore` (`backends/metadata.py`) manages persistent state in a MongoDB collection called `svr_metadata`. It replaces the config-file-based partition registry with a database-backed store suitable for multi-worker deployments.

### Document Types

The `svr_metadata` collection stores three document types, distinguished by `_id` prefix and `type` field:

| Document Type | `_id` Pattern | Purpose |
|---------------|---------------|---------|
| **partition** | `partition:<name>` | Partition state (replaces config registry entries) |
| **operation** | `op:<type>-<partition>-<timestamp>` | Repartition operations (pending/in-progress/done/failed) |
| **lock** | `lock:<id>` | Distributed locks for multi-worker coordination |

### Partition Documents

```json
{
  "_id": "partition:electronics",
  "type": "partition",
  "name": "electronics",
  "view_name": "svr_partition_electronics",
  "index_name": "svr_vector_idx_electronics",
  "filter_value": "electronics",
  "status": "active",
  "document_count": 150000,
  "health_history": [
    {"ts": "2026-02-10T00:00:00", "count": 148000},
    {"ts": "2026-02-11T00:00:00", "count": 149200},
    {"ts": "2026-02-12T00:00:00", "count": 150000}
  ]
}
```

The `health_history` array is capped at 30 data points via `$push` with `$slice: -30`.

### Operation Documents

```json
{
  "_id": "op:split-electronics-2026-02-12T14:30:00",
  "type": "operation",
  "action": "split",
  "target_partition": "electronics",
  "strategy": "secondary_field",
  "status": "pending",
  "steps": [
    {"step": 1, "action": "create_children", "status": "pending"},
    {"step": 2, "action": "build_indexes", "status": "pending"},
    {"step": 3, "action": "wait_indexes", "status": "pending"},
    {"step": 4, "action": "switch_routing", "status": "pending"},
    {"step": 5, "action": "cleanup_parent", "status": "pending"}
  ]
}
```

### Distributed Locking

Locks use MongoDB's `find_one_and_update` with `upsert` for atomic acquisition:

```python
async def acquire_lock(self, lock_id: str, holder: str, ttl_seconds: int = 300) -> bool:
    result = await self._collection.find_one_and_update(
        {
            "_id": f"lock:{lock_id}",
            "$or": [
                {"expires_at": {"$lt": now}},   # Expired lock
                {"holder": holder},              # Re-acquire own lock
            ],
        },
        {"$set": {"holder": holder, "expires_at": expires_at, ...}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return result is not None and result.get("holder") == holder
```

**Key properties:**
- **TTL-based**: Locks expire automatically after `ttl_seconds` (default 300s)
- **Re-entrant**: Same holder can re-acquire its own lock
- **Race-safe**: `DuplicateKeyError` on upsert race returns `False`

### Configurable Metadata Location

By default the metadata store shares the main backend's database. For multi-cluster deployments, a separate connection can be configured:

```json
{
  "lifecycle": {
    "metadata": {
      "connection_string_env": "METADATA_MONGODB_URI",
      "database": "svr_shared_state",
      "collection": "svr_metadata"
    }
  }
}
```

### Dual-Source Partition Resolution

The `PartitionResolver` is now fully async and checks two sources:
1. **MetadataStore** (primary): For deployments using the metadata collection
2. **Config file** (fallback): For backward compatibility or single-worker setups

### Migration from Config to Metadata

One-time idempotent migration copies partition entries from the config file into the metadata collection:

```python
async def migrate_from_config(self, config: SVRConfig) -> int:
    migrated = 0
    for name, partition in config.partitions.registry.items():
        existing = await self.get_partition(name)
        if existing is None:
            await self.save_partition(partition)
            migrated += 1
    return migrated
```

All MetadataStore methods use the `@with_retry` decorator with retryable MongoDB exceptions for resilience.

---

## Detection Pipeline

### Overview

The detection pipeline (`lifecycle/detector.py`) identifies partition health issues through a structured workflow. It runs periodically (via CLI cron, in-process timer, or manual invocation) and produces actionable signals.

### Signal Types

| Signal | Trigger | Auto-executable | Suggested Action |
|--------|---------|-----------------|------------------|
| `THRESHOLD_BREACH` | Partition count exceeds `threshold_vectors` | Yes | Split partition |
| `APPROACHING_THRESHOLD` | Trend analysis predicts breach within `trend_window_days` | No | Prepare split |
| `SEVERE_SKEW` | Max/avg ratio among siblings exceeds `skew_ratio` | No | Rebalance |
| `UNDERPOPULATED` | Leaf partition count below `min_threshold_vectors` | No | Merge |
| `STALE` | No growth in last 10 measurements | No | Archive |

### Workflow: COLLECT -> STORE -> ANALYZE -> DECIDE

```
1. COLLECT
   Count documents per partition via backend.count_documents()
       |
       v
2. STORE
   Append counts to health_history in metadata store
   (capped at 30 data points per partition)
       |
       v
3. ANALYZE
   Run all 5 detection checks:
   +-- _check_threshold_breaches()     -> THRESHOLD_BREACH signals
   +-- _check_approaching_thresholds() -> APPROACHING_THRESHOLD signals
   +-- _check_skew()                   -> SEVERE_SKEW signals
   +-- _check_underpopulated()         -> UNDERPOPULATED signals
   +-- _check_stale()                  -> STALE signals
       |
       v
4. DECIDE
   For auto-executable signals (currently THRESHOLD_BREACH only):
   Create operation documents in metadata store with status="pending"
   Non-auto signals are returned as suggestions for operator review
```

### Trend Analysis

The `APPROACHING_THRESHOLD` signal uses linear regression on health history:

```python
def _calculate_trend_slope(self, history: list[dict]) -> float:
    # Uses numpy if available, otherwise manual least squares
    # Returns slope in vectors/second
    # Converted to vectors/day for breach prediction
```

Requires at least 3 data points. The prediction window is configurable via `trend_window_days` (default: 30).

### Distributed Lock Coordination

For multi-worker deployments, detection runs are coordinated via the `run_detection_with_lock()` method:

```python
async def run_detection_with_lock(self) -> Optional[list[DetectionResult]]:
    holder = f"{hostname}-{pid}"
    acquired = await self.metadata.acquire_lock("monitor", holder)
    if not acquired:
        return None  # Another worker is running detection
    try:
        return await self.run_detection()
    finally:
        await self.metadata.release_lock("monitor", holder)
```

### Execution Models

| Model | How | Best For |
|-------|-----|----------|
| **CLI cron** | `svr monitor check` in a cron job | Simple deployments |
| **In-process** | Timer-based within the application | Always-on applications |
| **Manual** | `svr monitor check` on demand | Debugging, ad-hoc checks |

---

## Repartition Engine

### Overview

The `RepartitionEngine` (`lifecycle/repartition.py`) orchestrates multi-step repartitioning operations with resume capability and rollback support. It is designed for zero-downtime operation: the old partition stays active and searchable throughout the repartitioning process.

### 5-Step Workflow

```
Step 1: create_children
  Mark parent as SPLITTING
  Create child partitions using the specified strategy
  Update parent with child references

Step 2: build_indexes
  Verify all child indexes exist

Step 3: wait_indexes
  Poll until all child indexes are queryable
  Configurable timeout (default 1800s) and poll interval (default 10s)

Step 4: switch_routing
  Mark parent as RETIRED
  Mark all children as ACTIVE
  Queries now route to children

Step 5: cleanup_parent
  (Best-effort) Delete parent index and view
  Configurable via auto_cleanup_retired (default true)
```

### Zero-Downtime Guarantee

The old partition remains ACTIVE and searchable until step 4 (`switch_routing`). Only after all child indexes are verified queryable does the routing switch atomically from parent to children.

### Resume Capability

Each step's status is tracked in the operation document. On resume:

```python
for step in op.get("steps", []):
    if step["status"] == "done":
        continue  # Skip completed steps
    # Execute step...
```

If a repartition fails mid-way (e.g., network issue during index wait), it can be resumed later and will pick up from the last incomplete step.

### Rollback

Rollback reverses the repartitioning:

```python
async def rollback_operation(self, op_id: str) -> None:
    # 1. Delete child partitions (best-effort)
    # 2. Reset parent to ACTIVE status
    # 3. Clear parent's child_partitions list
    # 4. Mark operation as "failed" with "Rolled back by user"
```

### Schedule Support

The repartition configuration supports scheduling constraints:

```python
class RepartitionConfig(BaseModel):
    schedule: Optional[SplitScheduleConfig] = None  # allowed_days, allowed_hours
    index_wait_timeout_s: int = 1800
    index_poll_interval_s: int = 10
    auto_cleanup_retired: bool = True
```

The `SplitScheduleConfig` constrains when repartitioning can execute:

```python
class SplitScheduleConfig(BaseModel):
    allowed_days: list[str] = ["saturday", "sunday"]
    allowed_hours: dict[str, int] = {"start": 2, "end": 6}
    timezone: str = "UTC"
```

---

## Observability

Phase 6 introduced three pillars of observability: structured logging, metrics hooks, and embedding cache diagnostics.

### Structured Logging (`utils/logging.py`)

#### JSON Formatter

The `SVRLogFormatter` outputs structured JSON logs suitable for log aggregation systems (Datadog, ELK, Loki, etc.):

```python
class SVRLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        # Include extra fields set on the record
        # Include exception info if present
        return json.dumps(entry, default=str)
```

Example output:
```json
{"ts": "2026-02-12T14:30:00.123456+00:00", "level": "INFO",
 "logger": "semantic_vector_router.client", "msg": "search.complete",
 "correlation_id": "abc123def456", "duration_ms": 42.5}
```

Enable via configuration:
```json
{
  "logging": {
    "level": "INFO",
    "json_format": true,
    "log_query_text": false,
    "log_embeddings": false
  }
}
```

#### Correlation IDs

`ContextVar`-based correlation IDs propagate across `await` boundaries without threading issues:

```python
from semantic_vector_router.utils.logging import new_correlation_id, get_correlation_id

# Generate and set a new correlation ID for the current async context
cid = new_correlation_id()  # e.g., "abc123def456"

# All logs within this context automatically include the correlation_id
```

#### `log_operation` Decorator

Automatically logs operation start, completion, and errors with timing:

```python
from semantic_vector_router.utils.logging import log_operation, get_logger

logger = get_logger(__name__)

@log_operation(logger, "search", partitions=3)
async def execute_search(...):
    ...
# Emits:
#   search.start  {partitions: 3}
#   search.complete  {duration_ms: 42.5}
#   -- or on error --
#   search.error  {duration_ms: 12.1, error: "SearchError"}
```

### Metrics Hooks (`utils/metrics.py`)

#### MetricsHandler Protocol

Any observability backend can implement the `MetricsHandler` protocol:

```python
class MetricsHandler(Protocol):
    def handle(self, event: MetricEvent) -> None: ...
```

Example implementations:
```python
# Datadog
class DatadogHandler:
    def handle(self, event: MetricEvent) -> None:
        statsd.timing(event.metric_type.value, event.value, tags=event.tags)

# Prometheus
class PrometheusHandler:
    def handle(self, event: MetricEvent) -> None:
        histogram.labels(**event.tags).observe(event.value)
```

#### MetricsCollector

The `MetricsCollector` dispatches events to all registered handlers with error isolation:

```python
collector = MetricsCollector()
collector.add_handler(DatadogHandler())
collector.add_handler(PrometheusHandler())

# Emit metrics (fire-and-forget, never blocks hot path)
collector.emit_timing(MetricType.SEARCH_LATENCY, duration_ms=42.5, partition="electronics")
collector.emit_count(MetricType.CACHE_HIT, partition="electronics")
```

#### 12 Metric Types

| Metric Type | Kind | Description |
|-------------|------|-------------|
| `SEARCH_LATENCY` | Timing | End-to-end search duration |
| `SEARCH_PARTITION_LATENCY` | Timing | Per-partition search duration |
| `EMBEDDING_LATENCY` | Timing | Embedding API call duration |
| `RERANKING_LATENCY` | Timing | Reranking API call duration |
| `SEARCH_RESULTS` | Count | Number of results returned |
| `SEARCH_CANDIDATES` | Count | Number of candidates before filtering |
| `PARTITION_COUNT` | Gauge | Total active partitions |
| `DETECTION_RUN` | Count | Detection pipeline executions |
| `INDEX_BUILD_TIME` | Timing | Time to build a vector index |
| `CACHE_HIT` | Count | Embedding cache hits |
| `CACHE_MISS` | Count | Embedding cache misses |
| `ERROR` | Count | Errors by type |

A `NoOpCollector` is provided for deployments that don't need metrics. Configuration:

```json
{
  "metrics": {
    "enabled": true,
    "include_partition_tags": true,
    "include_query_tags": false
  }
}
```

### Embedding Cache (`utils/cache.py`)

#### LRU + TTL Cache

The `EmbeddingCache` provides in-memory caching of embedding vectors to skip repeated API calls:

```python
cache = EmbeddingCache(max_size=10_000, ttl_seconds=3600)

key = CacheKey(text="wireless headphones", model="voyage-4-lite", dimensions=1024, input_type="query")
vector = cache.get(key)  # None on miss
if vector is None:
    vector = await embedder.embed("wireless headphones")
    cache.put(key, vector)
```

**Key properties:**
- **Thread-safe**: Uses `threading.Lock` for all mutations
- **LRU eviction**: Oldest-accessed entry evicted when at capacity
- **TTL expiration**: Entries expire after `ttl_seconds` (checked on access)
- **Disableable**: Set `max_size=0` or `cache.enabled=False` in config

**Cache key includes:**
- `text`: The input text
- `model`: The embedding model name
- `dimensions`: Output dimensions
- `input_type`: "query" or "document" (different embeddings for asymmetric models)

#### Cache Statistics

```python
stats = cache.stats()
# {
#   "size": 4200,
#   "max_size": 10000,
#   "hits": 15000,
#   "misses": 4200,
#   "hit_rate": 0.781,
#   "evictions": 0
# }
```

Available via CLI: `svr cache stats`

Configuration:
```json
{
  "cache": {
    "enabled": true,
    "max_size": 10000,
    "ttl_seconds": 3600
  }
}
```

### Connection Pool Tuning

MongoDB connection pool parameters are exposed via `DatabaseConfig`:

```json
{
  "database": {
    "max_pool_size": 100,
    "min_pool_size": 0,
    "max_idle_time_ms": 0,
    "wait_queue_timeout_ms": 0
  }
}
```

These map directly to PyMongo's `AsyncMongoClient` parameters:
- `maxPoolSize`: Maximum connections in the pool (default 100)
- `minPoolSize`: Minimum connections maintained (default 0)
- `maxIdleTimeMS`: Close idle connections after this duration (0 = unlimited)
- `waitQueueTimeoutMS`: Error if no connection available within this time (0 = unlimited)

---

## Testing Strategy

### Test Organization

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/                          # Unit tests (no external deps, all mocked)
│   ├── test_client.py             # SVRClient tests
│   ├── test_config.py             # Config loading/saving
│   ├── test_models.py             # Pydantic model tests
│   ├── test_routing.py            # Resolver and merger tests
│   ├── test_embedders.py          # Embedder provider tests
│   ├── test_rerankers.py          # Reranker tests
│   ├── test_mongodb_backend.py    # MongoDB backend tests
│   ├── test_provisioner.py        # Provisioner tests
│   ├── test_scanner.py            # Scanner tests
│   ├── test_splitter.py           # Splitter tests
│   ├── test_watcher.py            # Watcher tests
│   ├── test_monitor.py            # Monitor tests
│   ├── test_vector_conversion.py  # BinData vector conversion tests
│   ├── test_field_analyzer.py     # Field analyzer tests
│   ├── test_retry.py              # Retry decorator tests
│   ├── test_logging.py            # Structured logging tests
│   ├── test_metrics.py            # Metrics hooks tests
│   ├── test_cache.py              # Embedding cache tests
│   ├── test_metadata_store.py     # MetadataStore tests
│   ├── test_detector.py           # Detection pipeline tests
│   ├── test_repartition.py        # Repartition engine tests
│   ├── test_cli.py                # CLI command tests (init, partitions, search, etc.)
│   ├── test_cli_monitor.py        # Monitor CLI tests
│   └── test_cli_repartition.py    # Repartition CLI tests
├── functional/                    # Functional tests against real MongoDB Atlas
│   └── test_end_to_end.py         # End-to-end workflow tests
└── integration/                   # Integration tests
    └── test_mongodb_backend.py    # MongoDB backend integration
```

### Test Counts

- **~750 unit tests** across 24 test files (558 test functions + parametrized expansions)
- **47+ functional tests** against real MongoDB Atlas
- All unit tests are fully mocked -- no network calls

### Key Testing Patterns

**1. Mock at use-site, not definition-site:**
```python
# CORRECT: Patch where it's imported
@patch("semantic_vector_router.cli.partitions.load_config")

# WRONG: Patch where it's defined
@patch("semantic_vector_router.config.load_config")
```

**2. Structured embeddings for deterministic functional tests:**

Functional tests use orthogonal category centroids (32 dimensions) instead of real embedding API calls. Each category gets a deterministic centroid vector, ensuring that search results are reproducible and don't depend on external API behavior.

**3. Click CliRunner for CLI tests:**
```python
from click.testing import CliRunner
from semantic_vector_router.cli import main

runner = CliRunner()
result = runner.invoke(main, ["partitions", "list"])
assert result.exit_code == 0
```

### Key Fixtures

```python
@pytest.fixture
def sample_config() -> SVRConfig:
    """Standard test configuration."""

@pytest.fixture
def sample_config_with_partitions(sample_config) -> SVRConfig:
    """Config with pre-registered partitions."""

@pytest.fixture
def mock_backend():
    """AsyncMock of MongoDB backend."""

@pytest.fixture
def mock_embedder():
    """AsyncMock of embedder."""
```

### Running Tests

```bash
# All unit tests
.venv/bin/pytest tests/unit/ -v

# Functional tests (requires MONGODB_URI and VOYAGE_API_KEY)
.venv/bin/pytest tests/functional/ -v -s

# With coverage
.venv/bin/pytest --cov=semantic_vector_router --cov-report=html
```

---

## Future Considerations

### Recently Implemented

1. **Three index location modes**: VIEWS (index per partition), SOURCE (single index with filtering), FIELDS (per-partition embedding fields)
2. **Voyage 4 support**: Shared embedding space with asymmetric embeddings
3. **Flexible dimensions**: Matryoshka learning support (256, 512, 1024, 2048)
4. **Quantization options**: float, int8, uint8, binary, ubinary
5. **Connection retry**: `@with_retry` decorator with exponential backoff, jitter, and `ResilienceConfig`
6. **Embedding cache**: LRU + TTL in-memory cache for repeated query embeddings
7. **Metadata store**: Database-backed partition state with distributed locking
8. **Detection pipeline**: 5 signal types with trend analysis and auto-split triggering
9. **Repartition engine**: Zero-downtime 5-step workflow with resume and rollback
10. **Structured logging**: JSON formatter, correlation IDs, operation timing
11. **Metrics hooks**: Pluggable `MetricsHandler` protocol with 12 metric types
12. **Full CLI**: 11 command groups covering all SDK operations

### Planned Enhancements (V2+)

1. **Inference-based routing**: Use LLM to determine relevant partitions from query
   ```python
   results = await svr.search("laptop for coding", infer_partitions=True)
   ```

2. **Learned partitioning**: Automatically discover optimal partition boundaries

3. **Cross-database support**: Add backends for:
   - PostgreSQL (pgvector)
   - Pinecone
   - Qdrant
   - Weaviate

4. **Analytics dashboard**: Query patterns, hit rates, latency metrics, partition utilization trends

5. **Managed service**: Hosted SVR that handles lifecycle as a service

6. **Webhook integration**: Notify external systems of:
   - New partitions created
   - Threshold breaches
   - Split completions

7. **Hybrid search**: Combine vector search with keyword/full-text search

### Known Limitations

1. **Single database**: Currently supports one database per config
2. **Synchronous config saves**: Could block on large configs
3. **In-memory cache only**: Embedding cache is per-process, not shared across workers (a distributed cache like Redis could be added in V2)

### Performance Considerations

1. **Parallel queries**: Limited by `max_partitions_per_query` (default 5)
2. **Reranking latency**: Adds 50-200ms per query
3. **View overhead**: Complex view pipelines add query latency
4. **Index build time**: Vector indexes can take minutes to build
5. **Voyage 4 asymmetric**: Query embedding ~3x faster with voyage-4-lite vs voyage-4-large
6. **Embedding cache**: Hit rates of 50-80% typical for applications with repeated queries, saving both latency and API costs
7. **Connection pooling**: Tune `max_pool_size` based on concurrency; default 100 is suitable for most workloads

---

## Appendix: Quick Reference

### Common Operations

```python
# Initialize
svr = SVRClient()
await svr.connect()

# Search
results = await svr.search("query", partitions=["a", "b"], limit=10)

# List partitions
partitions = svr.list_partitions()

# Create partition
await svr.create_partition("new_category")

# Check health
health = await svr.check_partition_health()

# Start watcher
await svr.start_watcher()

# Get partition (async)
partition = await svr.get_partition("electronics")

# Cleanup
await svr.disconnect()
```

### CLI Commands

```bash
# Setup
svr init                              # Interactive setup wizard
svr init --non-interactive --database my_db --collection products --field category

# Partition management
svr partitions list                   # List all partitions
svr partitions status                 # Detailed partition status
svr partitions create <name>          # Create a partition
svr partitions delete <name>          # Delete a partition
svr partitions refresh                # Refresh document counts
svr partitions scan                   # Scan for new partition values
svr partitions provision              # Provision pending partitions

# Search
svr search "wireless headphones" --partitions electronics,furniture

# Analyze
svr analyze                           # Analyze fields for filter suitability

# Watch (change streams)
svr watch start                       # Start watching for new partitions
svr watch status                      # Show watcher status
svr watch confirm <name>              # Confirm a pending partition
svr watch reject <name>               # Reject a pending partition
svr watch confirm-all                 # Confirm all pending partitions

# Split
svr split check                       # Check which partitions need splitting
svr split execute <name>              # Execute a split

# Config
svr config show                       # Show current configuration
svr config validate                   # Validate config file
svr config set <key> <value>          # Set a config value
svr config path                       # Show config file path

# Index
svr index status                      # Show index status for all partitions
svr index rebuild <name>              # Rebuild an index
svr index wait                        # Wait for all indexes to be queryable

# Monitor (detection pipeline)
svr monitor check                     # Run detection pipeline
svr monitor history <name>            # Show health history for a partition

# Repartition
svr repartition pending               # List pending repartition operations
svr repartition execute <op_id>       # Execute a repartition operation
svr repartition status <op_id>        # Show operation status
svr repartition rollback <op_id>      # Rollback a failed operation

# Cache
svr cache stats                       # Show embedding cache statistics
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `MONGODB_URI` | MongoDB connection string |
| `OPENAI_API_KEY` | OpenAI embedding/rerank API |
| `VOYAGE_API_KEY` | Voyage AI embedding/rerank API |
| `COHERE_API_KEY` | Cohere embedding/rerank API |
| `METADATA_MONGODB_URI` | (Optional) Separate MongoDB for metadata store |

---

*Document generated for code review purposes. Last updated: February 2026.*
