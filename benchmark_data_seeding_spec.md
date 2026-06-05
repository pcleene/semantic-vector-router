# SVR Phase 18 — Benchmark Data Seeding Specification

## E-Commerce Product Catalog · MongoDB Single Collection

**Purpose:** Seed a single `products` collection with synthetic e-commerce data designed to expose real scalability pain points at 10M, 50M, and 100M vector scale. Every design choice here exists to stress-test something specific.

---

## 1. The Collection: `products`

One collection. No sharding. One `$vectorSearch` index on `embedding` (1024 dims, cosine). This is the whole point — prove that SVR's partitioning on an unsharded single collection beats flat vector search.

---

## 2. Document Schema

```json
{
  "_id": ObjectId,
  "sku": "EL-SMPH-APL-001-BLK-256",
  "partition": "smartphones",

  "title": "Apple iPhone 15 Pro Max 256GB Space Black",
  "description": "Long-form product description, 50–800 words...",

  "brand": {
    "name": "Apple",
    "slug": "apple",
    "tier": "premium"
  },

  "category": {
    "l1": "Electronics",
    "l2": "Phones & Tablets",
    "l3": "Smartphones",
    "path": "electronics/phones-tablets/smartphones"
  },

  "pricing": {
    "currency": "USD",
    "list_price": 1199.00,
    "sale_price": 1049.00,
    "cost": 780.00,
    "margin_pct": 25.6,
    "flash_sale": {
      "active": false,
      "price": null,
      "ends_at": null
    }
  },

  "variants": [
    {
      "variant_id": "APL-15PM-BLK-256",
      "attributes": { "color": "Space Black", "storage": "256GB" },
      "price_modifier": 0.0,
      "in_stock": true,
      "stock_qty": 142,
      "warehouse_stock": {
        "us-east": 58, "us-west": 44, "eu-central": 30, "ap-southeast": 10
      }
    },
    {
      "variant_id": "APL-15PM-NAT-512",
      "attributes": { "color": "Natural Titanium", "storage": "512GB" },
      "price_modifier": 100.0,
      "in_stock": true,
      "stock_qty": 87,
      "warehouse_stock": {
        "us-east": 32, "us-west": 28, "eu-central": 17, "ap-southeast": 10
      }
    }
  ],

  "specifications": [
    { "key": "Screen Size", "value": "6.7", "unit": "inches" },
    { "key": "Battery", "value": "4422", "unit": "mAh" },
    { "key": "Processor", "value": "A17 Pro", "unit": null },
    { "key": "Weight", "value": "221", "unit": "g" },
    { "key": "RAM", "value": "8", "unit": "GB" },
    { "key": "OS", "value": "iOS 17", "unit": null }
  ],

  "reviews": {
    "avg_rating": 4.3,
    "count": 2847,
    "distribution": [12, 45, 234, 891, 1665],
    "recent": [
      {
        "reviewer_id": "usr_abc123",
        "rating": 5,
        "title": "Best phone I've owned",
        "body": "Detailed review, 20–300 words...",
        "date": ISODate("2026-01-15T08:30:00Z"),
        "helpful_votes": 42,
        "verified_purchase": true,
        "images": ["img_rev_001.jpg"]
      }
    ]
  },

  "availability": {
    "status": "in_stock",
    "regions": ["us", "eu", "ap"],
    "warehouses": ["us-east", "us-west", "eu-central", "ap-southeast"],
    "lead_time_days": 0,
    "backorder_allowed": false,
    "last_restock": ISODate("2026-02-10T00:00:00Z")
  },

  "tags": ["flagship", "5g", "premium", "bestseller", "free-shipping"],

  "compatible_with": [
    ObjectId("..."),
    ObjectId("..."),
    ObjectId("...")
  ],

  "marketplace": {
    "seller_id": "seller_xyz",
    "seller_name": "TechStore Official",
    "seller_rating": 4.8,
    "seller_sales": 125000,
    "fulfilled_by": "marketplace"
  },

  "media": [
    { "type": "image", "url": "https://cdn.example.com/...", "alt": "Front view", "position": 1 },
    { "type": "image", "url": "https://cdn.example.com/...", "alt": "Side view", "position": 2 },
    { "type": "video", "url": "https://cdn.example.com/...", "duration_sec": 90, "position": 3 }
  ],

  "embedding_fields": {
    "partition": "smartphones",
    "brand": "Apple",
    "tier": "premium",
    "title": "iPhone 15 Pro Max 256GB Space Black",
    "description": "The iPhone 15 Pro Max features a titanium design, A17 Pro chip...",
    "specs": {
      "Screen Size": "6.7 inches",
      "Battery": "4422 mAh",
      "Processor": "A17 Pro"
    },
    "tags": ["flagship", "5g", "premium", "bestseller"]
  },
  "embedding": [ /* 1024 float32 values */ ],

  "meta": {
    "created_at": ISODate("2025-09-15T12:00:00Z"),
    "updated_at": ISODate("2026-02-10T08:00:00Z"),
    "indexed_at": ISODate("2026-02-10T08:05:00Z"),
    "source": "catalog_feed",
    "doc_version": 3
  }
}
```

---

## 3. Why This Schema Hurts at Scale

The primary scalability pain is vector search accuracy and latency degradation as the HNSW index grows. Document size variance is a secondary amplifier. Every design choice targets one or both.

### 3.0 The Core Problem: HNSW Recall Collapses at Scale

HNSW (Hierarchical Navigable Small World) graph indexes are the de-facto standard for approximate nearest-neighbor search. They work brilliantly at small to moderate scale (1M–10M vectors) but exhibit measurable recall degradation at 50M–100M+ vectors, even with properly tuned parameters.

**HNSW recall@10 degradation by dataset size:**

| Dataset Size | Uniform Data (recall@10) | Adversarial Data (recall@10, semantic clustering) |
|---|---|---|
| 1M vectors | 0.98 | 0.95 |
| 10M vectors | 0.94 | 0.87 |
| 50M vectors | 0.88 | 0.72 |
| 100M vectors | 0.82 | 0.61 |

*Uniform data = random embeddings. Adversarial data = dense clusters with near-duplicates (realistic e-commerce scenario).*

**The ef_search tradeoff:**

MongoDB Atlas uses `ef_search` (exploration factor at search time) to tune recall vs latency. At 100M vectors:

- **Keep ef=100 (default):** Fast queries (p95 ~80ms), but terrible recall (0.61 on adversarial data).
- **Raise to ef=300-400:** Recover recall to ~0.88, but p95 latency increases 3-4× to ~250-320ms.

**Why SVR fixes this:**

SVR partitions the collection into 20 semantic buckets (e.g., `smartphones`, `accessories`, `laptops-computers`). Instead of searching 100M vectors with a degraded HNSW graph, the router sends the query to the relevant partition(s), searching ~500K–14M vectors. At these smaller scales, HNSW recall remains high (0.93–0.98) with standard ef_search=100.

**The graph memory cliff-edge:**

HNSW graphs must fit in RAM. At 100M × 1024 dims × float32:
- Vector data alone: 400GB
- HNSW graph overhead: +50% → **600GB total**
- M60 search node: 64GB RAM → **90%+ of the index is on disk**

When the graph doesn't fit in RAM, every search traversal incurs random disk I/O. Latency spikes from ~80ms (in-RAM) to 800ms+ (disk-bound). SVR keeps each partition's index small enough to stay RAM-resident, even at 100M total scale.

**The following subsections (3.1–3.6) describe how the document model amplifies the core HNSW problem above.**

### 3.1 Document Size Variance (2KB → 60KB+)

| Product Type | Variants | Reviews (recent) | Specs | Approx Doc Size |
|---|---|---|---|---|
| USB cable, basic accessory | 1 | 0 | 3 | ~2 KB |
| Mid-range headphones | 3 | 5 | 8 | ~8 KB |
| Running shoe (color × size matrix) | 30+ | 15 | 12 | ~25 KB |
| Flagship phone (all configs) | 12 | 50 | 25 | ~45 KB |
| High-end laptop (every SKU) | 20 | 50 | 30 | ~60 KB |

**What this stresses:** When Atlas `$vectorSearch` finds the top-100 nearest vectors, it must fetch and return full documents. If those 100 docs average 30KB, that's 3MB of document reads per query. At 50 concurrent queries, that's 150MB/sec of random reads. The M30 (8GB RAM) will be evicting cached documents constantly. The M50 will cope at 10M but struggle at 100M. The point is to show that SVR (which only searches within a partition) reads fewer, more targeted documents.

### 3.2 Embedded Reviews Array — The Silent Killer

The `reviews.recent` array is deliberately variable-length (0 to 50 embedded review objects, each 200–2000 bytes). This is the single biggest contributor to document size variance.

**Distribution:**
- 40% of products: 0 recent reviews (new/unpopular items)
- 30% of products: 1–5 reviews
- 20% of products: 6–20 reviews
- 8% of products: 21–40 reviews
- 2% of products: 41–50 reviews (bestsellers)

**What this stresses:** WiredTiger's block compression works best with uniform document sizes. When a query returns a mix of 2KB and 50KB docs, the cache is used inefficiently. Also, the storage engine must read full documents — it can't skip the reviews array even if you don't need it. This is a real-world pattern that every e-commerce platform suffers from.

### 3.3 Variant Explosion

The `variants` array ranges from 1 (simple product) to 30+ (fashion items with color × size matrix). Each variant carries per-warehouse stock levels — a nested embedded object inside an array of embedded objects.

**Distribution:**
- 25% of products: 1 variant (single-option items)
- 35% of products: 2–4 variants
- 25% of products: 5–12 variants
- 10% of products: 13–20 variants
- 5% of products: 21–35 variants (fashion, configurable items)

**What this stresses:** At 100M docs, the variants array alone could hold 500M+ embedded objects. Any query that touches variant data (stock checks, price comparisons) must deserialize the entire array per document. This makes filtered vector search (e.g., "find products in stock in eu-central") expensive at the document-read layer, even after the vector index narrows candidates.

### 3.4 Semantic Near-Collision (The Recall Destroyer)

Within each partition, we deliberately generate clusters of products that are semantically near-identical:

- "Wireless Bluetooth Earbuds with Active Noise Cancellation — Black"
- "Bluetooth Wireless Earbuds Active Noise Cancelling — Midnight"
- "ANC Wireless Earbuds Bluetooth 5.3 — Jet Black"
- "True Wireless Earbuds with ANC, Bluetooth — Onyx"

These products differ in wording but are functionally identical. Their embeddings will have cosine similarity > 0.97. At 10M vectors, there are thousands of these near-collision clusters.

**What this stresses:** The HNSW index's approximate nearest-neighbor algorithm must distinguish between genuinely relevant results and near-duplicates. When the true answer set is surrounded by thousands of near-misses, recall@10 drops significantly. This is where SVR's partition-scoped search should win — searching 500K vectors (one partition) instead of 10M means the HNSW graph is denser in the relevant region and recall improves.

**Generation rule:** For every "hero" product, generate 3–8 semantic siblings with:
- Synonymous phrasing (active noise cancellation ↔ ANC ↔ noise cancelling)
- Reordered title components
- Similar but not identical descriptions
- Different brands (20% same brand, different model)

### 3.5 Power-Law Partition Distribution

The 20 partitions are deliberately unbalanced. This breaks naive partitioning strategies and tests SVR's ability to handle asymmetric partition sizes.

| # | Partition | % of Total | At 10M | At 100M |
|---|---|---|---|---|
| 1 | `accessories` | 14.0% | 1,400,000 | 14,000,000 |
| 2 | `womens-fashion` | 11.0% | 1,100,000 | 11,000,000 |
| 3 | `mens-fashion` | 9.5% | 950,000 | 9,500,000 |
| 4 | `home-kitchen` | 8.5% | 850,000 | 8,500,000 |
| 5 | `beauty-personal-care` | 7.5% | 750,000 | 7,500,000 |
| 6 | `smartphones` | 6.0% | 600,000 | 6,000,000 |
| 7 | `sports-outdoors` | 5.5% | 550,000 | 5,500,000 |
| 8 | `health-wellness` | 5.0% | 500,000 | 5,000,000 |
| 9 | `audio-headphones` | 4.5% | 450,000 | 4,500,000 |
| 10 | `kids-baby` | 4.0% | 400,000 | 4,000,000 |
| 11 | `laptops-computers` | 3.5% | 350,000 | 3,500,000 |
| 12 | `home-furniture` | 3.5% | 350,000 | 3,500,000 |
| 13 | `grocery-gourmet` | 3.0% | 300,000 | 3,000,000 |
| 14 | `tv-home-theater` | 2.5% | 250,000 | 2,500,000 |
| 15 | `gaming-consoles` | 2.5% | 250,000 | 2,500,000 |
| 16 | `wearables` | 2.5% | 250,000 | 2,500,000 |
| 17 | `cameras-photo` | 2.0% | 200,000 | 2,000,000 |
| 18 | `pet-supplies` | 2.0% | 200,000 | 2,000,000 |
| 19 | `automotive` | 1.5% | 150,000 | 1,500,000 |
| 20 | `books-media` | 1.5% | 150,000 | 1,500,000 |

**What this stresses:** The `accessories` partition is 9.3× larger than `books-media`. A flat vector search scans all 10M/100M vectors for every query. SVR routes a "phone case" query to `accessories` (1.4M vectors) and a "sci-fi novel" query to `books-media` (150K vectors). The speedup is asymmetric — SVR's advantage is massive for small-partition queries and moderate for large-partition queries. The benchmark should capture this distribution of speedups, not just the average.

### 3.6 High-Cardinality Filter Fields

These are the fields that will be combined with `$vectorSearch` as pre-filters. Each adds a dimension to the filter-index intersection.

| Filter Field | Cardinality | Example Values |
|---|---|---|
| `partition` | 20 | `smartphones`, `accessories`, etc. |
| `category.l1` | 8 | `Electronics`, `Fashion`, `Home`, etc. |
| `category.l2` | 45 | `Phones & Tablets`, `Audio`, etc. |
| `category.l3` | 180 | `Smartphones`, `Earbuds`, etc. |
| `brand.name` | 2,000 | `Apple`, `Samsung`, `Nike`, etc. |
| `brand.tier` | 4 | `budget`, `mid`, `premium`, `luxury` |
| `pricing.sale_price` | continuous | 0.99 – 9,999.00 (range queries) |
| `availability.status` | 4 | `in_stock`, `low_stock`, `backorder`, `discontinued` |
| `availability.regions` | 4 elements | `us`, `eu`, `ap`, `other` |
| `reviews.avg_rating` | continuous | 1.0 – 5.0 (range queries) |
| `marketplace.fulfilled_by` | 2 | `marketplace`, `seller` |
| `tags` | 200 unique | `bestseller`, `free-shipping`, `eco`, etc. |

**What this stresses:** The real query pattern is never just vector search — it's always `$vectorSearch` + filter. When the pre-filter on `brand.name = "Apple"` AND `pricing.sale_price <= 500` narrows candidates to 50K docs out of 10M, the index must efficiently intersect the filter bitmap with the vector HNSW traversal. At 100M docs, this intersection becomes the bottleneck — not the vector search itself. SVR's partitioning reduces the vector index size, making the filter intersection cheaper because there are fewer candidates to evaluate.

### 3.7 Embedding Fields (Structured Object, Not Concatenated Text)

The `embedding_fields` field stores a structured JSON object containing the data passed to the embedding model (voyage-4-nano). This follows SVR's Phase 16 smart embedding pattern — instead of concatenating everything into a pipe-delimited string, we preserve structure.

**Why a structured object instead of concatenated text:**

1. **Field names provide semantic context.** The embedding model sees `{"title": "iPhone 15 Pro Max", "specs": {...}}`, not `"iPhone 15 Pro Max | 6.7 inches | 4422 mAh"`. Field labels help the model distinguish between "title text" and "spec values."

2. **Easier to audit and debug.** When centroid routing makes a mistake, you can inspect `embedding_fields` to see exactly what was embedded, with labels intact.

3. **Extensible.** Adding a new field (e.g., `"warranty_years": 2`) doesn't require re-parsing a concatenated string.

**Structure:**

```json
{
  "partition": "smartphones",
  "brand": "Apple",
  "tier": "premium",
  "title": "iPhone 15 Pro Max 256GB Space Black",
  "description": "The iPhone 15 Pro Max features a titanium design, A17 Pro chip with 6-core GPU, 48MP main camera system with 5x optical zoom, and USB-C... (truncated to ~200 words)",
  "specs": {
    "Screen Size": "6.7 inches",
    "Battery": "4422 mAh",
    "Processor": "A17 Pro",
    "RAM": "8 GB"
  },
  "tags": ["flagship", "5g", "premium", "bestseller"]
}
```

**Why reviews are excluded:** Reviews are user-generated content that changes over time and introduces noise into the embedding. A product's semantic identity should be stable based on its intrinsic properties (title, description, specs), not volatile based on recent customer sentiment.

**Why the partition prefix matters:** The `partition` field is always first in the object. This biases the embedding toward the product category, which helps SVR's centroid routing make correct partition assignments. Without this prefix, a query for "phone case" might embed closer to `smartphones` than `accessories` because of shared vocabulary. The partition prefix reinforces the category boundary.

---

## 4. Brand & Seller Distribution

### 4.1 Brands (2,000 total)

Follow a Zipf distribution — top 20 brands account for 35% of all products.

| Tier | Count | % of Products | Examples |
|---|---|---|---|
| `luxury` | 50 brands | 3% | Hermès, Bose, Bang & Olufsen |
| `premium` | 200 brands | 22% | Apple, Samsung, Nike, Sony |
| `mid` | 500 brands | 40% | Anker, Logitech, New Balance |
| `budget` | 1,250 brands | 35% | Generic/white-label, store brands |

**The trap:** Budget brands produce massive numbers of accessory SKUs (cables, cases, adapters). This inflates the `accessories` partition and creates a long tail of semantically similar products from different no-name brands. This is realistic — Amazon has exactly this problem.

### 4.2 Sellers (10,000 total)

| Seller Type | Count | % of Products |
|---|---|---|
| Marketplace-fulfilled (1P) | 50 | 30% |
| Pro sellers (>10K sales) | 500 | 35% |
| Mid sellers (1K–10K) | 2,000 | 25% |
| Small sellers (<1K) | 7,450 | 10% |

---

## 5. Temporal Patterns

### 5.1 Product Age Distribution

| Age | % of Catalog | Description |
|---|---|---|
| 0–30 days | 8% | New arrivals |
| 1–6 months | 25% | Current season |
| 6–12 months | 30% | Established products |
| 1–2 years | 25% | Mature, many reviews |
| 2+ years | 12% | Long-tail / evergreen |

**What this stresses:** Newer products have few/no reviews (small docs), old products have many reviews (large docs). This creates a temporal correlation with document size that can cause cache behavior issues — a "new arrivals" query returns small docs efficiently, but a "top-rated" query hits the fattest documents.

### 5.2 Pricing Patterns

- **Price clustering:** 40% of products are priced at .99 endings ($9.99, $19.99, $49.99, $99.99). This creates dense price clusters that range queries must handle.
- **Price range:** $0.99 to $9,999.00, log-normal distribution with median at $34.99.
- **Flash sales:** 5% of products have `pricing.flash_sale.active = true` at any given time. These products have two valid prices, which complicates filtered search.

---

## 6. Deterministic Seeding Rules

All data is generated from a single `seed` value. Given the same seed, every run produces identical output. This is critical for reproducible benchmarks.

### 6.1 SKU Generation

Format: `{L1_CODE}-{L3_CODE}-{BRAND_CODE}-{SEQ}-{VARIANT_ATTRS}`

Example: `EL-SMPH-APL-00142-BLK-256`

The SKU encodes partition membership, making it easy to verify partition distribution without reading the full document.

### 6.2 Batch Streaming

Documents are generated in batches of 10,000. Each batch is:
1. Generated in memory (documents without embeddings)
2. Serialized to BSON
3. Bulk-inserted with `ordered=False`
4. Embedding text extracted and queued for the embedding server
5. Embeddings written back via bulk `update_many` with `$set`

**Critical:** The generator never holds more than one batch (10K docs) in memory. At 100M docs, this means 10,000 batch cycles. The embedding cache (a side collection keyed by hash of `embedding_fields`) prevents re-computing embeddings for the incremental 50M → 100M step.

### 6.3 Scale Subsets

Datasets are additive — the first 10M documents at seed=42 are identical whether you generate 10M or 100M. This is achieved by deterministic RNG seeding per document index:

```
doc_rng = Random(seed=global_seed + doc_index)
```

So document #7,000,000 is identical regardless of total dataset size.

---

## 7. Index Configuration

### 7.1 Vector Search Index (Atlas)

```json
{
  "name": "vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "path": "embedding",
        "type": "vector",
        "numDimensions": 1024,
        "similarity": "cosine"
      },
      { "path": "partition", "type": "filter" },
      { "path": "category.l1", "type": "filter" },
      { "path": "category.l2", "type": "filter" },
      { "path": "brand.tier", "type": "filter" },
      { "path": "availability.status", "type": "filter" },
      { "path": "pricing.sale_price", "type": "filter" },
      { "path": "reviews.avg_rating", "type": "filter" },
      { "path": "tags", "type": "filter" },
      { "path": "marketplace.fulfilled_by", "type": "filter" }
    ]
  }
}
```

**Memory estimate:**
- 10M × 1024 × 4 bytes = **40 GB** vector data
- 50M × 1024 × 4 bytes = **200 GB** vector data
- 100M × 1024 × 4 bytes = **400 GB** vector data
- Plus HNSW graph overhead (~1.5×) and filter indexes

This is why the M60 search node (64GB RAM) is needed at 100M — the vector index alone won't fit in M50 (32GB) memory, forcing disk-based HNSW traversal that tanks latency.

### 7.2 Quantized Index (Scalar/int8)

Same structure, with `quantization: scalar` added. Reduces vector memory by ~4× (400GB → ~100GB at 100M). The benchmark tests whether quantized + SVR partitioning compounds the benefit.

### 7.3 Supporting Indexes

```javascript
// For baseline comparison queries and data verification
db.products.createIndex({ "partition": 1 })
db.products.createIndex({ "sku": 1 }, { unique: true })
db.products.createIndex({ "category.l1": 1, "category.l2": 1, "category.l3": 1 })
db.products.createIndex({ "brand.name": 1 })
db.products.createIndex({ "pricing.sale_price": 1 })
db.products.createIndex({ "reviews.avg_rating": -1 })
db.products.createIndex({ "meta.created_at": -1 })
db.products.createIndex({ "availability.status": 1, "availability.regions": 1 })
```

---

## 8. Query Workload Patterns

Six query types, weighted by real-world frequency. The workload mix ensures we stress different paths through the vector index and filter layer.

### 8.1 Single-Partition Targeted (35% of queries)

**Pattern:** User knows what category they want. SVR should route to exactly one partition.

```
Query: "noise cancelling headphones for running with long battery life"
Expected partition: audio-headphones
Filters: availability.status = "in_stock"
```

**Why it hurts at scale:** At 100M flat, this searches all 100M vectors. With SVR, it searches ~4.5M (the audio-headphones partition). The HNSW graph traversal at 4.5M is ~22× cheaper than at 100M.

### 8.2 Filtered Category Browse (25% of queries)

**Pattern:** User is browsing within a category with specific constraints.

```
Query: "lightweight waterproof jacket"
Expected partition: mens-fashion OR womens-fashion
Filters: category.l2 = "Outerwear", pricing.sale_price <= 150.00, brand.tier IN ["mid", "premium"]
```

**Why it hurts at scale:** The compound filter narrows candidates heavily, but the filter intersection with the vector index is expensive. At 100M, the filter might match 200K docs, but the vector index must still traverse its full graph to find which of those 200K are nearest. SVR reduces the graph size per partition.

### 8.3 Cross-Partition Ambiguous (15% of queries)

**Pattern:** Query is semantically ambiguous — could belong to multiple partitions. Tests SVR's routing accuracy.

```
Query: "yoga mat bag with pockets"
Could route to: sports-outdoors, accessories, health-wellness
```

**Why it matters:** This tests SVR's centroid router. If routing sends this to only one partition, recall could drop vs flat search. If SVR fans out to 2-3 partitions, the speedup is reduced but recall is preserved. The benchmark must capture both latency AND recall for these queries.

### 8.4 Brand-Specific Search (10% of queries)

**Pattern:** User searches within a brand. Tests filter + vector combination.

```
Query: "Apple laptop for video editing"
Filters: brand.name = "Apple"
Expected partition: laptops-computers
```

**Why it hurts:** The `brand.name = "Apple"` filter matches ~2% of the catalog across many partitions. The vector search must find the best matches within this filtered subset. At 100M, that's 2M Apple products — still a large vector search on a pre-filtered subset.

### 8.5 Fan-Out Discovery (10% of queries)

**Pattern:** Broad query with no clear partition. This is SVR's worst case.

```
Query: "gifts under 50 dollars for someone who has everything"
Filters: pricing.sale_price <= 50.00
No clear partition — could be anything
```

**Why it matters:** SVR must fan out to multiple (or all) partitions. This query type should show minimal SVR speedup or even slight slowdown vs flat. Including it in the benchmark is honest — we're not cherry-picking SVR's best case. The ratio of fan-out vs targeted queries determines SVR's overall speedup.

### 8.6 Exact Match + Neighbors (5% of queries)

**Pattern:** Find a specific product and its nearest neighbors (for "similar items" recommendations).

```
Step 1: Fetch product by SKU (not a vector query)
Step 2: Use that product's embedding to find top-20 similar products
Filters: same category.l1, exclude the source product
```

**Why it matters:** This is the classic "more like this" pattern. The embedding of the source product should land very close to its semantic siblings (see section 3.4). Tests whether the vector index can distinguish between truly similar products and near-collision noise.

---

## 9. What "Success" Looks Like

The data model is designed so that:

1. **Flat search recall degrades significantly** — as the collection grows from 10M to 100M, recall@10 drops from 0.94 to 0.82 (uniform data) or 0.87 to 0.61 (adversarial clustering). P95 latency increases ~3-5× due to larger HNSW graph traversal and more document reads.

2. **SVR maintains recall while being faster** — at 100M, SVR searches 500K–14M vectors per partition (depending on query). HNSW recall at these scales stays high (0.93–0.98). P95 latency stays low because partition indexes fit in RAM and graph traversal is cheap.

3. **The ef_search tax is brutal for flat search** — if flat search tries to recover recall at 100M by raising ef_search from 100 to 300-400, latency increases 3-4× (p95 ~250-320ms) just to match SVR's baseline recall. SVR avoids this penalty entirely by keeping partition indexes small.

4. **Document size variance amplifies the effect** — even after the vector index finds candidates, fetching large documents (reviews, variants) is slower on flat (fetching from a 100M collection) than on SVR (fetching from a partition-aware projection).

5. **Quantization compounds with partitioning** — scalar int8 reduces memory by ~4× and SVR reduces the searched set by ~20× (for targeted queries). These are multiplicative, not additive.

**Target metrics at 100M, concurrency=50:**

| Metric | Flat (ef=100, low recall) | Flat (ef=400, recall-matched) | SVR (targeted) | SVR (overall mix) |
|---|---|---|---|---|
| p50 latency | ~80ms | ~180ms | ~20ms | ~35ms |
| p95 latency | ~250ms | ~650ms | ~60ms | ~120ms |
| recall@10 | 0.61–0.82 | 0.88–0.92 | 0.93–0.98 | 0.90–0.95 |
| Throughput (QPS) | ~200 | ~80 | ~800 | ~500 |

---

## 10. Vocabulary & Text Generation Strategy

**No LLM per document.** Text generation is pure template + vocabulary slot-filling, running at ~10K docs/sec on a single CPU core. An LLM is used ONCE to generate the vocabulary pools and templates themselves (a one-time cost of ~$2 and ~10 minutes). Those become static inputs to the deterministic template engine.

### 10.1 Three-Tier Vocabulary Architecture

Vocabulary is organized in three tiers to ensure embedding diversity at 100M scale:

**Tier 1 — Partition vocabulary (20 pools, ~500-2,000 words each):**
Base domain vocabulary shared by all products in a partition.

- **smartphones:** `titanium, AMOLED, refresh-rate, megapixel, 5G, mmWave, eSIM, haptic, ceramic-shield, ProMotion...`
- **beauty-personal-care:** `hyaluronic, retinol, SPF, sulfate-free, peptide, niacinamide, serum, moisturizer, exfoliant, ceramide...`
- **automotive:** `torque, horsepower, OBD-II, all-season, synthetic-blend, ceramic-brake, dashcam, tonneau, catalytic...`

**Tier 2 — L3 micro-vocabularies (180 pools, ~100-150 words each):**
Subcategory-specific terms that create sharper subclusters within partitions. Without this tier, 14M accessories products would form one uniform embedding blob. With it, phone cases, cables, chargers, and screen protectors cluster distinctly.

Examples within `accessories`:
- **phone-cases:** `silicone, bumper, MagSafe, slim-fit, drop-proof, TPU, clear-back, kickstand, wallet-style, antimicrobial...`
- **cables:** `braided, USB-C, Lightning, fast-charge, 10Gbps, right-angle, tangle-free, reinforced-joint, daisy-chain...`
- **chargers:** `GaN, 65W, PD-3.0, multi-port, Qi2, foldable-prongs, travel-adapter, passthrough, LED-indicator...`
- **screen-protectors:** `tempered-glass, oleophobic, 9H-hardness, bubble-free, edge-to-edge, matte-finish, privacy-filter, self-healing...`

**Tier 3 — Brand vocabulary (50 custom + 4 tier defaults):**
Top 50 brands (covering ~35% of products) get brand-specific vocabulary. Remaining 1,950 brands use tier-level defaults.

- **Apple:** `Retina, ProMotion, Ceramic Shield, MagSafe, Dynamic Island, Spatial Audio, Neural Engine...`
- **Samsung:** `Dynamic AMOLED, Exynos, One UI, S Pen, SmartThings, Knox, Infinity Display...`
- **Nike:** `Flyknit, Air Max, React foam, Dri-FIT, ZoomX, Flywire, carbon-fiber plate...`
- **_premium_default:** `precision-engineered, flagship, handcrafted, aerospace-grade, signature...`
- **_mid_default:** `reliable, versatile, well-built, everyday, solid performance...`
- **_budget_default:** `affordable, value, compatible-with, multi-pack, basic, lightweight...`

Each document draws from **partition vocab + L3 micro-vocab + brand vocab**. The three tiers combine to create a rich, realistic embedding distribution where:
- Products in different partitions are well-separated
- Products in different L3 categories within a partition form distinct subclusters
- Products from different brands within the same category have subtle vocabulary differences

**Cross-partition contamination (deliberate):** 10% of each product's description includes terms from an adjacent partition's vocabulary. A phone case description might include "premium leather" (fashion vocab) or a fitness tracker description might include "heart rate zones" (health-wellness vocab). This creates the semantic bleed that makes routing non-trivial.

**Vocabulary generation:** All three tiers are generated ONCE using an LLM (Claude or GPT-4), reviewed for quality, and stored as static Python dictionaries in `benchmark/vocabulary.py`. Total: ~180 L3 pools + 50 brand pools + 20 partition pools = ~250 vocabulary pools, each 100-2,000 words. One-time cost: ~$2, ~10 minutes.

### 10.2 Paragraph-Fragment Description Generation

Descriptions are composed from **pre-written sentence/paragraph fragments**, not word-level slot-filling. Each fragment is a coherent, self-contained piece of product description text written by an LLM once and stored statically. The runtime engine picks and combines fragments — no LLM is called per document, runs at ~10K docs/sec on CPU.

**Why fragments, not word-level templates:** Word-level slot-filling (e.g., `{adj} {product_type} with {feature}`) produces nonsensical combinations at scale — "premium braided phone case with oleophobic bumper" doesn't make sense. Fragment-level composition guarantees every piece reads coherently because each piece WAS written coherently by an LLM.

**Fragment structure per L3 category (~100 fragments each):**

```python
# Generated ONCE by LLM, stored in benchmark/vocabulary.py
# Each fragment is tagged with a topic to prevent redundancy (see 10.3)
FRAGMENTS["phone-cases"] = {
    "intro": [  # 30 fragments
        {"text": "Designed for everyday protection without the bulk.", "topic": "protection"},
        {"text": "Military-grParts Distributor drop protection meets minimalist design.", "topic": "protection"},
        {"text": "Keep your phone safe without hiding its original look.", "topic": "aesthetics"},
        {"text": "The case your phone deserves — slim, tough, and stylish.", "topic": "general"},
        # ...
    ],
    "features": [  # 40 fragments, ~8-10 distinct topics
        {"text": "The raised bezels guard your screen and camera from flat-surface drops.", "topic": "drop_protection"},
        {"text": "Precision cutouts ensure full access to all ports and buttons.", "topic": "ports"},
        {"text": "Built-in magnets align perfectly with MagSafe accessories.", "topic": "wireless"},
        {"text": "The textured grip prevents accidental slips without adding thickness.", "topic": "grip"},
        # ...
    ],
    "closing": [  # 20 fragments
        {"text": "Available in 12 colors to match your style.", "topic": "colors"},
        {"text": "Backed by our lifetime warranty against manufacturing defects.", "topic": "warranty"},
        {"text": "Ships with a free microfiber cleaning cloth.", "topic": "extras"},
        # ...
    ],
    "comparison": [  # 10 fragments (medium/long only)
        {"text": "Unlike cheaper alternatives, this case maintains its shape after months of daily use.", "topic": "durability"},
        {"text": "Where most cases sacrifice looks for protection, this one delivers both.", "topic": "general"},
        # ...
    ],
}
```

**Runtime composition with topic deduplication:**

```python
def pick_fragment(doc_rng, fragments, used_topics):
    """Pick a fragment whose topic hasn't been used yet."""
    candidates = [f for f in fragments if f["topic"] not in used_topics]
    if not candidates:
        candidates = fragments  # fallback if all topics exhausted
    chosen = candidates[doc_rng.randint(0, len(candidates) - 1)]
    used_topics.add(chosen["topic"])
    return chosen["text"]

def generate_description(doc_rng, l3_category, brand, template_tier):
    frags = FRAGMENTS[l3_category]
    used_topics = set()

    if template_tier == "short":   # 50-100 words — budget/accessory
        parts = [
            pick_fragment(doc_rng, frags["intro"], used_topics),
            pick_fragment(doc_rng, frags["features"], used_topics),
        ]
    elif template_tier == "medium":  # 100-300 words — mid-range
        parts = [
            pick_fragment(doc_rng, frags["intro"], used_topics),
            pick_fragment(doc_rng, frags["features"], used_topics),
            pick_fragment(doc_rng, frags["features"], used_topics),  # different topic guaranteed
            pick_fragment(doc_rng, frags["comparison"], used_topics),
            pick_fragment(doc_rng, frags["closing"], used_topics),
        ]
    else:  # "long" — 300-800 words — premium/flagship
        parts = [
            pick_fragment(doc_rng, frags["intro"], used_topics),
            pick_fragment(doc_rng, frags["features"], used_topics),
            pick_fragment(doc_rng, frags["features"], used_topics),
            pick_fragment(doc_rng, frags["features"], used_topics),
            pick_fragment(doc_rng, frags["comparison"], used_topics),
            pick_fragment(doc_rng, frags["comparison"], used_topics),
            pick_fragment(doc_rng, frags["closing"], used_topics),
        ]

    # Inject brand vocabulary into 1-2 fragments
    brand_inject = get_brand_phrase(doc_rng, brand)  # e.g., "Powered by MagSafe technology."
    parts.insert(doc_rng.randint(1, len(parts) - 1), brand_inject)

    return " ".join(parts)
```

The `used_topics` set ensures no two fragments in the same description cover the same aspect. A medium description with two feature fragments will always describe two DIFFERENT features (e.g., drop protection + wireless charging), never the same feature twice with different wording.

**Diversity math at 100M:**
- Per L3 category: 30 intros × 40 features × 20 closings = **24,000 unique short descriptions**
- Medium (with 2nd feature + comparison): 30 × 40 × 39 × 10 × 20 = **9.4M unique combinations**
- With 180 L3 categories: **1.7B unique medium descriptions** before brand/price variation
- Long descriptions (more fragment picks): combinatorial space exceeds 100M easily

At 100M documents, descriptions read like real product listings because every fragment WAS written as a real product description fragment. The engine just composes them.

**Fragment generation:** All fragments are generated ONCE by an LLM (Claude Code session), reviewed for quality, and stored as static Python dictionaries. Total: ~100 fragments × 180 L3 categories = ~18,000 fragments. One session, ~$2, reuse forever.

---

## 11. Seeding Execution Summary

| Phase | Scale | New Docs | Embedding Time (4× A10G) | Insert Time | Total |
|---|---|---|---|---|---|
| Dev test | 1K | 1,000 | ~10 sec | ~5 sec | ~30 sec |
| Local test | 100K | 100,000 | ~3 min | ~2 min | ~10 min |
| Smoke test | 1M | 1,000,000 | ~12 min | ~8 min | ~30 min |
| Scale 1 | 10M | 10,000,000 | ~25 min | ~40 min | ~1.5 hr |
| Scale 2 | 50M | 40,000,000 (incremental) | ~1.5 hr | ~2.5 hr | ~5 hr |
| Scale 3 | 100M | 50,000,000 (incremental) | ~2 hr | ~3 hr | ~6 hr |

**Total for full 100M:** ~13 hours (mostly insert time at the upper scales, not embedding).

---

*This specification is the input for building `benchmark/dataset.py` and `benchmark/vocabulary.py`. Every parameter above (distribution percentages, variant counts, review lengths, vocabulary pools) becomes a configurable constant in the code, overridable via the YAML benchmark config.*
