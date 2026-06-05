# SVR Documentation Index

## Directory Structure

```
docs/
  architecture/       Architecture deep-dives and subsystem documentation
  executive/          CTO/Eng Manager summaries and strategic documents
  phases/             Build prompts for each development phase (2-16)
  reviews/            Code review and audit reports
  roadmap/            Product roadmap and future plans
```

## Architecture Documentation

| File | Description |
|------|-------------|
| `architecture/ARCHITECTURE.md` | Full technical architecture reference (~87K) |
| `architecture/LIFECYCLE_ARCHITECTURE.md` | Lifecycle subsystem deep-dive (~26K) |

## Executive Summaries

| File | Description |
|------|-------------|
| `executive/EXECUTIVE_CTO_SUMMARY.md` | CTO-level technical overview |
| `executive/ENG_MANAGER_RECAP.md` | Engineering manager recap (Phases 1-15) |
| `executive/SVR_STRATEGIC_IDEAS.md` | Product strategy and market positioning |

## Roadmap

| File | Description |
|------|-------------|
| `roadmap/ROADMAP.md` | Full development roadmap |
| `roadmap/FUTURE_PHASES.md` | Planned future phases |
| `roadmap/MULTI_BACKEND_ROADMAP.md` | Multi-backend expansion plan |
| `roadmap/MULTI_BACKEND_REMARKS.md` | Backend abstraction notes |

## Phase Build Prompts

Prompts used to build each phase of the SDK (stored for reproducibility):

| File | Phase |
|------|-------|
| `phases/PHASE2_PROMPT.md` | Phase 2: Testing foundation |
| `phases/PHASE3_PROMPT.md` | Phase 3: Retry/backoff, resilience |
| `phases/PHASE4_PROMPT.md` | Phase 4: CLI commands |
| `phases/PHASE5_PROMPT.md` | Phase 5: MetadataStore, detection |
| `phases/PHASE6_PROMPT.md` | Phase 6: Logging, metrics, caching |
| `phases/PHASE7_PROMPT.md` | Phase 7: Ingestion pipeline |
| `phases/PHASE8_PROMPT.md` | Phase 8: Developer experience |
| `phases/PHASE9_PROMPT.md` | Phase 9: Testing excellence, mypy |
| `phases/PHASE10_PROMPT.md` | Phase 10: Centroid routing |
| `phases/PHASE11_PROMPT.md` | Phase 11: Job scheduler, events |
| `phases/PHASE11_5_PROMPT.md` | Phase 11.5: Structural refactoring |
| `phases/PHASE12_PROMPT.md` | Phase 12: Backend abstraction |
| `phases/PHASE12_5_PROMPT.md` | Phase 12.5: Backend interface |
| `phases/PHASE13_PROMPT.md` | Phase 13: PostgreSQL backend |
| `phases/PHASE14_PROMPT.md` | Phase 14: Audit hardening |
| `phases/PHASE14_5_PROMPT.md` | Phase 14.5: Upsert fixes |
| `phases/PHASE15_PROMPT.md` | Phase 15: Zero-config DX |
| `phases/PHASE16_EMBEDDING_TEXT_PROMPT.md` | Phase 16: Smart embedding text |

## Reviews & Audits

| File | Description |
|------|-------------|
| `reviews/PHASE12_REVIEW.md` | Phase 12 code review |
| `reviews/PHASE13_REVIEW.md` | Phase 13 code review |
| `reviews/PHASE13_FINAL_AUDIT.md` | Phase 13 final audit |

## Changelog

| File | Description |
|------|-------------|
| `CHANGELOG.md` | Version-by-version change log |
