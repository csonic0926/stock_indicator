# Universe Pipeline

How the system goes from **SEC `company_tickers_exchange.json`** to a stable,
FF12-classified **production-candidate universe** used for audited production
symbol additions.

Yahoo Finance price download is **out of scope** here — it consumes the active
production runtime contract, not the candidate contract directly. See
`daily_job.update_all_data_from_yf` for that.

---

## Scope

| In scope | Out of scope |
|---|---|
| SEC ticker + SIC ingestion | Yahoo price CSV download |
| Hard filter (suffix + obvious title) | yfinance per-symbol retry / rate limiting |
| LLM semantic classification | Trading signal / simulation |
| Policy override layer | Risk score gate |
| yf availability quarantine (consumes existing audit CSV) | Order placement |
| SEC exchange provenance + local price-history maturity gate | Historical price backfill |
| Runtime guardrail LLM audit | Adaptive TP/SL rolling state |
| FF12 assignment (SIC mapping + Yahoo secondary + LLM secondary) | |

---

## Runtime and candidate outputs

| File | Meaning |
|---|---|
| `data/symbols.txt` | Active production runtime symbol contract. Newline-separated. Runtime trusts this exactly. |
| `data/production_symbols.txt` | Explicit active production symbol contract used by production configs. |
| `data/production_symbols_with_sector.parquet` | Active production universe + `cik`, `sic`, `ff12`, `ff48`, `ff49`, `ff_label`, `sic_desc`, `ff12_source` columns. Used for sector-aware Pick-N. |
| `data/production_symbols_with_sector.csv` | Human-readable mirror of active production parquet. |
| `data/production_candidate_symbols.txt` | Audited candidate contract produced by `update_universe_pipeline`; this is the source for future production additions, not a direct runtime swap. |
| `data/production_candidate_symbols_with_sector.parquet` | Candidate universe with 100% FF12 coverage. `sync_production_ff12_sector` reads this when promoted symbols need sector rows. |
| `data/production_candidate_symbols_with_sector.csv` | Human-readable mirror of candidate parquet. |
| `data/symbol_tradability_gate_audit.csv` | Persistent audit of the exchange + maturity gate that blocks OTC / non-primary listings and IPO-ish symbols before final publish. |

Runtime rule: any symbol in `data/production_symbols.txt` / `data/symbols.txt`
**must** have a row in `data/production_symbols_with_sector.parquet` with a
non-null `ff12` (1-12). Candidate rule: any symbol in
`data/production_candidate_symbols.txt` **must** have a row in
`data/production_candidate_symbols_with_sector.parquet` with the same FF12
guarantee. Coverage is **100%** in both contracts.

---

## Pipeline stages (in order)

```
Stage 1   SEC fetch                         → cache + intermediate frame
Stage 2   Hard filter                       → deterministic non-common-stock removals
Stage 3   Second-layer select               → small LLM candidate subset
Stage 4   LLM semantic classify             → sticky semantic cache
Stage 5   Policy override                   → design-time mechanism decisions
Stage 6   yf availability                   → consume sticky price-source quarantine
Stage 7   Runtime guardrail LLM             → one-shot reconciliation cache
Stage 8   Tradability + maturity gate       → trusted SEC exchange + mature local history
Stage 9   Common-equity policy              → baby bonds, shells, fund-preferred removals
Stage 10  FF12 assignment                   → final symbols / sector rows = 100% coverage
```

### Stage 1 — SEC fetch

**Source**
- Universe + exchange provenance: <https://www.sec.gov/files/company_tickers_exchange.json>
- Legacy ticker/CIK fallback: <https://www.sec.gov/files/company_tickers.json>
- Per-CIK SIC: <https://data.sec.gov/submissions/CIK{padded}.json>

**Code**
- `src/stock_indicator/sector_pipeline/sec_api.py`
  - `fetch_company_ticker_exchange_table()` — downloads ticker/CIK/title/exchange for the production universe pipeline
  - `fetch_company_ticker_table()` — legacy ticker/CIK/title helper for callers that do not need exchange provenance
  - `fetch_submissions_json(cik)` — downloads per-CIK metadata, **uses local cache `cache/submissions/CIK{padded}.json`**
  - `extract_standard_industrial_classification(json)` — pulls `"sic"` field

**Cache policy**
- Submissions cache is **sticky** — once written, always re-used (`sec_api.py:96`)
- Freshness is delivered by re-running Stage 1 on a schedule, **not** by stale-checking the cache
- New IPOs appear in `company_tickers_exchange.json` automatically (SEC live), but the tradability gate keeps them out until local price history is mature enough

**Trigger**
- `update_sector_data` (manage shell) → `pipeline.build_sector_classification_dataset()`
- Or implicit via `pipeline.update_latest_dataset()` (uses `cache/last_run.json`)

**Intermediate frame** (in memory)
- Columns: `ticker`, `cik`, `sic`, `title`, `exchange`
- Cardinality: ~10,365 (varies with SEC updates)

---

### Stage 2 — Hard filter

**Purpose**: cheap deterministic exclusion of clearly non-common-stock instruments.

**Code**: `scripts/build_symbol_universe_final_audit.py` (or equivalent hard-filter script)

**Rules** (must all be unambiguous suffix or substring matches; no false-positive substring match):

| Rule | Match | Examples |
|---|---|---|
| Warrant / unit / right / preferred suffix | `.WS` / `.WT` / `.W` / `.U` / `.R` / `.P` with delimiter | `ACHR.WT`, `ABR.PD` |
| Nasdaq-style suffix | trailing `U/W/R/P/WS/WT/RT` **with delimiter** (not bare substring) | `AACBU`, `AACBR` |
| Title contains | `ETF / ETN / EXCHANGE-TRADED / WARRANT / UNIT / RIGHT / PREFERRED / NOTE / BOND / DEBENTURE / TREASURY` | iShares Bitcoin Trust ETF |
| Title is ETF-like trust | `SPDR GOLD TRUST` / `INVESCO QQQ TRUST` / physical metal trusts | SPY, QQQ, GLD |

**Output**
- `data/symbol_hard_filter_audit.csv` — columns: `symbol`, `sec_title`, `hard_filter_decision`, `hard_filter_reason`
- `data/symbols_hard_filtered_from_sec.txt` — hard-include list

**Important: strict delimiter for suffix**
- Don't match bare `WS`/`WT` substring — that kills `CWT` (California Water), `FLWS` (1-800-Flowers), `CRWS` (Crown Crafts).
- Old `NON_COMMON_STOCK_SYMBOL_PATTERN` had this bug — see `Architectural Notes` below.

**Numbers** (2026-05-22 run)
- include: 8,874
- exclude: 1,491
  - nasdaq_non_common_suffix: 863
  - non_common_symbol_suffix: 486
  - obvious_non_common_title: 83
  - obvious_etf_like_trust_title: 58
  - missing_symbol: 1

---

### Stage 3 — Second-layer candidate selection

**Purpose**: narrow LLM scope. Most of the 8,874 hard-includes are obvious common equity (AAPL, JPM, XOM). Only ambiguous names need semantic judgment.

**Code**: `scripts/classify_symbol_second_layer_with_llm.py` (selection logic before LLM call)

**Trigger keywords** in `sec_title` or symbol:
- `ACQUISITION CORP` → SPAC
- `FUND` (plain) → potential investment vehicle
- `TRUST` (plain) → may be REIT / investment trust / royalty / muni
- `BDC` → Business Development Company
- `ROYALTY TRUST`
- `MUNICIPAL` / `CLO`
- `NOTE` / debt abbreviation

**Output**
- `data/symbol_second_layer_candidate_audit.csv` — columns: `symbol`, `sec_title`, `hard_filter_decision`, `second_layer_reasons`

**Numbers**
- LLM candidates: 793 (~9% of hard-include)
- Clean pass (LLM-skip): 8,081 (~91%)
- Token saving: ~91% reduction vs. classifying all

**Conservative principle**: only flag on **unambiguous keyword pattern**. If in doubt, send to LLM. Names that have institutional broadcast nature (banks like NTRS = Northern Trust Corp) intentionally fall into LLM scope for disambiguation.

---

### Stage 4 — LLM semantic classification

**Purpose**: judge whether each candidate represents an operating-company / REIT / BDC / etc. equity footprint, OR a basket / passive vehicle / SPAC shell / derivative.

**Code**: `scripts/classify_symbol_second_layer_with_llm.py`

**Prompt**: `data/symbol_universe_llm_prompt.md` — kept versioned for audit.

**Model**: production prompt cached in repo; provider-agnostic but production runs use Claude Sonnet / equivalent. Use `temperature=0` for determinism.

**Input per symbol**: `symbol`, `sec_title`, `sic` (if available), `ff12_label` (if available)

**Output schema**
```
symbol, sec_title, decision, semantic_type, confidence, reason
```
- `decision`: `include` / `exclude` / `quarantine`
- `semantic_type`: enumerated tag (`reit_common_equity`, `bdc_common_equity`, `commodity_trust`, `closed_end_fund`, `royalty_trust`, `operating_company_common_equity`, etc.)
- `confidence`: `high` / `medium` / `low`
- `reason`: free-text justification

**Quarantine policy (fail-safe)**
- `quarantine` results stay out of `symbols.txt` (not traded). Quarantine is the safe direction — better to miss than to mis-include.
- Low-confidence include can be promoted to quarantine via policy override.

**Output files**
- `data/symbol_universe_llm_classification.csv` — per-symbol decisions
- `data/symbol_universe_llm_classification_batches.jsonl` — raw batch I/O for audit

**Sticky caching (for cron)**
- An existing symbol's classification is **never re-classified** unless `sec_title` changes or manual flag is set.
- Only **new tickers** since last run are sent to LLM (diff-only).
- This makes daily cron LLM cost trivial (~handful of new IPOs/day).

**Numbers**
- LLM input: 793
- include: 122
- exclude: 670
- quarantine: 1

---

### Stage 5 — Policy override

**Purpose**: Mechanism-justified overrides on top of LLM judgment. These are **design-time decisions**, not runtime telemetry.

**File**: `data/symbol_universe_policy_overrides.csv`

**Schema**
```
match_field, match_value, override_decision, override_semantic_type, override_reason
```

**Two pattern types**
- **Pattern-based** (`match_field=semantic_type`) — covers a whole class
  - `semantic_type=royalty_trust → exclude` (commodity pass-through, not operating equity)
  - `semantic_type=bdc_common_equity → exclude` (NAV / yield / credit-portfolio driven)
- **Symbol-specific** (`match_field=symbol`) — individual exceptions
  - `symbol=BXDC → exclude` (Blackstone Private Credit fund variant masquerading as Trust)

**Auditable**: each override records its `reason`. Future revisits can re-read these and reconsider.

**Numbers** (current policy)
- royalty trust: 15 excluded
- BDC / private credit: 13 excluded
- BXDC: 1 excluded
- Plus 46 added 2026-05-23 (see Stage 9): baby_bond_or_note (36), shell_company (9), fund_or_preferred_security (1)

**Why a separate layer**: LLM prompt may evolve, but mechanism-grounded exclusions (royalty trust = commodity pass-through) are stable design decisions. Keeping them out of the prompt itself avoids prompt churn.

---

### Stage 6 — yf availability quarantine (consumes existing audit, does NOT download)

**Purpose**: a symbol can pass Stage 1-5 (looks like real equity) but Yahoo has no usable price for it (delisted, suspended, ticker mismatch). Don't trade what we can't price.

**Code**: `scripts/build_symbol_universe_final_audit.py` (audit reader; download happens in `daily_job.update_all_data_from_yf`)

**Input** (this pipeline does NOT trigger download)
- `data/yf_update_audit_*.csv` — produced by an earlier `update_all_data_from_yf` run
- `data/symbols_price_source_quarantine.csv` — sticky quarantine list

**Rules**
| Audit class | Quarantine? | Rationale |
|---|---|---|
| Empty CSV (no data ever returned) | Yes — immediate | Cannot price → cannot trade |
| Stale CSV (≥5 trading days behind) | Yes — sticky | Likely delisted / suspended |
| Stale 1-3 days | No | Transient (weekend, corporate action) |
| YF failure log (transient) | No | Retry recovers; only if persistent → escalate to stale/empty |

**Sticky + N-strike rule**
- Sticky: once quarantined, stays quarantined until manual audit or N consecutive successful refreshes
- N-strike for entering quarantine: needs N consecutive days of empty/stale, not single-day flicker
- Protects sim reproducibility — universe doesn't fluctuate day-to-day

**Numbers**
- 537 quarantined (146 empty + 56 stale + others)
- Universe after this stage: 7,637

---

### Stage 7 — Runtime guardrail LLM audit (one-shot reconciliation)

**Purpose**: history left behind 4 layers of runtime guardrails (`daily_job.load_runtime_download_symbols`) that conflicted with the new pipeline. This stage absorbs those conflicts via LLM review so the runtime guardrails can be cleanly disabled.

**Inputs (the 4 old guardrails)**
1. `strategy.load_symbols_excluded_by_industry` — SIC 6221 (commodity ETF) + 6770 (SPAC) + sector_overrides.csv (2 entries: TEST + TLT)
2. `daily_job.load_symbols_rejected_by_asset_metadata` — Alpha Vantage ETF list
3. `daily_job.load_symbols_rejected_by_listing_name` — listing-name regex (`warrants?|units?|rights?|preferred|preference`)
4. `daily_job.NON_COMMON_STOCK_SYMBOL_PATTERN` — symbol-suffix regex (HAS BUG, see Architectural Notes)

**Process**
- Find symbols that would be rejected by any of the 4 old guardrails but pass Stage 1-6
- Send each to LLM with full context (sec_title, sic, why old guardrail flagged it, new pipeline decision)
- LLM emits `include` / `exclude` / `quarantine`
- Decisions written to `symbols.txt` and audit CSVs; old guardrails are then disabled in `daily_job.py` + `strategy.py`

**One-shot vs. recurring**
- This stage was a **one-shot reconciliation** during the 2026-05-22 universe refactor
- After old guardrails are disabled, this stage is no-op on subsequent cron runs (no conflicts to reconcile)
- Audit preserved for traceability

**Files**
- `data/runtime_guardrail_conflict_llm_audit.csv` — full LLM review
- `data/runtime_guardrail_conflict_llm_prompt.md` — prompt
- `data/symbols_removed_by_runtime_guardrail_llm.csv` — what was removed
- `data/runtime_universe_contract_after_guardrail_cleanup_audit.csv` — post-cleanup contract

**Numbers**
- 198 conflicts reviewed
- 129 exclude + 15 quarantine + 54 include
- Universe after this stage: 7,493

---

### Stage 8 — Tradability provenance + maturity gate

**Purpose**: make the production-candidate contract enforce what the historical
production baseline used to imply: a symbol must not be admitted just because
it appears in SEC reporting data. It must also look like a primary-exchange
tradable listing with enough local daily history for the strategy mechanism to
be meaningful.

**Code**: `src/stock_indicator/universe_pipeline.py`
- `apply_tradability_gate(...)`
- `load_price_history_metadata_for_symbols(...)`
- `fetch_company_ticker_exchange_table()` in `sector_pipeline/sec_api.py`

**Inputs**
- SEC exchange field from `company_tickers_exchange.json`
- `data/stock_data_2010_yf_clean_prepare_audit.csv`
- Fallback: `data/stock_data_2010_yf_clean/{SYMBOL}.csv`

**Rules**
| Gate | Rule | Fail-closed reason |
|---|---|---|
| Exchange provenance | SEC exchange must be one of `Nasdaq`, `NYSE`, `NYSE American`, `NYSE Arca`, `Cboe BZX` | `untrusted_or_missing_sec_exchange` |
| Local history exists | Symbol must have local long-history metadata or a readable long-history CSV | `missing_price_history` |
| Minimum rows | At least 252 daily rows | `price_history_rows_below_252` |
| Minimum age | First-to-last local history span at least 365 calendar days | `price_history_age_below_365_days` |

**Manual includes**
- `decision_source=policy_override` bypasses this automated gate.
- This is intentional for explicit audited policy exceptions; those symbols remain auditable as `tradability_status=manual_include`.

**Output**
- `data/symbol_tradability_gate_audit.csv`
- `data/symbol_universe_final_audit.csv` also carries `sec_exchange`, `tradability_status`, `tradability_reason`, `price_history_rows`, `price_history_first_date`, and `price_history_last_date`.

**Cron behavior**
- New SEC tickers now fail closed until they have trusted exchange provenance and mature local price history.
- This is the key difference from the rejected 7.4K broad SEC universe: SEC-reporting status alone is no longer enough.

---

### Stage 9 — Common-equity-only policy (mechanism check on missing-SIC LLM output)

**Purpose**: When Stage 10's `secondary_llm` LLM review filled missing-SIC FF12 assignments, some flagged "tradeable but not common equity" (baby bonds, shell companies, preferred-as-fund). These violate the strategy mechanism (institutional broadcast → retail stereotyped reaction).

**Decision**: exclude from universe. Same policy as royalty trust / BDC — mechanism mismatch ≠ tradeable in this system.

**Implementation**: 46 entries added to `data/symbol_universe_policy_overrides.csv`.

**Output**
- Historical 2026-05-23 cleanup: `symbols.txt` 7,493 → 7,447 before the later tradability gate was added.

---

### Stage 10 — FF12 assignment

**Purpose**: assign every symbol in `symbols.txt` to a Kenneth French 12-industry group (1-11 or 12=Other).

**Three FF12 sources, applied in order**

| Order | Source | Code | Applies when |
|---|---|---|---|
| 1 | `sic_mapping` | `sector_pipeline/ff_mapping.py:attach_fama_french_groups` | SEC SIC exists AND is in `data/sic_to_ff.csv` (49 ranges) |
| 2 | `secondary_yfinance` | `sector_pipeline/secondary_classification.build_secondary_classifications_from_yfinance_metadata` | SEC SIC missing; Yahoo `info.sector` available |
| 3 | `secondary_llm` | `sector_pipeline/secondary_classification.load_secondary_classifications` | SEC SIC missing AND Yahoo also missing; LLM judges from `sec_title` + context |

**Fallback**: any symbol not covered → `ff12=12` (Other), `ff_label='Other'` (Kenneth French canonical fallback in `attach_fama_french_groups`).

After Stage 9, the pipeline guarantees **0 fallback** — every symbol in `symbols.txt` has a non-fallback FF12 assignment.

**SIC → FF12 mapping table** (`data/sic_to_ff.csv`)
- 49 rows covering 5,450 unique 4-digit SIC codes
- Covers Kenneth French canonical FF12=1 to FF12=11 ranges
- **FF12=12 ("Other") has no explicit row** — it is the merge `fillna(12)` default
- Authoritative source: Kenneth French Data Library, <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

**FF12 source breakdown** (historical 2026-05-23 7,447 universe before the later tradability gate)
```
sic_mapping           5,584   (SEC SIC + sic_to_ff.csv match)
sic_unmapped_other    1,195   (SEC SIC present, falls into Kenneth French Other)
secondary_yfinance      620   (SEC no SIC; Yahoo metadata supplied)
secondary_llm            48   (Both SEC and Yahoo missing; LLM judged)
─────────────────────────────
total                 7,447
```

**Output**
- `data/symbols_with_sector.parquet` — added column `ff12_source` records which of the three sources assigned the FF12
- `data/symbols_with_sector.csv` — human-readable mirror

**Secondary classification files**
- `data/sector_secondary_classifications.csv` — Yahoo metadata-derived assignments
- `data/sector_missing_sic_llm_classification.csv` — full LLM decisions for missing-SIC symbols
- `data/sector_missing_sic_llm_quarantine.csv` — LLM's exclude/quarantine subset (feeds Stage 9 overrides)
- `data/sector_missing_sic_remaining_after_llm.csv` — symbols still unresolved (currently 0)

---

## Audit trail map

| File | Stage | Purpose |
|---|---|---|
| `symbol_hard_filter_audit.csv` | 2 | Per-symbol hard-filter decision + reason |
| `symbols_hard_filtered_from_sec.txt` | 2 | Hard-include list |
| `symbol_second_layer_candidate_audit.csv` | 3 | Which symbols go to LLM and why |
| `symbol_universe_llm_classification.csv` | 4 | LLM per-symbol decisions |
| `symbol_universe_llm_prompt.md` | 4 | Prompt version used |
| `symbol_universe_llm_classification_batches.jsonl` | 4 | Raw LLM I/O |
| `symbol_universe_policy_overrides.csv` | 5, 9 | Design-time mechanism overrides |
| `symbol_universe_hard_plus_llm_audit.csv` | 4-7 merged | Combined hard + LLM + policy + guardrail decisions before price/tradability gates |
| `yf_update_audit_*.csv` | 6 (input) | yf refresh telemetry |
| `symbols_price_source_quarantine.csv` | 6 | Sticky yf quarantine |
| `symbols_hard_plus_llm_from_sec.txt` | 7 output | Symbols before price-source quarantine and tradability gates |
| `runtime_guardrail_conflict_llm_audit.csv` | 7 | One-shot guardrail reconciliation |
| `runtime_guardrail_conflict_llm_prompt.md` | 7 | Prompt |
| `symbols_removed_by_runtime_guardrail_llm.csv` | 7 | What removed |
| `runtime_universe_contract_after_guardrail_cleanup_audit.csv` | 7 | Post-cleanup contract |
| `symbol_tradability_gate_audit.csv` | 8 | Exchange provenance + local history maturity decisions |
| `sector_secondary_classifications.csv` | 10 | Yahoo metadata FF12 |
| `sector_missing_sic_llm_classification.csv` | 10 | LLM missing-SIC FF12 |
| `sector_missing_sic_llm_quarantine.csv` | 10 → 9 | LLM exclude (feeds policy override) |
| `sector_missing_sic_remaining_after_llm.csv` | 10 | Still unresolved (0 currently) |
| `symbols.txt` | Final | Universe contract |
| `symbols_with_sector.parquet` / `.csv` | Final | Universe + FF12 |
| `symbols_price_source_usable.txt` | Final | Mirror of symbols.txt (yf usability already enforced) |

---

## Architectural notes

### Decoupling: stock eligibility ≠ sector classification

- `symbols.txt` is the **single source of truth for stock eligibility**
- FF12 is a **sector pick aid** (Pick-N balance), NOT a universe gate
- Old code conflated these — FF12=12 silent drop was an eligibility gate disguised as a sector classifier
- Fixed in `strategy.py:load_ff12_groups_by_symbol` (now retains group 12) and `strategy.py:_build_eligibility_mask` (group_id=12 is valid)

### Runtime guardrails (in `daily_job.py`) are disabled

After Stage 7's one-shot reconciliation:
- `load_symbols_excluded_by_industry()` returns empty set (legacy no-op)
- `symbols_rejected_by_asset_metadata` — AlphaVantage ETF list no longer applied
- `symbols_rejected_by_listing_name` — listing-name regex no longer applied
- `NON_COMMON_STOCK_SYMBOL_PATTERN` — symbol-suffix regex no longer applied

Runtime trusts `symbols.txt` exactly, requires only that FF12 row exists.

### Bug fixed: `NON_COMMON_STOCK_SYMBOL_PATTERN`

```python
# OLD (buggy):
r"(?:[.-](?:WT|WS|W|U|R)|(?:WS|WT))$"
# Second alternation matches bare WS/WT suffix without delimiter → kills
# CWT (California Water), FLWS (1-800-Flowers), CRWS (Crown Crafts)
```

Hard filter (Stage 2) uses strict delimiter-only suffix matching to avoid this class of false positive.

### Pandas NA parsing bug fixed

Audit scripts must read CSVs with `keep_default_na=False`. Without it, ticker `NA` (a real symbol — Nano Labs) is silently read as blank, polluting downstream audits.

### Open issue: Problem B (sic_unmapped_other = 1,195)

Kenneth French taxonomy assigns Mines/Constr/Trans/Hotels/Bus Serv/Entertainment to FF12=12 by design. In the historical 2026-05-23 7,447 run, **1,195 valid common equities remained in FF12=12**:
- MAR / HLT / MGM / H / LVS / CZR / BYD (Hotels)
- AAL / DAL / UAL / LUV / JBLU / ALK (Airlines)
- AEM / B / FNV / AU (Gold/Metal Mining)
- DHI / BZH (Builders)
- DIS / DKNG (Entertainment)
- CCL / CUK (Cruise)

Impact: Pick-N sector balance in FF12=12 has 1,195 symbols across 6 industries competing for the same N slots. A single industry's bull cycle (e.g. gold mining 1996/2011) can monopolize the group's slots.

**Not a current blocker** (universe contract is correct), but a future fairness issue. Three resolution paths:
- **(A) FF49** — Kenneth French's 49-industry finer taxonomy; would split Mines/Hotels/Airlines into own groups. Requires `sic_to_ff.csv` ff48/ff49 columns to be populated. Pick-N might need recalibration (49 groups too granular for current Pick value).
- **(B) GICS 11 sectors** — modern finance standard. Requires separate data source (S&P / Bloomberg / FMP).
- **(C) Custom 16-group** — extend FF12 with Mines / Hospitality / Transport / Construction / etc. Flexible but self-maintained.

---

## Re-run conditions

| Stage | Re-run when |
|---|---|
| 1 SEC fetch | Daily (cheap) — pulls new IPOs, ticker changes |
| 2 Hard filter | Whenever Stage 1 has new symbols |
| 3 Second-layer select | Whenever Stage 2 has new symbols |
| 4 LLM classify | **Diff-only**: only new symbols since last run. Sticky cache for existing. |
| 5 Policy override | Stateless — re-applied on every build |
| 6 yf availability | Consumes whatever yf audit CSV is current; sticky quarantine |
| 7 Guardrail audit | One-shot 2026-05-22; no-op going forward unless guardrails re-enabled |
| 8 Tradability + maturity gate | Every run; uses SEC exchange provenance and local long-history metadata |
| 9 Common-equity policy | Stateless (overrides file) |
| 10 FF12 assignment | Re-run when new SIC data, new Yahoo metadata, or new LLM decisions |

---

## Cron design

### Order

Production cron consumes the active production contracts only. The candidate
pipeline is intentionally not part of `run_daily_job.sh`, because candidate
refreshes must never rewrite active production files during the daily price or
signal window.

```bash
# === Phase A: Price data (consumes active production symbols.txt) ===
update_all_data_from_yf               # download per symbols.txt

# === Phase B: Signal generation ===
multi_bucket_daily_signal data/multi_bucket_production.json
```

### Frequency recommendation

| Pipeline | Suggested frequency | Why |
|---|---|---|
| Production daily cache | Daily | Uses active `data/symbols.txt` / `data/production_symbols.txt` |
| Production daily signal | Daily, after price refresh | Uses `data/multi_bucket_production.json` |
| Production-candidate refresh | Manual / controlled cron only | Publishes `production_candidate_*` staging outputs and must not change active production files |

Heavy SEC submissions API calls are cached, so daily re-runs only fetch new CIKs (typically a handful per day).

### Atomic swap discipline

- Never edit production `symbols.txt` / production sector parquet in place.
- Build to temp files, then replace with same-filesystem atomic swaps.
- `update_universe_pipeline` writes only the production-candidate contract:
  `production_candidate_symbols.txt` and `production_candidate_symbols_with_sector.*`.
- `sync_production_ff12_sector` writes active production sector outputs
  (`production_symbols_with_sector.*`) and runtime mirrors (`symbols.txt` plus
  `symbols_with_sector.*`) from the audited production contract.

### Pre-swap validation criteria

The candidate pipeline validates the candidate contract before any atomic replace:

| Check | Rule | Why |
|---|---|---|
| Symbol count drift | Proposed candidate contract must not drop by more than 5% versus current `production_candidate_symbols.txt` | Catches half-broken SEC responses that return only a partial ticker table |
| Tradability gate | Included symbols must pass trusted SEC exchange + mature local history, unless manually policy-included | Blocks SEC-reporting OTC/foreign/IPO-ish noise from entering candidate output |
| Sector row count | `production_candidate_symbols_with_sector` ticker set must exactly equal `production_candidate_symbols.txt` | Prevents promoted symbols with no FF12 row |
| FF12 validity | Every row must have `ff12` in 1-12 | Pick-N grouping needs a valid bucket |
| Confidence | No `classification_confidence=low` | Low confidence means unresolved missing-SIC fallback |
| Source fallback | No `ff12_source=missing_sic_fallback` | Every symbol must be classified by SIC, Yahoo secondary, or LLM secondary |

The 5% drop guard is asymmetric: large additions are allowed (e.g. many IPOs);
large removals require manual audit. For controlled one-time migrations, run
with an explicit larger threshold such as `--maximum-drop-ratio 0.50`; after
that, return to the default 5% guard because normal daily churn should be tiny.

### Failure handling

| Failure | Behavior |
|---|---|
| SEC API down / timeout | Candidate refresh aborts; active production remains unchanged |
| LLM API down / rate limit | Stage 4 aborts; existing classifications retained; new symbols stay out until next run |
| yf cumulative failure | Production daily cache refresh is partial; signal generation uses whatever CSVs exist |
| Validation / parquet rebuild fail | Candidate refresh aborts; existing candidate files remain unchanged |

**Fail-closed for new symbols**: if a new ticker cannot be classified, keep it
OUT of `production_candidate_symbols.txt` until classified. Do not default-include
new noise.

### Cron implementation status

- `run_daily_job.sh` does not call `update_universe_pipeline`.
- The candidate refresh command is `python -m stock_indicator.manage update_universe_pipeline`.
- Controlled refreshes may use `python -m stock_indicator.manage update_universe_pipeline --dry-run --maximum-drop-ratio 0.50`.
- Last `update_sector_data` run: see `cache/submissions/*.json` mtimes; latest = 2026-04-08.

---

## Official production add-symbol policy

The production rule is **stable baseline plus audited additions**:

1. Existing symbols and FF12 rows in `data/production_symbols.txt` and
   `data/production_symbols_with_sector.*` stay frozen unless deliberately
   edited.
2. New SEC symbols are processed by `update_universe_pipeline` into the
   production-candidate contract.
3. A new symbol may enter the candidate contract only after it passes:
   hard filter, second-layer LLM/policy review, price-source quarantine,
   trusted SEC exchange provenance, local price-history maturity, and FF12
   classification validation.
4. The maturity gate requires `price_history_rows >= 252` and at least 365
   calendar days between first and last local history dates. The code excludes
   `row_count < 252`, so exactly 252 rows satisfies the row-count test.
5. To promote a symbol, add it to `data/production_symbols.txt`, then run
   `python -m stock_indicator.manage sync_production_ff12_sector --dry-run`.
   If the diff is correct, run the same command without `--dry-run`; the sync
   also refreshes `data/symbols.txt` and `data/symbols_with_sector.*` runtime
   mirrors from the production contract.
6. `sync_production_ff12_sector` preserves existing production FF12 rows and
   appends the promoted symbol's audited row from
   `data/production_candidate_symbols_with_sector.*`.
7. A promoted symbol still cannot fire an entry until it has an audited row in
   `data/production_symbol_eligibility.csv` and the trade date is on or after
   `first_eligible_trade_date`.
8. Baseline production symbols are eligible from `2010-01-01`; missing
   eligibility rows fail closed and block entries.
9. The default new-symbol seasoning helper is promotion date + 365 calendar
   days. This is separate from the 252-row candidate maturity gate; the system
   does **not** interpret 252 bars after promotion as the live-entry seasoning
   rule.

## Sources

- SEC company ticker exchange provenance: <https://www.sec.gov/files/company_tickers_exchange.json>
- SEC legacy company tickers: <https://www.sec.gov/files/company_tickers.json>
- SEC submissions: <https://data.sec.gov/submissions/CIK{padded}.json>
- Kenneth French FF industry definitions: <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>
- Local Yahoo metadata audit: `data/yfinance_metadata_for_missing_sic.csv`
- LLM prompts: `data/symbol_universe_llm_prompt.md`, `data/runtime_guardrail_conflict_llm_prompt.md`
