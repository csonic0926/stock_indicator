# Runtime Guardrail Conflict Classification Prompt

You classify US-listed symbols for a trading strategy universe.

Context: the current `symbols.txt` universe came from SEC company_tickers.json plus deterministic and LLM filters. A later runtime layer still rejected these symbols using older guardrails. Some old guardrails are correct, but some are stale, over-broad, or affected by ticker reuse. Your job is to decide whether each symbol should remain in the strategy's stock universe.

## Strategy scope

Eligible symbols are traded equity for an individual company-like issuer whose own shares can show institutional accumulation/distribution. Include ordinary common stock, ADR/common equity, operating companies, banks, insurers, miners, hotels, airlines, builders, software companies, REITs, mortgage REITs, and similar operating-company-like listed equity.

Exclude packaged investment vehicles, derivatives, financing instruments, and shells where the ticker price mainly represents a basket/index/fund portfolio, bond/note claim, preferred distribution claim, warrant value, commodity passthrough, royalty passthrough, BDC/private-credit portfolio, or pre-merger SPAC rather than operating/company-like equity.

## Evidence interpretation

- `sec_title` is the issuer name from SEC company_tickers.json. It may omit security class details for notes/preferred/ETNs.
- `yf_quote_type` and `yf_long_name` are current Yahoo Finance metadata. If Yahoo says ETF/ETN, treat that as strong exclude evidence.
- `latest_listing_name` comes from older AlphaVantage listing snapshots and can be stale because tickers are reused. Do not blindly exclude an operating company only because an old listing name was an ETF/fund with the same ticker.
- `sic` and old runtime reasons are conflict evidence, not final authority. SIC 6770 often means SPAC/blank-check, but some post-merger or operating issuers may retain stale SIC. SIC 6221 is not enough by itself to exclude.
- The heuristic recommendation is only an audit hint; override it when the evidence says otherwise.

## Decision labels

Use exactly one decision per symbol:

- `include`: eligible for the strategy universe.
- `exclude`: should not enter the strategy universe.
- `quarantine`: evidence is insufficient or genuinely ambiguous; do not include until reviewed.

Use quarantine when current evidence conflicts and you cannot tell whether the traded security is common equity.

## Exclude guidance

Classify as `exclude` when evidence indicates:

- ETF, ETN, exchange-traded product, index tracker, leveraged/inverse product;
- mutual fund, closed-end fund, income fund, municipal fund, target term trust, credit fund, CLO fund, or portfolio fund;
- pre-merger SPAC, blank-check, or acquisition shell;
- preferred share, depositary preferred, note, bond, debenture, structured product, warrant, right, or unit;
- commodity/physical-asset/royalty pass-through;
- BDC, business-development company, private-credit fund, direct-lending fund, or specialty-lending fund.

## Output schema

Return JSON only. Do not include markdown.

For each input record, return one output record with:

- `symbol`: same as input symbol.
- `decision`: one of `include`, `exclude`, `quarantine`.
- `semantic_type`: short snake_case type, for example `operating_company_common_equity`, `reit_common_equity`, `exchange_traded_note`, `closed_end_fund`, `spac_shell`, `preferred_security`, `note_or_debt`, `commodity_or_royalty_pass_through`, `private_credit_or_bdc_vehicle`, `ambiguous`.
- `confidence`: one of `high`, `medium`, `low`.
- `reason`: concise natural-language reason, 20 words or fewer.

Classify the provided records now.
