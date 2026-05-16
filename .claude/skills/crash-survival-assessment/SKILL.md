---
name: crash-survival-assessment
description: 評估某個股災情境（歷史、假想、或當前）下，本 stock_indicator 系統 (fish_head_vacuum_turn + fish_tail_blow_off_top + fish_head_b30_35 三 bucket) 能否生還。透過比對歷史股災 signature、定位系統 structural failure mode、查當前 macro 狀態 (含 Fed 干預能力)，輸出生還機率與關鍵 trigger 訊號。Cal 想知道「這次/下次/某種 X 形態股災會不會打掛系統」時用。
---

# Crash Survival Assessment for stock_indicator

## When to use this skill

Cal asks any of:
- 「這個系統能撐過 X 股災嗎」
- 「現在 macro 危險嗎、系統會死嗎」
- 「如果 [scenario] 發生會怎樣」
- 「[某歷史/假想] 股災下系統表現預估」

Do NOT use for:
- Tactical config tuning (那是另一條 thread)
- Trade-level debugging
- Bug fixes

## ⚠️ Mandatory: use web search tools

**You MUST use a web-search-class tool (WebSearch, WebFetch, or equivalent) to gather current macro state before answering.**

Reasons this is non-negotiable:
- Knowledge cutoff makes prior-knowledge-only answers stale and dangerous
- Cal's question is about a moving target (current AI bubble state, current Fed posture, current oil price, current bank exposures change weekly)
- The 6-signature checklist requires fresh data points (rates, inflation prints, FSB warnings, default rates, CRE delinquency, etc.)
- Historical anchors in this SKILL are static, but the **mapping** from current state to those anchors REQUIRES fresh web evidence

If web search tool is unavailable in current environment:
- Tell Cal explicitly: "I cannot run this assessment without web search — historical anchors alone don't capture the time-sensitive state needed."
- Do NOT fabricate or estimate from training data. Refuse and surface the constraint.

If web search returns insufficient data on a critical dimension (e.g., can't find current Fed inflation print):
- Surface the gap in the output, don't paper over it.
- Lower confidence band accordingly.

## Critical context (must understand before evaluation)

### System architecture & failure dependency

System runs 3 buckets concurrently:
- **fish_head_vacuum_turn (fh)**: detects "retail panic 賣 + CTA forced sell + 機構 cover-buy 第一波 + retail FOMO 跟"
  - Edge depends on: **institutional cover-buy 第一波 materializes within ~weeks of vacuum_turn signal**
  - When 機構 doesn't come back fast → fh 持續 detect pattern but cover-buy 不來 → catastrophic loss
- **fish_tail_blow_off_top (ft)**: detects "A-peak distribution + retail FOMO surge"
  - Less exposed to liquidity crises (blow-off rarely fires in deflationary spiral)
- **fish_head_b30_35**: fh-family mid-band detector, same failure mode as fh

**fh 是 canary**. ft + b30_35 partial offset, but fh -40%+ year overwhelms portfolio.

### The structural failure mode (concentrate here)

System fails when **institutional bid is absent for sustained period (>>weeks, into months)** because:
1. vacuum_turn pattern keeps visually matching (CTA forced sell ✓, retail panic ✓)
2. But 第二層假設 (institutional cover-buy materializes) **doesn't fire**
3. Position enters → no rebound → slides into max_hold cut → typically -25% to -50% per trade
4. fh entries fire at 0.5/day rate during stress → universe-wide concurrent damage
5. Signal exit is **trend-reversal cross-down detector**, NOT falling-knife exit. Cannot save losing position in sustained downtrend (single-bar event, gets swallowed by min_hold gate).

**Therefore the question collapses to**: "Will institutional bid be absent for how long?"

### The key macro variable: Fed intervention capacity

機構 absence duration is driven by **how fast Fed acts as market maker of last resort**.
- Fast Fed → 機構 quickly return → V-shape → fh survives (often profits on rebound capture)
- Slow Fed → 機構 stay sidelined → sustained deleveraging → fh dies

Fed's speed is constrained by:
- **Rate space**: how much room to cut
- **Balance sheet space**: how much QE room before market loses faith in dollar
- **Inflation constraint**: cutting/QE while inflation >3-4% is politically explosive
- **Counterparty / banking sector health**: if banks themselves are the patient, Fed has more friction

## Historical crash database

| Crash | Trigger | Mechanism | Fed response speed | fh result |
|---|---|---|---|---|
| **2008 GFC** | Subprime/Lehman | **Repo market freeze + leveraged institutions FORCED to sell, can't access funding to buy back** | Slow (TARP 10/3, multiple rounds through 2009) | **-37% to -50%** (catastrophic, only failure case) |
| **2020 COVID** | Pandemic | Dealer inventory crash, institutions briefly withdraw | **Very fast** (PDCF/SMCCF within days, "market maker of last resort") | **+24%** (caught V-shape rebound) |
| **2018 Q4** | Fed hike scare + trade war | Brief liquidity scare, mechanical | Fast (Powell pivot January) | -0.77% (flat) |
| **2011 Eurozone** | Greek/PIIGS debt fears | Sector-specific (European banks) | N/A (Fed not needed for US) | **+17.62%** |
| **2015** | China devaluation | Brief vol spike | Fast | **+20.92%** |
| **2022** | Inflation/Fed hike | Slow grind, sector rotation | N/A (deliberate hike cycle) | **+23.62%** |
| **2002 (dot-com tail)** | Bubble unwind | Slow grind, sector rotation (tech→energy/staples) | N/A | **+23.67%** |
| **2001 (dot-com)** | Bubble peak | Equity bubble pop, no banking crisis | Fed cut aggressively | -10.45% (mild) |
| **1996** | EM crises | Minor | N/A | -7.57% (mild) |
| **1994** | Fed hike shock | Bond rout | N/A | -6.47% (mild) |

(All numbers: 1994_clean baseline sim, fh bucket only, max_hold=14, exit_alpha_factor=3, no reset_hold_on_reentry. Total portfolio fh+ft+b30_35 numbers will differ — typically fh dominates loss in bad years.)

**Pattern**: 唯一 catastrophic = 2008. Differentiator = Fed response speed × banking-sector-as-patient.

### What macro signatures predict 2008-class crash

| Signature | 2008 had it | Lookalike scenarios |
|---|---|---|
| Leveraged bubble in core financial sector | ✓ (MBS, CDS) | Real estate booms, private credit booms |
| **Banks ARE the patient** (not just providers) | ✓ | When banks own the bubble assets |
| Counterparty opacity | ✓ (CDS network) | Shadow banking, private credit opacity |
| **Fed capacity constrained** | partially (rate cuts effective but slow) | **Inflation-constrained Fed** (stagflation) |
| Slow visible unwind (months not days) | ✓ | Maturity walls, mark-to-market lags |
| No sector rotation possible | ✓ (broad) | Bubble spans market vs concentrated |

A crash with ALL 6 signatures = 2008-class = system likely fails.
A crash with 0-2 signatures = like 2018 / 2011 / 2022 = system thrives.
A crash with 3-4 signatures = uncertain; depends on speed of escalation.

## Evaluation workflow

### Step 1 — Identify the candidate scenario

Pull Cal's stated scenario (or "current macro"). Get specifics:
- What's the bubble / overheating asset?
- What sector is leveraged into it?
- What banking exposure exists?

### Step 2 — Web search for current state (MANDATORY, see top-of-file directive)

Run WebSearch (or WebFetch, or equivalent web-search-class tool) **in parallel** covering at minimum:

```
Query 1: "<bubble asset> 2026 valuations capex debt overheating"
Query 2: "<commodity/macro shock> 2026 inflation Fed rate cut outlook"
Query 3: "US banking system stress 2026 regional banks <relevant exposure>"
Query 4: "private credit shadow banking systemic risk 2026"
Query 5: (optional) "<sector> rotation defensive vs cyclical 2026"
```

Substitute current year. Use `<>` placeholders for scenario specifics.

### Step 3 — Map to historical analog

Walk the 6-signature checklist (table above). For each signature, mark ✓/⚠️/✗ based on web evidence.

Examples:
- AI bubble + oil shock + CRE wave → check signature 1 (✓ AI), 2 (⚠️ banks via private credit), 3 (✓ private credit opacity), 4 (✓ Fed inflation-constrained), 5 (⚠️ CRE maturity wave), 6 (partial; AI rotation possible)
- VIX spike alone → check signature 1 (depends), 2-6 generally ✗ → mild

### Step 4 — Estimate Fed intervention capacity

Critical sub-step. Score:
- Rate space: how far from zero?
- Balance sheet: room to QE without dollar credibility hit?
- Inflation print: above or below 3%?
- Banking health: are banks the patient or the rescuer?

If all 4 favorable → Fed can act = ~2020-like rescue available
If 1-2 constrained → Fed limited, partial rescue
If 3-4 constrained → **Fed paralyzed** = 2008-class risk surfaces

### Step 5 — Output survival probability + branches

Always structure output as branches (don't give a single number, give the path tree):

**Path A — Soft landing**: probability + what it looks like + fh likely outcome
**Path B — Sector unwind without banking crisis**: probability + analog (e.g. 2001-02) + fh likely outcome
**Path C — 2008-class systemic**: probability + analog (2008) + fh likely outcome (-40%+)

P(survive) = P(A) + P(B), where survive means fh year-loss < ~-20%.

### Step 6 — List trigger signals to monitor

The user wants advance warning. List 3-5 specific data points that, if they materialize, would shift probability toward Path C. Examples:
- "Private credit default rate breaks above X%"
- "Fed cuts rates while inflation > 3% (= stagflation surrender)"
- "Regional bank failure announces CRE mark-down"
- "VIX > Y for N consecutive weeks"

## Output format template

This is the canonical output for the monthly automation run. Keep the
section order and headers identical so downstream parsing stays stable.

```
**[scenario name or "Monthly Macro Snapshot YYYY-MM-DD"] — Survival Assessment**

## Risk Score: <0-100>
Composite scalar derived per the scoring rubric below.

## Survival Probability
P(survive) = X% — fh bucket year-loss < ~-20%

## Recommendation: <continue | reduce / risk-limit | stop>
One-line rationale tying recommendation to risk score bracket.

## Key Reasons
- 3-5 bullets, each pointing to specific macro evidence and what it
  implies for the system's failure mode (institutional cover-buy
  durability).

## Macro state snapshot
| Dimension | 2008 condition | Current | Signature ✓/⚠️/✗ |
|---|---|---|---|
| Leveraged bubble in core financial sector | ... | ... | ✓ |
| Banks ARE the patient | ... | ... | ⚠️ |
| Counterparty opacity | ... | ... | ... |
| Fed capacity constrained | ... | ... | ... |
| Slow visible unwind | ... | ... | ... |
| No sector rotation possible | ... | ... | ... |

## Fed intervention capacity
- Rate space: ...
- Balance sheet space: ...
- Inflation constraint: ...
- Banking sector health: ...
- Verdict: <can act | partially constrained | paralyzed>

## Path branches
A. Soft landing — P(~X%): brief description
B. Sector unwind — P(~Y%): brief description
C. 2008-class — P(~Z%): brief description

## Trigger signals to monitor
- Specific data point + threshold + what shift it implies
- ...

## Sources
- [Title](URL)
- [Title](URL)
- ...
```

## Scoring rubric (Risk Score 0-100)

Composite score is signature_count_component + fed_paralysis_component:

**Signature component** (0-60): sum of signatures from the macro table
- Each ✓ signature: 10 points
- Each ⚠️ partial: 5 points
- Each ✗ absent: 0 points
- Max: 6 × 10 = 60

**Fed paralysis component** (0-40):
- Fed "can act" (rate space + balance sheet + inflation room + healthy banks): 0
- One constraint flagged: 10
- Two constraints flagged: 20
- Three constraints flagged: 30
- "paralyzed" (all four constrained): 40

**Recommendation thresholds**:
| Risk Score | Recommendation | Operational implication |
|---:|---|---|
| 0-29 | **continue** | normal operation, no changes |
| 30-59 | **reduce / risk-limit** | consider lowering max_positions, tightening dollar volume filter, or pausing the most stress-exposed bucket (fh) — final call is Cal's |
| 60-100 | **stop** | the macro state is approaching 2008-class; pause new fh entries until at least one signature/Fed constraint clears — final call is Cal's |

Always include the numeric score, the bracket it falls in, and the
recommendation — even when the recommendation is "continue".

## Calibration anchors

To stay calibrated across sessions, anchor estimates to these historical priors:
- Year with ZERO 2008-class signatures: P(survive) ≈ 95-99% (like 2018, 2022)
- Year with 1-2 signatures: P(survive) ≈ 85-95% (like 2011, 2015)
- Year with 3-4 signatures, Fed has capacity: P(survive) ≈ 75-90% (like 2020 ex-ante)
- Year with 3-4 signatures, Fed paralyzed: P(survive) ≈ 60-80%
- Year with 5-6 signatures: P(survive) ≈ 30-50% (rare; 2008 only example)

Don't output probabilities < 30% or > 95% unless evidence is overwhelming — be epistemically humble.

## What this skill is NOT

- Not a trading recommendation
- Not a forecast of market direction
- Not a substitute for active monitoring
- Only assesses: "will the existing strategy mechanism break under this scenario"

Cal already knows the strategy can ride bull markets. The question this skill answers is **downside survival**, narrowly defined as "will fh have a >-20% year that the other buckets can't compensate".
