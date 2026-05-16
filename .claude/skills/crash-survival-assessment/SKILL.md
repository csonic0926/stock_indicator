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

System fails when **institutional cover-buy behaviour is net-absent
across fh's detection universe for sustained period (~1-2 months+)**
because:

1. vacuum_turn pattern keeps visually matching (CTA forced sell ✓, retail panic ✓)
2. But 第二層假設 (institutional cover-buy materializes) **doesn't fire**
3. Position enters → no rebound → slides into max_hold cut → typically -25% to -50% per trade
4. fh entries fire at 0.5/day rate during stress → universe-wide concurrent damage
5. Signal exit is **trend-reversal cross-down detector**, NOT falling-knife exit. Cannot save losing position in sustained downtrend (single-bar event, gets swallowed by min_hold gate).

Note on "net-absent": institutions never literally disappear (defensives, treasuries, money markets still see flows). The clinical meaning is: **net institutional buying of vacuum-pattern dislocations is gone across the broad set of names fh would target**. For fh's purposes that is effectively "the market". Use this clinical reading, not literal "all institutions gone".

### Two-axis survival framework (primary reasoning tool)

Map each crash candidate onto two axes:

**Axis 1 — Duration**: how long is institutional bid net-absent from fh's universe?
- **Fast** (< 1 month): Fed has rate space + balance sheet room + low inflation + healthy banks → can act as market maker of last resort within days/weeks. Examples: 2018, 2020, 2015, 1987.
- **Medium** (1-3 months): one or two Fed constraints flagged. Partial rescue, partial pain. Examples: 2011 partial, 2022 partial.
- **Slow** (3+ months): Fed paralyzed by inflation/banking health/exhausted tools. Examples: 2008.

**Axis 2 — Breadth**: does the stress span the full universe, or is there a sectoral refuge?
- **Contained**: stress lives in one sector; defensives, other sectors keep institutional bid; rotation possible. Examples: 2011 (Eurozone banks), 2018 Q4, 2022 (growth-vs-value).
- **Mixed**: spreading but with partial refuges. Examples: 2015 China shock, 2001 dot-com tail.
- **Universal**: cross-asset / cross-sector contagion, no refuge. Examples: 2008, 2020 COVID (initially).

Cross-product grid:

|                | **Fast Fed**          | **Medium Fed**         | **Slow Fed**          |
|---|---|---|---|
| **Contained**  | survive (2018, 2011) | survive (2022)        | survive likely (rotation works) |
| **Mixed**      | survive (2015, 2001) | survive likely        | uncertain — case dependent |
| **Universal**  | survive (2020, 1987) | uncertain             | **2008-class — DEATH** |

**Only the bottom-right corner (Slow + Universal) is fatal.** Every other cell, at least one survival mechanism applies:
- Fast Fed → time-based survival (institutions return before fh max_hold cuts)
- Contained breadth → sector-rotation survival (fh still finds bid in other sectors)
- Both → safe with margin

### Inputs to the two-axis evaluation

When evaluating a candidate scenario, gather evidence for each axis:

**Duration axis inputs** (= Fed/external intervention speed):
- Fed rate space (how far from zero?)
- Fed balance sheet space (QE room without dollar credibility hit?)
- Inflation constraint (cutting/QE while inflation > 3-4% is politically explosive — Fed will hesitate)
- Banking sector health (if banks ARE the patient, Fed has more friction)
- Fiscal capacity (TARP-style backstop politically feasible?)
- **Pre-condition: is a shock catalyst already visible?** (Bear Stearns
  precursor, repo freeze, mark-to-market cliff approaching, FSB
  active-crisis warning, etc.) — if NO and banking is healthy, the
  Duration axis collapses to Fast regardless of the other inputs (see
  Step 3 gate). Fed paralysis is a contingent risk that only matters
  when paralysis must be tested.

**Breadth axis inputs** (= sectoral refuge availability):
- Where does the stress originate? Single sector or cross-cutting (banking system)?
- Are defensives unaffected? (KO, PG, WMT type — staples should hold if breadth is contained)
- Is there cross-asset contagion? (Equity + credit + commodities + FX all moving together)
- Is there a hiding-place sector with institutional bid? (Energy in 2001, defensives in 2022)

## Historical crash database (tagged with two-axis position)

| Crash | Duration axis | Breadth axis | Trigger / mechanism | fh result |
|---|---|---|---|---|
| **2008 GFC** | **Slow** (6+ months until Fed/TARP/QE fully ramped) | **Universal** (repo run, all leveraged names hit, banks ARE patient) | Subprime / Lehman / repo freeze / forced deleveraging | **-37% to -50%** (catastrophic — the only failure) |
| **2020 COVID** | Fast (Fed acted in days: PDCF, SMCCF) | Universal initially (every risk asset sold) | Pandemic / dealer inventory crash | **+24%** (caught V-shape) |
| **2018 Q4** | Fast (Powell pivot January) | Contained (mechanical de-risking, defensives held) | Fed hike scare + trade war | -0.77% (flat) |
| **2011 Eurozone** | Fast (Fed not needed for US; ECB acted) | Contained (sectoral — Euro banks; US defensives fine) | Greek/PIIGS sovereign stress | **+17.62%** |
| **2015** | Fast (rebound within weeks) | Mixed (China shock spilled into EM/commodities, US partial) | China devaluation / oil crash | **+20.92%** |
| **2022** | Medium (deliberate hike cycle; Fed *causing* the stress) | Contained (rotation: growth → value, energy outperformed) | Inflation / Fed hikes | **+23.62%** |
| **2002 (dot-com tail)** | Medium (slow grind 2000-02; Fed cut throughout) | Contained (tech destroyed, energy/staples rallied) | Bubble unwind, sector rotation | **+23.67%** |
| **2001 (dot-com)** | Fast Fed cuts (550bps in 12 months) | Mixed (tech-led but eventually broader) | Equity bubble pop, no banking crisis | -10.45% (mild) |
| **1996** | Fast | Contained (EM-only) | EM stresses (Mexico aftershock) | -7.57% (mild) |
| **1994** | Fast | Mixed (bond rout, equity wobble) | Fed hike shock | -6.47% (mild) |

All fh-bucket numbers: 1994_clean baseline sim, fh only, max_hold=14, exit_alpha_factor=3, no reset_hold_on_reentry. Total portfolio numbers differ (typically fh dominates the loss in bad years).

**Pattern**: only the cell (Slow Fed × Universal breadth) produces catastrophic failure. The framework predicts the historical record without exception.

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

### Step 3 — Score the Duration axis (Fed intervention speed)

**The semantic question this axis answers**: *"If an external shock hits
tomorrow that requires Fed to act as market maker of last resort, can
Fed deliver that rescue fast enough to keep institutional bid net-absence
under ~1-2 months?"*

This is NOT "is Fed's policy stance loose or tight". 2022 had 0% rates
and 6.8% inflation but Fed was the active agent driving an orderly hike
cycle — no rescue was needed and none failed.

**Gate — apply before counting constraints:**

If BOTH of these hold:
- Banking system health = "banks rescuing" (no visible mark-down stress,
  no failures, big banks providing rather than receiving liquidity)
- No visible external shock catalyst already materialising (no Bear
  Stearns event, no Lehman-precursor, no repo freeze, no obvious mark-
  to-market cliff approaching)

→ classify Duration as **Fast (0)** regardless of the constraint table
below. Fed paralysis only matters when Fed is needed; absent a shock
plus healthy banks, low-rate-space + high-inflation alone don't trigger
the failure path.

**Otherwise** (banking already stressed OR shock catalyst visible):
walk the full constraint table.

| Input | Favorable (Fed fast) | Constrained | Severely constrained |
|---|---|---|---|
| Rate space | > 300 bps above zero | 100-300 bps | < 100 bps |
| Balance sheet | < 25% of GDP | 25-35% of GDP | > 35% of GDP |
| Inflation | < 2.5% | 2.5-4% | > 4% |
| Banking health | banks rescuing | mixed signals | banks ARE the patient |
| Fiscal capacity | bipartisan TARP-style possible | partisan stalemate likely | gridlocked |

Tally constraints (excluding rate space if balance sheet still has room and vice versa — they substitute partially). Classify:
- 0-1 constraint flagged → **Fast** (axis score 0)
- 2-3 flagged → **Medium** (axis score 25)
- 4-5 flagged → **Slow** (axis score 50)

**Gate sanity check examples** (from retrospective backtest):
- Q4 2021 pre-2022: rate 0% + CPI 6.8% looks Slow, but banking healthy
  + no shock catalyst → **Fast (0)** by gate. Outcome: fh +24%. ✓
- Q2 2008 pre-GFC: Bear Stearns failed Q1, banks ARE the patient → gate
  doesn't apply → walk table → 4 constraints → **Slow (50)**. ✓
- Q4 2019 pre-COVID: banking healthy + no visible shock → **Fast (0)**.
  COVID itself was exogenous and unforecastable. Outcome: fh +24%. ✓

### Step 4 — Score the Breadth axis (sectoral refuge availability)

Evaluate breadth-axis inputs:

| Input | Contained | Mixed | Universal |
|---|---|---|---|
| Stress origin | one sector | multi-sector but with refuges | banking system / cross-asset |
| Defensives state | unaffected | mildly hit | also under pressure |
| Cross-asset contagion | equity-only | equity + credit | equity + credit + commodities + FX |
| Hiding-place sector | clear (e.g. energy 2001, defensives 2022) | partial | none |

Classify:
- All inputs lean "contained" → **Contained** (axis score 0)
- Mixed signals → **Mixed** (axis score 25)
- All inputs lean "universal" → **Universal** (axis score 50)

### Step 5 — Compose risk score + recommendation

Risk Score = Duration axis score + Breadth axis score (0-100).

The grid (from "Two-axis survival framework"):
- Score 0 — fast Fed × contained breadth → trivial, system thrives (like 2018)
- Score 25-50 — one axis bad → still surviving (like 2020 universal-but-fast, or 2022 contained-but-medium)
- Score 75 — both axes worsening → uncertain (no clean historical precedent; closer to 2001-02 if rotation still works)
- Score 100 — slow Fed × universal breadth → 2008-class → **system fails**

### Step 6 — Output survival probability + path branches

Translate risk score into P(survive) using calibration anchors, then expand into 3 paths so Cal sees the conditional tree:

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
Decomposition: Duration <0/25/50> + Breadth <0/25/50> = <total>

## Survival Probability
P(survive) = X% — fh bucket year-loss < ~-20%

## Recommendation: <continue | reduce / risk-limit | stop>
One-line rationale tying recommendation to risk score bracket.

## Key Reasons
- 3-5 bullets, each pointing to specific macro evidence and what it
  implies for the system's failure mode (institutional cover-buy
  durability in fh's universe).

## Duration axis (Fed/external intervention speed)
| Input | Current state | Constraint flagged? |
|---|---|---|
| Rate space | <bps above zero> | ✓/✗ |
| Balance sheet | <% of GDP> | ✓/✗ |
| Inflation | <print %> | ✓/✗ |
| Banking health | <state> | ✓/✗ |
| Fiscal capacity | <state> | ✓/✗ |

Classification: <Fast / Medium / Slow> → axis score <0/25/50>

## Breadth axis (sectoral refuge availability)
| Input | Current state | Tendency |
|---|---|---|
| Stress origin | <sector / cross-asset> | contained / mixed / universal |
| Defensives state | <unaffected / hit> | ... |
| Cross-asset contagion | <scope> | ... |
| Hiding-place sector | <named or none> | ... |

Classification: <Contained / Mixed / Universal> → axis score <0/25/50>

## Path branches
A. Soft landing — P(~X%): brief description, closest historical analog
B. Sector unwind — P(~Y%): brief description, closest historical analog
C. 2008-class — P(~Z%): brief description, the system-killer path

## Trigger signals to monitor
- Specific data point + threshold + which axis it would shift, and in what direction
- ...

## Sources
- [Title](URL)
- [Title](URL)
- ...
```

## Scoring rubric (Risk Score 0-100)

Composite of the two survival axes (defined in detail in Steps 3-4 of
the workflow):

**Duration axis (0-50)** — how long is Fed intervention delayed?
- Fast (0-1 constraint flagged): 0
- Medium (2-3 constraints): 25
- Slow (4-5 constraints): 50

**Breadth axis (0-50)** — is there a sectoral refuge?
- Contained: 0
- Mixed: 25
- Universal: 50

**Total = Duration + Breadth** (0-100).

Why this works: the historical record shows the system survives every
cell in the 3×3 grid EXCEPT the bottom-right corner (Slow × Universal,
= score 100 = 2008). Any single axis at "safe" (0) brings score below
the 60-point fatal threshold and at least one survival mechanism (fast
recovery via Fed, or sector rotation refuge) kicks in.

**Recommendation thresholds**:

| Risk Score | Cell position | Recommendation | Operational implication |
|---:|---|---|---|
| 0-29 | both axes safe-to-mild | **continue** | normal operation |
| 30-59 | one axis mild, the other safe-to-mild | **reduce / risk-limit** | consider lowering max_positions, tightening dollar volume filter, or pausing fh — final call is Cal's |
| 60-100 | at least one axis severe with the other not safe | **stop** | macro is approaching 2008-class; pause new fh entries until at least one axis backs off — final call is Cal's |

Always emit the numeric score, the axis decomposition (e.g. "Duration 25 + Breadth 50 = 75"), and the recommendation.

## Calibration anchors

Anchor P(survive) to historical record by risk-score bracket:

| Risk Score | Closest historical analog | P(survive) bracket |
|---:|---|---|
| 0-15 | 2018, 2011 | 95-99% |
| 20-35 | 2022, 2015, 2001, 1994 | 85-95% |
| 40-55 | 2020 (universal but Fed acted) | 75-90% |
| 60-75 | no clean analog; closest = 2001-02 if rotation still saves it | 55-75% |
| 80-100 | 2008 only | 30-55% |

Don't output probabilities < 30% or > 95% unless evidence is overwhelming — be epistemically humble.

## What this skill is NOT

- Not a trading recommendation
- Not a forecast of market direction
- Not a substitute for active monitoring
- Only assesses: "will the existing strategy mechanism break under this scenario"

Cal already knows the strategy can ride bull markets. The question this skill answers is **downside survival**, narrowly defined as "will fh have a >-20% year that the other buckets can't compensate".
