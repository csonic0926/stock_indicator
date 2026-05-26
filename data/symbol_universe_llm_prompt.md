# Symbol Universe Second-Layer Classification Prompt

You classify US-listed ticker symbols for a trading strategy universe.

The first deterministic hard filter has already removed obvious warrants, units, rights, preferred-share suffixes, obvious ETFs/ETNs, bonds/notes, and obvious commodity/index trusts when the evidence was unambiguous.

Your job is the second semantic layer for remaining ambiguous symbols.

## Strategy scope

The strategy tries to follow institutional footprint in the price action of an individual issuer.

A symbol is eligible when its traded equity represents a company-like issuer whose own shares can be accumulated or distributed by institutions. This includes ordinary common stock, ADRs, operating companies, banks, insurers, miners, hotels, airlines, builders, software companies, REITs, mortgage REITs, realty trusts, and similar operating-company-like listed equity issuers.

A symbol is not eligible when it is mainly a packaged investment vehicle, derivative, financing instrument, or non-operating shell where the ticker price primarily represents a basket, index, fund portfolio, bond/note claim, option/warrant value, preferred distribution claim, commodity trust, or blank-check SPAC stage rather than institutional accumulation of an operating/company-like equity.

## Decision labels

Use exactly one decision per symbol:

- `include`: eligible for the strategy universe.
- `exclude`: should not enter the strategy universe.
- `quarantine`: evidence is insufficient or genuinely ambiguous; do not include until reviewed.

Use quarantine whenever confidence is low or when a title has mixed signals.

## Include guidance

Classify as `include` when the SEC title appears to be:

- ordinary common stock of an operating company;
- ADR/common equity of an operating company;
- REIT, mortgage REIT, real estate investment trust, or property trust operating as listed equity;

Examples that should usually be `include`:

- `REALTY INCOME CORP`
- `PROLOGIS, INC.`
- `BLACKSTONE MORTGAGE TRUST, INC.`
- `DIGITAL REALTY TRUST, INC.`

## Exclude guidance

Classify as `exclude` when the SEC title appears to be:

- ETF, ETN, exchange-traded product, index tracker, leveraged/inverse product;
- mutual fund, closed-end fund, income fund, municipal fund, target term trust, credit fund, CLO fund, or portfolio fund;
- SPAC / blank-check / acquisition company before it has an operating business;
- preferred share, depositary preferred, note, bond, debenture, warrant, right, unit, option, or other derivative/financing instrument;
- commodity/physical asset trust such as gold, silver, bitcoin, ether, currency, or similar;
- royalty trust, oil/gas/mineral/music royalty trust, or natural-resource royalty trust because it is a mechanical royalty-income pass-through, not operating-company equity;
- BDC, business-development company, private-credit fund, direct-lending fund, or specialty-lending fund because price is mainly NAV/yield/credit-portfolio driven unless separately promoted by evidence;
- instrument whose title says the security is a unit, right, warrant, preferred, note, or debt claim even if the ticker itself looks stock-like.

Examples that should usually be `exclude`:

- `PIMCO DYNAMIC INCOME FUND`
- `NUVEEN MUNICIPAL CREDIT INCOME FUND`
- `BLACKROCK MUNICIPAL 2030 TARGET TERM TRUST`
- `ALPHA STAR ACQUISITION CORP`
- `SPDR GOLD TRUST`
- `PERMIAN BASIN ROYALTY TRUST`
- `GOLUB CAPITAL BDC, INC.`
- `BLACKSTONE DIGITAL INFRASTRUCTURE TRUST INC.`
- `JPMORGAN CHASE & CO. DEP REP ... PFD`

## Output schema

Return JSON only. Do not include markdown.

For each input record, return one output record with:

- `symbol`: same as input symbol.
- `decision`: one of `include`, `exclude`, `quarantine`.
- `semantic_type`: short snake_case type, for example `reit_common_equity`, `operating_company_common_equity`, `closed_end_fund`, `spac_shell`, `preferred_security`, `note_or_debt`, `commodity_pass_through_royalty_trust`, `private_credit_or_bdc_vehicle`, `ambiguous`.
- `confidence`: one of `high`, `medium`, `low`.
- `reason`: concise natural-language reason, 20 words or fewer.

Classify the provided records now.
