# IBKR Multi-Broker Dashboard Plan

## Status

- **State:** Paused / archived
- **Archived on:** 2026-08-13
- **Resume condition:** The IBKR margin account and the required API/trading
  permissions are active, and an IBKR paper-trading login is available.
- **Current production broker:** Futu
- **Implementation status:** Research and design only. No IBKR trading code,
  dependency, service, or dashboard change has been deployed.

Do not enable IBKR live order submission as part of account setup. Resume with
read-only connectivity and paper trading first.

---

## Objective

Extend the existing dashboard so Futu and IBKR can operate concurrently. IBKR
must be added as a second broker rather than replacing Futu.

The intended flow is:

```text
Cron strategy signals
        |
        v
Global portfolio and slot allocation
        |
        v
Broker routing decision
        +--> Futu adapter --> Futu OpenD
        +--> IBKR adapter --> IB Gateway
```

The signal layer remains broker-neutral. Broker state participates only in the
order/allocation and execution layers.

---

## API Decision

### Recommended production connection

Use the **TWS API through IB Gateway**, with IBKR's official Python `ibapi`
client.

- Production gateway default: `127.0.0.1:4001`
- Paper gateway default: `127.0.0.1:4002`
- TWS live default: `127.0.0.1:7496`
- TWS paper default: `127.0.0.1:7497`
- Use TWS for early visual verification if useful; use IB Gateway for the
  eventual persistent service.

IB Gateway is preferred for this repository's launchd-based operation because
it is less resource intensive. IBKR also recommends IB Gateway for IBHK users
because inactivity locking in TWS can disconnect API sessions.

References:

- [TWS API introduction](https://www.interactivebrokers.com/docs/tws-api/doc/introduction)
- [TWS and IB Gateway comparison](https://www.interactivebrokers.com/docs/tws-api/doc/download-tws-or-ib-gateway/download-tws-or-ib-gateway)
- [IBKR API settings](https://www.interactivebrokers.com/docs/tws-api/doc/tws-settings/introduction)
- [TWS API connection lifecycle](https://www.interactivebrokers.com/docs/tws-api/doc/connectivity/establishing-an-api-connection)

### Why Client Portal Web API is not the primary path

For an individual account, the Client Portal Gateway requires interactive
authentication, cannot automate the brokerage-session login, requires daily
reauthentication, and needs periodic session keepalive calls. This does not fit
the existing unattended launchd workflow as well as IB Gateway.

Reference:

- [Client Portal Web API authentication and session behavior](https://ibkrcampus.com/campus/ibkr-api-page/cpapi-v1/)

### Python client policy

Prefer the official `ibapi` package installed from the current TWS API
distribution. The original `ib_insync` package is no longer maintained.
`ib_async` may be useful for prototyping but is third-party and is not supported
by IBKR API Support.

Reference:

- [IBKR guidance on ib_insync and ib_async](https://www.interactivebrokers.com/docs/tws-api/doc/third-party-api-platforms/non-standard-tws-api-languages-and-packages/ib-insync-and-ib-async)

---

## Resume Prerequisites

Before restarting implementation, verify all of the following:

1. IBKR margin account permission is active.
2. The account is funded and is eligible for the TWS API.
3. A paper-trading account and its distinct paper username are available.
4. IB Gateway or TWS is installed.
5. The official TWS API Python client is installed in this repository's venv.
6. API socket access is enabled and restricted to localhost.
7. The socket port is confirmed.
8. A dedicated API username is considered so Client Portal or another trading
   application does not disrupt the Gateway session.
9. Required US market-data subscriptions or an approved alternative quote
   source are available.

Suggested paper configuration:

```text
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=0
IBKR_ACCOUNT_ID=<paper account id>
IBKR_TRADING_ENV=PAPER
IBKR_TRADING_ENABLED=false
```

For the first read-only probe, keep trading disabled. In Gateway/TWS, enable
socket clients, verify the port, and keep the API read-only until paper order
testing begins.

The connection is not ready merely because the TCP socket opened. The client
must wait for `nextValidId` before issuing dependent requests or placing orders.

---

## Local Baseline at Archive Time

Inspection on 2026-08-13 found:

- Futu OpenD listening on `127.0.0.1:11111`.
- Existing dashboard responding on `127.0.0.1:8080`.
- No IB Gateway or TWS installation detected.
- `ibapi`, `ib_async`, and `ib-insync` were not installed in the project venv.
- IBKR ports `4001`, `4002`, `7496`, and `7497` were not listening.
- Port `5000` belonged to macOS Control Center, not an IBKR Client Portal
  Gateway.
- Files named `multi_bucket_production_*_ibkr.json` configure simulator broker
  costs only; they are not live IBKR integrations.

Recheck this baseline rather than assuming it is still current when the plan is
resumed.

---

## Existing Futu Coupling to Remove Carefully

The current live path is broker-specific in several places:

- `dashboard.py` directly creates and closes Futu contexts.
- `/api/futu/positions` exposes a single-broker response.
- Order preview reads only Futu account, position, order, and quote state.
- Global slots are calculated from Futu holdings and orders only.
- Position sizing assumes Futu account values in HKD and a fixed HKD/USD rate.
- Execution directly imports Futu order enums.
- Order authorization is keyed by `(side, symbol)` without broker or account.
- Pending entry intents and order logs do not identify a broker or account.
- Entry metadata recovery depends partly on Futu order remarks.
- The pre-open entry scheduler and TP/SL scheduler are Futu-specific.
- The UI has one Futu connection badge, one account card, and one broker
  position table.

The following behavior must be preserved while this coupling is extracted:

- The cron publishes broker-neutral tradable candidates.
- The dashboard remains the single real allocation pass.
- Live broker positions and orders remain execution truth.
- Adaptive TP/SL virtual history remains statistical state, not a portfolio
  ledger.
- Expectancy and WR phantom positions continue to occupy global slots without
  deploying capital.

---

## Formal Multi-Broker Model

Let `B` be the enabled broker-account set, initially containing Futu and later
IBKR. For broker account `b` on market date `t`:

- `P_b(t)` is its positive live position set.
- `U_b(t)` is its active, unfilled BUY set.
- `H(t)` is the broker-independent open phantom set.
- `C` is the production global maximum position count.

The occupied global portfolio is:

```text
O(t) = union over b of P_b(t)
       union union over b of U_b(t)
       union H(t)
```

Available slots are:

```text
remaining_slots(t) = max(0, C - cardinality(O(t)))
```

`C` remains the existing global limit. Adding IBKR must not implicitly turn a
seven-position strategy into fourteen positions.

The set operation must use the configured same-symbol policy. A position on
Futu and the same pending BUY on IBKR cannot silently consume or create the
wrong number of logical strategy slots.

### State partitions

1. **Observable broker state**
   - account values and currencies
   - positions
   - open orders
   - executions and fills
   - connection and snapshot timestamps

2. **Strategy/statistical state**
   - signal candidates
   - adaptive TP/SL virtual history
   - expectancy and WR sensor state

3. **Allocation state**
   - selected strategy candidate
   - broker and account route
   - target notional and quantity
   - bucket and frozen TP/SL metadata
   - phantom status

4. **Execution state**
   - broker order identifiers
   - normalized order status
   - fills
   - protective-order coverage

No persisted object should silently serve more than one of these roles.

### Required allocation identity

Every selected candidate needs a stable logical allocation identifier and at
least these fields:

```text
allocation_id
signal_date
strategy_symbol
broker_symbol or broker_contract_id
bucket
strategy_id
broker_id
account_id
environment
target_notional
quantity
tp_sl_metadata
```

Broker order identifiers are execution references, not logical allocation
identities.

---

## Broker Adapter Boundary

Create a broker package rather than adding more broker conditionals to
`dashboard.py`:

```text
src/stock_indicator/brokers/
    models.py
    protocol.py
    registry.py
    futu_adapter.py
    ibkr_adapter.py
    ibkr_session_manager.py
```

The normalized adapter surface should cover:

- connection health and data freshness
- account snapshots
- positions
- open and completed orders
- executions
- quotes required by live sizing
- contract/symbol resolution
- place, modify, and cancel operations
- broker buying-power or order-impact validation
- protective-order reconciliation

IBKR uses asynchronous callbacks and subscriptions. `ibkr_session_manager.py`
should own one persistent connection, callback thread, request correlation,
timeouts, next order ID, and normalized cached snapshots. FastAPI request
handlers must not open and close a new IBKR socket for every HTTP request.

Use `permId` together with the API order ID and local `allocation_id` when
persisting IBKR order identity. Client-ID ownership matters for order
modification and cancellation.

References:

- [Portfolio data callbacks](https://www.interactivebrokers.com/docs/tws-api/doc/quick-start/requesting-portfolio-data)
- [IBKR client ID behavior](https://www.interactivebrokers.com/docs/tws-api/doc/order-management/client-id-0-and-the-master-client-id)
- [IBKR order modification rules](https://www.interactivebrokers.com/docs/tws-api/doc/orders/modifying-orders)

---

## Routing Policy

Allocation and routing are separate decisions:

1. Rank broker-neutral strategy candidates.
2. Apply global risk, expectancy, slot, bucket, and same-symbol rules once.
3. Assign each accepted funded allocation to exactly one broker account.
4. Size it using that broker/account's current equity, buying power, currency,
   and price.
5. Revalidate a fresh preview immediately before execution.

For the first dual-broker release, display a recommended broker and permit the
operator to choose the route in preview. Automatic routing can be introduced
only after paper operation is stable.

A later automatic routing policy may consider:

- configured capital target per broker
- broker or account slot ceiling
- available buying power
- current broker exposure
- connection and snapshot freshness
- supported instruments and sessions
- transaction cost policy

If one broker rejects an order, execution must not automatically retry it on
the other broker. Generate a new preview because buying power, pending orders,
and global slot ownership may have changed.

---

## Dashboard Design

Preserve the current dense, dark operational style, but make broker health,
portfolio risk, and pending execution the primary scan path.

### Portfolio overview

Show:

- global occupied and available slots
- phantom slots
- strategy/bucket exposure
- reporting currency and FX timestamp
- risk-score, expectancy, and WR gate state

### Broker health strip

For each enabled broker show:

- broker and account
- connected/disconnected/stale status
- PAPER/SIMULATE/LIVE environment
- read-only or trading-enabled state
- last successful snapshot/callback time
- IBKR client ID where applicable

Stale broker data must be visually distinct from an empty account.

### Accounts

Display each broker account in its native currency. Do not directly add HKD
and USD values. Any consolidated total requires an explicit reporting currency,
an FX source, and an as-of timestamp.

### Positions

Use a consolidated table with broker/account columns and broker filters. Keep
separate rows when the same symbol exists at both brokers. Include:

- broker and account
- symbol or contract
- bucket and strategy
- quantity, cost, price, market value, and P/L
- entry metadata status
- protective-order coverage
- data freshness

### Order preview

Each row must include:

- broker and account
- route reason
- side, symbol, bucket, rank, quantity, and reference price
- native notional and reporting-currency equivalent
- TP/SL metadata
- current broker buying-power validation
- preview snapshot timestamp

Group confirmation results by broker. A partial broker failure must be shown
explicitly and must not trigger hidden compensating orders.

### Protective orders

Futu may retain the existing post-fill TP/SL reconciliation initially. IBKR can
use bracket/OCA orders, but live TP/SL prices must preserve the strategy's
current fill-price semantics and min-hold rules.

Reference:

- [IBKR profit taker and stop-loss support](https://www.interactivebrokers.com/docs/tws-api/doc/orders/place-order/adding-a-profit-taker-and-stop-loss)

All new user-facing text should be moved to a resource file rather than adding
more hard-coded text to the dashboard module.

---

## Implementation Phases

### Phase 0: Read-only connectivity spike

- Install IB Gateway/TWS and the official Python client.
- Connect only to the paper account.
- Wait for `nextValidId`.
- Load account, positions, open orders, and a known US stock contract.
- Verify reconnect and timeout behavior.
- Do not expose `placeOrder` through the dashboard.

Deliverable: a tested read-only probe and documented local setup.

### Phase 1: Extract the existing Futu adapter

- Introduce normalized broker models and protocol.
- Move Futu access behind `FutuBrokerAdapter`.
- Preserve all existing Futu behavior and API compatibility.
- Keep IBKR disabled.

Deliverable: all current Futu/dashboard tests pass without production behavior
changes.

### Phase 2: Add IBKR read-only dashboard support

- Add the persistent IBKR session manager.
- Add normalized broker/account and portfolio endpoints.
- Retain `/api/futu/positions` temporarily for compatibility.
- Show IBKR account, health, positions, and orders.
- Exclude IBKR from allocation and execution.

Deliverable: concurrent read-only Futu and IBKR portfolio display.

### Phase 3: Add global allocation and broker routing

- Calculate global slots across all broker accounts and phantoms.
- Add broker/account/allocation identity to preview keys.
- Add explicit manual route selection.
- Add broker/account fields to pending intents and order logs.
- Persist strategy metadata independently from broker remarks.
- Preserve exactly-once expectancy acceptance per logical allocation.

Deliverable: safe, non-executing dual-broker order preview.

### Phase 4: Add IBKR paper execution

- Normalize place, cancel, modify, order-status, and execution callbacks.
- Persist IBKR API order ID, `permId`, and local allocation ID.
- Implement broker-aware entry intent and pre-open reconciliation.
- Implement broker-aware TP/SL coverage.
- Exercise reconnect and restart recovery.

Deliverable: end-to-end IBKR paper entry, fill, protection, and exit.

### Phase 5: Controlled live canary

- Keep an explicit IBKR live feature flag disabled by default.
- Require a current fresh-server preview before confirmation.
- Initially allow no more than one IBKR live BUY in a batch.
- Validate entry, fill, TP/SL, signal exit, max-hold exit, restart recovery,
  and reconciliation against the IBKR UI.
- Expand only after repeated clean cycles.

Deliverable: controlled concurrent Futu and IBKR live operation.

---

## Safety Invariants and Acceptance Criteria

Implementation is not complete until all of these hold:

1. Enabling IBKR does not change the global seven-slot cap to fourteen.
2. Bucket, same-symbol, pending-order, and phantom constraints apply across
   brokers.
3. A BUY fails closed when any required broker snapshot is missing or stale.
4. A SELL is routed to the broker account that owns the position.
5. The same candidate cannot be bought once at Futu and once at IBKR by
   accident.
6. Every live order can be traced to broker, account, environment, allocation,
   and signal metadata.
7. A rejection at one broker does not silently route to the other broker.
8. Futu's current real-order behavior and tests remain unchanged until an
   explicitly reviewed migration step.
9. Expectancy acceptance is recorded once per logical allocation, not once per
   broker API attempt.
10. Broker execution results cannot rewrite statistical virtual history as if
    it were a live position ledger.
11. Native currencies are not directly summed; consolidated figures carry an
    FX source and timestamp.
12. Paper and live states are visibly distinct and cannot share an execution
    endpoint accidentally.
13. Order confirmation is authorized against a new server-side preview that
    includes broker and account identity.
14. Partial execution and partial broker outages are surfaced, logged, and
    reconciled without hidden compensation.
15. IBKR client-ID/order-ID ownership permits reliable modification and
    cancellation after restart.

IBKR `WhatIf` margin-impact checks may be used sparingly for final validation,
not for every candidate or every dashboard refresh.

Reference:

- [IBKR WhatIf order-impact guidance](https://www.interactivebrokers.com/docs/tws-api/doc/orders/test-order-impact-what-if)

---

## First Actions When Resuming

1. Re-read this plan and the repository's current `AGENTS.md`.
2. Reinspect the live dashboard, Futu service, git status, and any changes made
   since 2026-08-13.
3. Confirm IBKR account type, region, paper username, permissions, market-data
   entitlements, and intended reporting currency.
4. Install IB Gateway/TWS and the official `ibapi` client without changing the
   production launchd services.
5. Run Phase 0 against paper in read-only mode.
6. Record actual callback fields and account-currency behavior before finalizing
   normalized models.
7. Only then begin Phase 1 adapter extraction.

Do not begin by renaming Futu APIs to IBKR or by adding IBKR conditionals inside
the existing 3,000+ line dashboard module. The broker boundary and global
allocation model must be established first.
