Implement Sprint 6b: WebSocket + Redis Subscriber + React Frontend for the Phase 6 dashboard.

This sprint adds real-time data streaming from Redis to connected browser clients via WebSocket, and a React frontend. Sprint 6a (REST API) must be complete first.

Read these files before writing any code:
- `src/dashboard/api.py` — the existing FastAPI app (you'll add the WebSocket endpoint here)
- `src/dashboard/schemas.py` — response models
- `src/common/types.py` — TickMessage, EventMessage, SignalMessage, PositionUpdateMessage, SystemAlertMessage (the Redis message types)
- `docs/sprint_phase4_5_6_decomposition.md` Sprint 6b section — this is your spec

Do NOT modify `src/dashboard/schemas.py`, `src/dashboard/pnl_service.py`, or any `src/execution/`/`src/orchestrator/` files.

## Step 1: Create `src/dashboard/ws_handler.py`

### Class: `DashboardWSManager`

**Constructor:** `__init__(self, redis_client)`
- `self.redis = redis_client`
- `self.active_connections: list[WebSocket] = []`
- `self.subscribed_matches: set[str] = set()`
- `self._pubsub = None`
- `self._subscriber_task: asyncio.Task | None = None`

**Method: `async def connect(self, websocket: WebSocket)`**
- `await websocket.accept()`
- Add to `self.active_connections`
- If `self._subscriber_task` is None, start the Redis subscriber background task

**Method: `async def disconnect(self, websocket: WebSocket)`**
- Remove from `self.active_connections`
- If no connections remain, consider cleaning up subscriptions

**Method: `async def start_redis_subscriber(self)`**
- Create a pubsub connection from `self.redis`
- Subscribe to global channels: `position_update`, `system_alert`
- Subscribe to match-specific channels based on `self.subscribed_matches`:
  - For each match_id: `tick:{match_id}`, `event:{match_id}`, `signal:{match_id}`
- CRITICAL: Subscribe once, then listen in a loop. Do NOT subscribe inside the message loop. This is the anti-pattern from architecture.md §3.6.
- On each message: broadcast to all active connections as JSON:
  ```json
  {"channel": "tick:4190023", "data": {...parsed message...}}
  ```
- On Redis connection error: reconnect with exponential backoff (1s base, 30s max, 10 retries)

**Method: `async def update_match_subscriptions(self, match_ids: set[str])`**
- Compute diff: `new = match_ids - self.subscribed_matches`, `removed = self.subscribed_matches - match_ids`
- For each new match_id: subscribe to `tick:{id}`, `event:{id}`, `signal:{id}`
- For each removed match_id: unsubscribe from those channels
- Update `self.subscribed_matches = match_ids`

**Method: `async def broadcast(self, channel: str, data: dict)`**
- For each connection in `self.active_connections`:
  - Try to send JSON `{"channel": channel, "data": data}`
  - On error: remove the dead connection from the list

## Step 2: Add WebSocket endpoint to `src/dashboard/api.py`

Add to the existing FastAPI app:

```python
from fastapi import WebSocket, WebSocketDisconnect

# Create manager instance (needs redis_client from app startup)
ws_manager: DashboardWSManager | None = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if ws_manager is None:
        await websocket.close(code=1011)
        return
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "subscribe":
                match_ids = set(data.get("match_ids", []))
                await ws_manager.update_match_subscriptions(match_ids)
            elif msg_type == "unsubscribe":
                match_ids = set(data.get("match_ids", []))
                current = ws_manager.subscribed_matches - match_ids
                await ws_manager.update_match_subscriptions(current)
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
```

WebSocket message protocol:

Client → Server:
- `{"type": "subscribe", "match_ids": ["4190023", "4190024"]}`
- `{"type": "unsubscribe", "match_ids": ["4190023"]}`

Server → Client:
- `{"channel": "tick:4190023", "data": {...TickMessage fields...}}`
- `{"channel": "event:4190023", "data": {...EventMessage fields...}}`
- `{"channel": "signal:4190023", "data": {...SignalMessage fields...}}`
- `{"channel": "position_update", "data": {...PositionUpdateMessage fields...}}`
- `{"channel": "system_alert", "data": {...SystemAlertMessage fields...}}`

CRITICAL: All JSON keys use `sigma_MC`, not Greek `σ_MC`. TypeScript cannot use Greek letters as property names.

## Step 3: Create `src/dashboard/frontend/`

React application using Vite + TypeScript.

Initialize:
```bash
cd src/dashboard
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

Create these components:

**`src/hooks/useWebSocket.ts`**
- Manages WebSocket connection to `ws://localhost:8001/ws`
- Auto-reconnect with exponential backoff: start at 1s, double on each failure, cap at 30s, max 10 retries
- On reconnect: re-send subscribe message with current match_ids
- Parse incoming messages and dispatch to React state via context or callbacks
- Export: `useWebSocket(url)` returning `{connected, lastMessage, sendMessage, subscribe, unsubscribe}`

**`src/components/CommandCenter.tsx`**
- Main view: grid of MatchCard components
- Fetches initial match list from `GET /api/matches`
- Subscribes to all active match IDs via WebSocket
- Cards appear/disappear as matches start/end

**`src/components/MatchCard.tsx`**
- Props: match data (score, minute, engine_phase, P_model, surprise_score, positions, etc.)
- Displays: score, match minute, engine_phase badge
- Three probability bars (home_win, draw, away_win) from P_model
- SurpriseScore indicator (orange when > 0.5)
- Open positions count + unrealized P&L
- Green/red light for order_allowed status
- Click → navigate to MatchDeepDive

**`src/components/MatchDeepDive.tsx`**
- Detailed view for a single match
- Fetches tick history from `GET /api/matches/{id}/ticks`
- P_model time series chart (use recharts or similar)
- Position entry/exit markers
- Goal events annotated with SurpriseScore

**`src/components/PnLDashboard.tsx`**
- Fetches from `GET /api/pnl` and `GET /api/pnl/history`
- Cumulative P&L line chart
- Breakdown tables by market type and league
- Win rate display

**`src/components/SystemStatus.tsx`**
- Fetches from `GET /api/system/status`
- Active container count, open positions, bankroll display (paper + live)
- Alert feed from system_alert messages

## Step 4: Create tests

`tests/dashboard/test_ws_handler.py`:
- `test_connect_adds_to_active`: connect → verify len(manager.active_connections) == 1
- `test_disconnect_removes`: connect then disconnect → len == 0
- `test_subscribe_adds_channels`: send subscribe with ["4190023"] → subscribed_matches contains "4190023"
- `test_unsubscribe_removes_channels`: subscribe then unsubscribe → subscribed_matches empty
- `test_broadcast_to_all_clients`: 2 connected clients → both receive message
- `test_dead_connection_cleaned`: mock a connection that raises on send → removed from active list

Integration test:
```python
async def test_sprint_6b_ws_integration():
    """Publish a tick to Redis, verify WebSocket client receives it."""
    # Create test app with real Redis
    # Connect WebSocket client
    # Send subscribe for match "4190023"
    # Publish TickMessage to Redis channel "tick:4190023"
    # Receive on WebSocket, verify channel and data.t value
```

## Step 5: Verify

1. `python -m pytest tests/dashboard/ -v` — all Sprint 6a + 6b tests pass
2. `cd src/dashboard/frontend && npm run build` — React app builds without errors
3. Existing tests unaffected
