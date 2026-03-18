# Sprint 1 Migration — Calibration + Math

Reference: `docs/MMPP_v5_Complete.md` §4.5–§4.8 (Phase 1 Steps 6–8)

## Preliminary: The `src/math/` DO NOT MODIFY Conflict

**Decision: Lift the restriction for v5-motivated changes.**

**Justification:** The original rule protected verified v3 math code from accidental modification. v5 requires fundamentally new parameter shapes (asymmetric δ, η stoppage multipliers) inside the Numba JIT inner loop — this cannot be achieved by wrappers without defeating JIT compilation. The resolution: add a **new** `mc_simulate_remaining_v5` function alongside the existing one, preserving the old function as a backward-compatible wrapper. The old function delegates to a shared `_mc_simulate_core`. No existing caller changes until Sprint 3.

**CLAUDE.md update:** Change "DO NOT modify" to "Modify only for v5-motivated changes; preserve old function signatures as backward-compatible wrappers."

---

## Task 1.1: Update CLAUDE.md math restriction

**What:** Lift the `src/math/` DO NOT MODIFY rule to allow v5-motivated additions while preserving backward compatibility.

**Files touched:**
- `CLAUDE.md` — update `src/math/` entry

**Detailed steps:**
1. In the project structure table, change the `src/math/` comment from "4 core files (copied from v3, DO NOT modify)" to "4 core files (copied from v3, extend for v5 — preserve old signatures as wrappers)"

**Breaking changes:** None

**Test impact:**
- Existing tests that break: None
- New tests to add: None

**Verify:** `make test`

---

## Task 1.2: Add v5 parameter fields to ProductionParams

**What:** Add new calibration output fields to `ProductionParams` in `types.py` with defaults so existing code continues to work.

**Files touched:**
- `src/common/types.py` — add fields to `ProductionParams`

**Detailed steps:**
1. After the existing `delta_A: float` field (~line 60), add:
```python
# v5 asymmetric score-state effects (defaults = None → use symmetric delta_H/delta_A)
delta_H_pos: list[float] | None = None   # shape [5], home when leading
delta_H_neg: list[float] | None = None   # shape [5], home when trailing
delta_A_pos: list[float] | None = None   # shape [5], away when trailing
delta_A_neg: list[float] | None = None   # shape [5], away when leading
# v5 stoppage time multipliers (default 0.0 = no stoppage adjustment)
eta_H: float = 0.0     # 1st half stoppage, home
eta_A: float = 0.0     # 1st half stoppage, away
eta_H2: float = 0.0    # 2nd half stoppage, home
eta_A2: float = 0.0    # 2nd half stoppage, away
# v5 EKF process noise (default 0.01 = small drift)
sigma_omega_sq: float = 0.01
# v5 full gamma arrays (defaults = None → use scalar gamma_H/gamma_A)
gamma_H_full: list[float] | None = None  # shape [4]
gamma_A_full: list[float] | None = None  # shape [4]
# v5 full delta arrays (defaults = None → use scalar delta_H/delta_A)
delta_H_full: list[float] | None = None  # shape [5]
delta_A_full: list[float] | None = None  # shape [5]
```

**New/changed types:**
- `ProductionParams.delta_H_pos`: `list[float] | None = None`
- `ProductionParams.delta_H_neg`: `list[float] | None = None`
- `ProductionParams.delta_A_pos`: `list[float] | None = None`
- `ProductionParams.delta_A_neg`: `list[float] | None = None`
- `ProductionParams.eta_H/eta_A/eta_H2/eta_A2`: `float = 0.0`
- `ProductionParams.sigma_omega_sq`: `float = 0.01`
- `ProductionParams.gamma_H_full/gamma_A_full`: `list[float] | None = None`
- `ProductionParams.delta_H_full/delta_A_full`: `list[float] | None = None`

**Breaking changes:** None — all fields have defaults.

**Test impact:**
- Existing tests that break: None
- New tests to add: None (covered in Task 1.11)

**Verify:** `make test`

---

## Task 1.3: Refactor mc_core for v5 parameters

**What:** Extract simulation logic into `_mc_simulate_core` with asymmetric δ + η support. Old `mc_simulate_remaining` becomes a backward-compatible wrapper. New `mc_simulate_remaining_v5` exposes the full v5 interface.

**Files touched:**
- `src/math/mc_core.py` — refactor + add new function

**Detailed steps:**
1. Rename existing `mc_simulate_remaining` body into `_mc_simulate_core` with expanded signature:
```python
@njit(cache=True)
def _mc_simulate_core(
    t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
    b, gamma_H, gamma_A,
    delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
    Q_diag, Q_off, basis_bounds, N, seed,
    eta_H, eta_A, eta_H2, eta_A2,
    stoppage_1_start, stoppage_2_start,
):
```
2. Inside intensity computation (currently ~line 73-76), change delta lookup:
```python
# Old: delta_H[di]
# New:
if sd > 0:  # home leading
    dH = delta_H_pos[di]
    dA = delta_A_pos[di]
else:
    dH = delta_H_neg[di]
    dA = delta_A_neg[di]
```
3. Add η stoppage multiplier to intensity:
```python
eta_h = 0.0
eta_a = 0.0
if stoppage_1_start < t_current < stoppage_1_start + 10.0:
    eta_h = eta_H
    eta_a = eta_A
elif stoppage_2_start < t_current < stoppage_2_start + 10.0:
    eta_h = eta_H2
    eta_a = eta_A2
lam_H = np.exp(a_H + b[bi] + gamma_H[st] + dH + eta_h)
lam_A = np.exp(a_A + b[bi] + gamma_A[st] + dA + eta_a)
```
4. Create backward-compatible wrapper (old signature, delegates to core):
```python
@njit(cache=True)
def mc_simulate_remaining(
    t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
    b, gamma_H, gamma_A, delta_H, delta_A,
    Q_diag, Q_off, basis_bounds, N, seed,
):
    return _mc_simulate_core(
        t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
        b, gamma_H, gamma_A,
        delta_H, delta_H, delta_A, delta_A,  # symmetric: pos=neg=original
        Q_diag, Q_off, basis_bounds, N, seed,
        0.0, 0.0, 0.0, 0.0,  # no η
        45.0, 90.0,  # default stoppage starts
    )
```
5. Create v5 function:
```python
@njit(cache=True)
def mc_simulate_remaining_v5(
    t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
    b, gamma_H, gamma_A,
    delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
    Q_diag, Q_off, basis_bounds, N, seed,
    eta_H, eta_A, eta_H2, eta_A2,
    stoppage_1_start, stoppage_2_start,
):
    return _mc_simulate_core(
        t_now, T_end, S_H, S_A, state, score_diff, a_H, a_A,
        b, gamma_H, gamma_A,
        delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
        Q_diag, Q_off, basis_bounds, N, seed,
        eta_H, eta_A, eta_H2, eta_A2,
        stoppage_1_start, stoppage_2_start,
    )
```

**Breaking changes:** None — old `mc_simulate_remaining` signature unchanged.

**Test impact:**
- Existing tests that break: None — `test_model.py::test_T_exp_affects_mc_probabilities` calls `mc_simulate_remaining` which still works identically.
- New tests to add: `tests/test_math_core.py::test_mc_simulate_v5_asymmetric_delta` — verify asymmetric delta produces different results from symmetric. `test_mc_simulate_v5_eta_stoppage` — verify η increases goal rate in stoppage time.

**Verify:** `make test`

---

## Task 1.4: Add compute_remaining_mu_v5

**What:** Add a v5 variant of `compute_remaining_mu` that accepts asymmetric δ and η parameters. Old function unchanged.

**Files touched:**
- `src/math/compute_mu.py` — add new function

**Detailed steps:**
1. Add `compute_remaining_mu_v5(model, override_delta_S=None)` that:
   - Checks for `model.delta_H_pos` attribute; if present, uses asymmetric delta lookup based on sign of delta_S
   - Applies η multiplier when subinterval falls within stoppage periods
   - Falls back to original logic if asymmetric params not present
2. Old `compute_remaining_mu` remains unchanged.

**Breaking changes:** None — purely additive.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/test_math_core.py::test_compute_mu_v5_asymmetric` — verify asymmetric delta produces different μ for leading vs trailing.

**Verify:** `make test`

---

## Task 1.5: Create asymmetric delta estimation function

**What:** Implement Step 6 of v5 Phase 1: estimate separate δ⁺/δ⁻ arrays from historical intervals by MLE on leading vs trailing subsets.

**Files touched:**
- `src/calibration/step_1_6_asymmetric_delta.py` — new file

**Detailed steps:**
1. Create function:
```python
def estimate_asymmetric_delta(
    intervals_by_match: dict[str, list[IntervalRecord]],
    opt_result: OptimizationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate asymmetric score-state effects from historical intervals.

    Splits intervals by sign of delta_S, fits MLE for each subset.
    Returns (delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg), each shape (5,).
    """
```
2. Logic: partition intervals by `delta_S > 0` (leading) vs `delta_S <= 0` (trailing). For each partition, compute goal rate per delta_S bin. Take log-ratio relative to bin 2 (ΔS=0) as the delta value.
3. If insufficient data for a bin (<20 intervals), fall back to the symmetric `opt_result.delta_H[bin]` for that entry.
4. Clamp values to [-0.5, 0.5] consistent with NLL optimizer constraints.

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/calibration/test_asymmetric_delta.py` — verify output shapes are (5,) each; verify bin 2 (ΔS=0) is near 0.0; verify pos/neg differ for non-zero bins when data supports it.

**Verify:** `make test`

---

## Task 1.6: Create stoppage time η estimation function

**What:** Implement Step 7 of v5 Phase 1: estimate first/second-half stoppage intensity multipliers from historical intervals.

**Files touched:**
- `src/calibration/step_1_7_stoppage_eta.py` — new file

**Detailed steps:**
1. Create function:
```python
def estimate_stoppage_eta(
    intervals_by_match: dict[str, list[IntervalRecord]],
    opt_result: OptimizationResult,
) -> tuple[float, float, float, float]:
    """Estimate stoppage time intensity multipliers.

    Returns (eta_H, eta_A, eta_H2, eta_A2).
    """
```
2. Logic: identify intervals in stoppage periods (t > 45 in first half, t > 90 in second half using `alpha_1` from IntervalRecord). Compare goal rate in stoppage vs non-stoppage. η = log(rate_stoppage / rate_normal).
3. If <50 stoppage intervals, return (0.0, 0.0, 0.0, 0.0) — no adjustment.

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/calibration/test_stoppage_eta.py` — verify returns 4 floats; verify η ≥ 0 when stoppage rate > normal rate.

**Verify:** `make test`

---

## Task 1.7: Create σ²_ω estimation function

**What:** Implement Step 8 of v5 Phase 1: estimate EKF process noise from within-match team strength drift.

**Files touched:**
- `src/calibration/step_1_8_sigma_omega.py` — new file

**Detailed steps:**
1. Create function:
```python
def estimate_sigma_omega_sq(
    intervals_by_match: dict[str, list[IntervalRecord]],
    opt_result: OptimizationResult,
) -> float:
    """Estimate EKF process noise σ²_ω.

    Method: For each match, estimate a_H using first-half goals only vs
    second-half goals only. σ²_ω = Var(Δa) / 45 minutes.
    """
```
2. Logic: for each match, split intervals at halftime. Compute first-half and second-half goal rates → implied a_H for each half. Record drift Δa = a_second - a_first. σ²_ω = Var(Δa across matches) / 45.
3. Clamp to [0.001, 0.1] to avoid degenerate values.
4. If <30 matches, return default 0.01.

**Breaking changes:** None — new file.

**Test impact:**
- Existing tests that break: None
- New tests to add: `tests/calibration/test_sigma_omega.py` — verify returns positive float; verify decreases when matches have consistent strength across halves.

**Verify:** `make test`

---

## Task 1.8: Update _Phase1Result and save_production_params

**What:** Store full gamma/delta arrays and new v5 parameters in the database. Fix the latent scalar-vs-array mismatch between save (scalars) and load (expects arrays).

**Files touched:**
- `src/calibration/phase1_worker.py` — update `_Phase1Result`, `save_production_params`, result construction

**Detailed steps:**
1. Update `_Phase1Result` dataclass (lines 56-72):
```python
# Change from scalar to full arrays:
gamma_H: np.ndarray | None = None   # shape (4,)
gamma_A: np.ndarray | None = None   # shape (4,)
delta_H: np.ndarray | None = None   # shape (5,)
delta_A: np.ndarray | None = None   # shape (5,)
# Add new v5 fields:
delta_H_pos: np.ndarray | None = None  # shape (5,)
delta_H_neg: np.ndarray | None = None
delta_A_pos: np.ndarray | None = None
delta_A_neg: np.ndarray | None = None
eta_H: float = 0.0
eta_A: float = 0.0
eta_H2: float = 0.0
eta_A2: float = 0.0
sigma_omega_sq: float = 0.01
```
2. Update result construction (~lines 192-205) to store full arrays:
```python
gamma_H=best_opt_result.gamma_H,      # full (4,) array
gamma_A=best_opt_result.gamma_A,      # full (4,) array
delta_H=best_opt_result.delta_H,      # full (5,) array
delta_A=best_opt_result.delta_A,      # full (5,) array
```
3. Update `save_production_params` signature (lines 208-253):
   - Change `gamma_H: float` → `gamma_H: np.ndarray`; store as `json.dumps(gamma_H.tolist())`
   - Same for `gamma_A`, `delta_H`, `delta_A`
   - Add new parameters: `delta_H_pos`, `delta_H_neg`, `delta_A_pos`, `delta_A_neg` (np.ndarray, stored as JSON)
   - Add: `eta_H`, `eta_A`, `eta_H2`, `eta_A2` (float)
   - Add: `sigma_omega_sq` (float)
4. Update SQL INSERT to include new columns:
```sql
INSERT INTO production_params
    (league_id, Q, b, gamma_H, gamma_A, delta_H, delta_A,
     delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg,
     eta_H, eta_A, eta_H2, eta_A2, sigma_omega_sq,
     sigma_a, xgb_model_blob, feature_mask, trained_at,
     match_count, brier_score, is_active)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
        $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, TRUE)
RETURNING version
```

**Breaking changes:** `save_production_params` signature changes (gamma/delta now arrays, new params added). Fix included: YES — caller (`run_phase1`) is updated in Task 1.10.

**Test impact:**
- Existing tests that break: None — `test_phase1_worker.py::test_phase1_epl_smoke` is marked `@pytest.mark.slow` and skipped by default. It also requires DB access.
- New tests to add: None (covered in Task 1.11)

**Verify:** `make test`

---

## Task 1.9: Update load_production_params

**What:** Load full arrays and new v5 parameters from database. Backward-compatible with old schema (missing columns get defaults).

**Files touched:**
- `src/prematch/phase2_pipeline.py` — update `load_production_params` (~lines 188-238)

**Detailed steps:**
1. Update SQL query to SELECT new columns (with COALESCE for backward compat):
```sql
SELECT version, league_id, Q, b, gamma_H, gamma_A, delta_H, delta_A,
       COALESCE(delta_H_pos, NULL) as delta_H_pos,
       COALESCE(delta_H_neg, NULL) as delta_H_neg,
       COALESCE(delta_A_pos, NULL) as delta_A_pos,
       COALESCE(delta_A_neg, NULL) as delta_A_neg,
       COALESCE(eta_H, 0.0) as eta_H,
       COALESCE(eta_A, 0.0) as eta_A,
       COALESCE(eta_H2, 0.0) as eta_H2,
       COALESCE(eta_A2, 0.0) as eta_A2,
       COALESCE(sigma_omega_sq, 0.01) as sigma_omega_sq,
       sigma_a, xgb_model_blob, feature_mask, ...
FROM production_params WHERE league_id = $1 AND is_active = TRUE
```
2. Parse new fields: JSON arrays via `json.loads()`, floats directly.
3. If `gamma_H` column is a float (old schema), wrap in array: `[0.0, val, -val, 0.0]`. If JSON string, parse as array. This provides backward compat.
4. Same reconstruction logic for delta_H/delta_A if stored as scalar.

**Breaking changes:** None — backward compatible with old and new DB schemas.

**Test impact:**
- Existing tests that break: None — `test_phase2_pipeline.py::test_load_production_params` requires DB and is skipped.
- New tests to add: None (covered in Task 1.11)

**Verify:** `make test`

---

## Task 1.10: Update _run_calibration to call new estimation steps

**What:** Wire the new estimation functions (Tasks 1.5-1.7) into the Phase 1 calibration pipeline after the NLL optimization.

**Files touched:**
- `src/calibration/phase1_worker.py` — update `_run_calibration` (~lines 75-205) and `run_phase1` (~lines 256-295)

**Detailed steps:**
1. After best NLL optimization result is selected (~line 171), add:
```python
# Step 6: Asymmetric delta estimation
from src.calibration.step_1_6_asymmetric_delta import estimate_asymmetric_delta
delta_H_pos, delta_H_neg, delta_A_pos, delta_A_neg = estimate_asymmetric_delta(
    intervals_by_match, best_opt_result
)

# Step 7: Stoppage time η estimation
from src.calibration.step_1_7_stoppage_eta import estimate_stoppage_eta
eta_H, eta_A, eta_H2, eta_A2 = estimate_stoppage_eta(
    intervals_by_match, best_opt_result
)

# Step 8: EKF process noise estimation
from src.calibration.step_1_8_sigma_omega import estimate_sigma_omega_sq
sigma_omega_sq = estimate_sigma_omega_sq(intervals_by_match, best_opt_result)
```
2. Add new fields to `_Phase1Result` return (~line 197):
```python
delta_H_pos=delta_H_pos, delta_H_neg=delta_H_neg,
delta_A_pos=delta_A_pos, delta_A_neg=delta_A_neg,
eta_H=eta_H, eta_A=eta_A, eta_H2=eta_H2, eta_A2=eta_A2,
sigma_omega_sq=sigma_omega_sq,
```
3. Update `run_phase1` call to `save_production_params` (~lines 269-283) to pass new fields:
```python
delta_H_pos=result.delta_H_pos,
delta_H_neg=result.delta_H_neg,
...
sigma_omega_sq=result.sigma_omega_sq,
```

**Breaking changes:** None — the pipeline produces additional outputs; existing outputs unchanged.

**Test impact:**
- Existing tests that break: None
- New tests to add: None (covered in Task 1.11)

**Verify:** `make test`

---

## Task 1.11: Add Sprint 1 tests

**What:** Add comprehensive tests for all new Sprint 1 functionality.

**Files touched:**
- `tests/calibration/test_asymmetric_delta.py` — new
- `tests/calibration/test_stoppage_eta.py` — new
- `tests/calibration/test_sigma_omega.py` — new
- `tests/test_math_core.py` — add v5 tests

**Detailed steps:**
1. `test_asymmetric_delta.py`:
   - `test_asymmetric_delta_shapes` — output shapes are each (5,)
   - `test_asymmetric_delta_center_bin_near_zero` — bin 2 (ΔS=0) ≈ 0.0
   - `test_asymmetric_delta_fallback_on_insufficient_data` — returns symmetric when <20 intervals per partition

2. `test_stoppage_eta.py`:
   - `test_eta_shapes` — returns 4 floats
   - `test_eta_zero_on_insufficient_data` — returns (0,0,0,0) when <50 stoppage intervals
   - `test_eta_positive_when_stoppage_rate_higher` — η > 0 for synthetic data with higher stoppage goal rate

3. `test_sigma_omega.py`:
   - `test_sigma_omega_positive` — returns positive float
   - `test_sigma_omega_default_on_few_matches` — returns 0.01 when <30 matches
   - `test_sigma_omega_clamped` — result in [0.001, 0.1]

4. `test_math_core.py` additions:
   - `test_mc_simulate_v5_import` — `mc_simulate_remaining_v5` is importable and callable
   - `test_mc_v5_symmetric_matches_old` — when delta_H_pos == delta_H_neg == delta_H, v5 result ≈ old result (same seed)
   - `test_mc_v5_eta_increases_stoppage_goals` — with large η, total goals increase compared to η=0

**Breaking changes:** None

**Test impact:**
- Existing tests that break: None
- ~10 new tests added

**Verify:** `make test`

---

## Sprint 1 Dependency Graph

```
1.1 ─────┬──→ 1.2 ──→ 1.8 ──→ 1.9 ──→ 1.10 ──→ 1.11
         ├──→ 1.3 ──→ 1.4
         ├──→ 1.5 ─┐
         ├──→ 1.6 ─┼──→ 1.8
         └──→ 1.7 ─┘
```
