import time
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.integrate import solve_ivp

from Model_NZ import model_ode
from Params_and_IC import IC
from Params_and_IC import params as imported_params

# ---- Config -----------------------------------------------------------------
YEARS = 200
TOTAL_DAYS = YEARS * 365
T_EVAL = np.linspace(0, TOTAL_DAYS, TOTAL_DAYS + 1)

BETA_CSV = "LHS_Samples.csv"
OUT_CSV  = "final_derivatives_results.csv"
TOL_PREV = 0.01  # 1% absolute change tolerance

# ---- Helpers ----------------------------------------------------------------
def build_y0(IC_dict):
    """Map ICs to state order expected by model_ode."""
    return np.array([
        IC_dict['S_C']['args']['value'],
        IC_dict['E_C']['args']['value'],
        IC_dict['I_C']['args']['value'],
        IC_dict['S_A']['args']['value'],
        IC_dict['E_A']['args']['value'],
        IC_dict['I_A']['args']['value'],
        IC_dict['S_D']['args']['value'],
        IC_dict['E_DA']['args']['value'],
        IC_dict['I_DA']['args']['value'],
        IC_dict['L_A']['args']['value'],
        
    ], dtype=float)

def run_one(params_dict, y0, t_eval):
    """Integrate once with LSODA; returns SciPy OdeResult."""
    p = SimpleNamespace(**params_dict)
    return solve_ivp(
        fun=lambda t, y: model_ode(t, y, p),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        method="LSODA",
        t_eval=t_eval,
        vectorized=False,
        rtol=1e-6,
        atol=1e-9,
    )

# ---- Main -------------------------------------------------------------------
def main():
    start_all = time.time()

    # 1) Read beta samples
    beta_df = pd.read_csv(BETA_CSV)
    required_cols = {"beta_C", "beta_A", "beta_DA"}
    if not required_cols.issubset(beta_df.columns):
        missing = sorted(required_cols - set(beta_df.columns))
        raise ValueError(f"Missing columns in {BETA_CSV}: {missing}")

    y0_base = build_y0(IC)
    results = []
    prev_year_steps = 365

    # 2) Run each parameter set
    for idx, row in beta_df.iterrows():
        params = imported_params.copy()
        params["beta_C"]  = float(row["beta_C"])
        params["beta_A"]  = float(row["beta_A"])
        params["beta_DA"] = float(row["beta_DA"])

        print(f"[{idx+1}/{len(beta_df)}] "
              f"beta_C={params['beta_C']}, beta_A={params['beta_A']}, beta_DA={params['beta_DA']}")

        t0 = time.time()
        sol = run_one(params, y0_base, T_EVAL)
        solve_time = time.time() - t0

        if not sol.success:
            print(f"  Solver failed: {sol.message}")
            continue

        final_state = sol.y[:, -1]
        final_idx = -1
        prev_idx  = -(prev_year_steps + 1)

        # ---- Prevalence checks ----
        SC, EC, ICv = sol.y[0, :], sol.y[1, :], sol.y[2, :]
        SA, EA, IA = sol.y[3, :], sol.y[4, :], sol.y[5, :]
        SD, EDA, IDA = sol.y[6, :], sol.y[7, :], sol.y[8, :]
        L_A = sol.y[9, :]

        # Children
        tot_children_final = SC[final_idx] + EC[final_idx] + ICv[final_idx]
        tot_children_prev  = SC[prev_idx] + EC[prev_idx] + ICv[prev_idx]
        prev_I_C_final = (ICv[final_idx] / tot_children_final) if tot_children_final > 0 else 0.0
        prev_I_C_prev  = (ICv[prev_idx] / tot_children_prev) if tot_children_prev > 0 else 0.0
        within_C = int(abs(prev_I_C_final - prev_I_C_prev) <= TOL_PREV)

        # Adults
        tot_adults_final = SA[final_idx] + EA[final_idx] + IA[final_idx]
        tot_adults_prev  = SA[prev_idx] + EA[prev_idx] + IA[prev_idx]
        prev_I_A_final = (IA[final_idx] / tot_adults_final) if tot_adults_final > 0 else 0.0
        prev_I_A_prev  = (IA[prev_idx] / tot_adults_prev) if tot_adults_prev > 0 else 0.0
        within_A = int(abs(prev_I_A_final - prev_I_A_prev) <= TOL_PREV)

        # Dogs (lineage A)
        tot_dogs_final = SD[final_idx] + EDA[final_idx] + IDA[final_idx]
        tot_dogs_prev  = SD[prev_idx] + EDA[prev_idx] + IDA[prev_idx]
        prev_I_DA_final = (IDA[final_idx] / tot_dogs_final) if tot_dogs_final > 0 else 0.0
        prev_I_DA_prev  = (IDA[prev_idx] / tot_dogs_prev) if tot_dogs_prev > 0 else 0.0
        within_DA = int(abs(prev_I_DA_final - prev_I_DA_prev) <= TOL_PREV)

        # Larval pools
        within_LA = int(L_A[final_idx] != 0 and abs(L_A[final_idx] - L_A[prev_idx]) <= TOL_PREV * abs(L_A[final_idx]))
       

        # ---- Store results ----
        results.append({
            "Particle": idx + 1,
            "beta_C": params["beta_C"],
            "beta_A": params["beta_A"],
            "beta_DA": params["beta_DA"],
            "prevalence_C_within_1pct": within_C,
            "prevalence_A_within_1pct": within_A,
            "prevalence_DA_within_1pct": within_DA,
            "L_A_within_1pct": within_LA,
    
            "final_prevalence_I_C": prev_I_C_final,
            "final_prevalence_I_A": prev_I_A_final,
            "final_prevalence_I_DA": prev_I_DA_final,
            
            # Save full final state:
            "S_C_final": final_state[0],
            "E_C_final": final_state[1],
            "I_C_final": final_state[2],
            "S_A_final": final_state[3],
            "E_A_final": final_state[4],
            "I_A_final": final_state[5],
            "S_D_final": final_state[6],
            "E_DA_final": final_state[7],
            "I_DA_final": final_state[8],
            "L_A_final_state": final_state[9],
           
        })

    # 3) Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved results → {OUT_CSV}")
    print(f"Total wall time: {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
