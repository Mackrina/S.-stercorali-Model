import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from model_NZ import model

# Initial conditions + params module you already have
from Params_and_IC import IC
from Params_and_IC import params as imported_params

# ---- Config -----------------------------------------------------------------

YEARS = 200                               # integration horizon
TOTAL_DAYS = YEARS * 365                  # days
T_EVAL = np.linspace(0, TOTAL_DAYS, TOTAL_DAYS + 1)  # daily output

BETA_CSV = "LHS_Samples.csv"
OUT_CSV  = "final_derivatives_results.csv"
TOL_PREV = 0.01                           # 1% absolute change tolerance

# ---- Helpers ----------------------------------------------------------------

def build_y0(IC_dict):
    """
    Map ICs to state order expected by model_ode:
    [S_C, E_C, I_C, S_A, E_A, I_A, S_D, E_DB, I_DB, L_A, L_B]
    """
    return [
        IC_dict['S_C']['args']['value'],
        IC_dict['E_C']['args']['value'],
        IC_dict['I_C']['args']['value'],
        IC_dict['S_A']['args']['value'],
        IC_dict['E_A']['args']['value'],
        IC_dict['I_A']['args']['value'],
        IC_dict['S_D']['args']['value'],
        IC_dict['E_DB']['args']['value'],
        IC_dict['I_DB']['args']['value'],
        IC_dict['L_A']['args']['value'], 
        IC_dict['L_B']['args']['value'],  
    ]

def run_one(params, y0, t_eval):
    """Integrate once with LSODA; returns SciPy OdeResult."""
    return solve_ivp(
        fun=lambda t, y: model_ode(t, y, params),
        t_span=(0.0, float(t_eval[-1])),
        y0=np.asarray(y0, dtype=float),
        method="LSODA",
        t_eval=t_eval,
        vectorized=False
    )

# ---- Main -------------------------------------------------------------------

def main():
    start_all = time.time()

    # 1) Read beta samples
    beta_df = pd.read_csv(BETA_CSV)
    required_cols = {"beta_C", "beta_A", "beta_DB"}
    if not required_cols.issubset(beta_df.columns):
        missing = sorted(required_cols - set(beta_df.columns))
        raise ValueError(f"Missing columns in {BETA_CSV}: {missing}")

    # 2) Build base y0
    y0_base = build_y0(IC)

    results = []
    prev_year_steps = 365
    if len(T_EVAL) < (prev_year_steps + 1):
        raise ValueError("Time horizon must be at least 1+ years to check 1-year change.")

    # 3) Sweep over beta samples
    for idx, row in beta_df.iterrows():
        # params is a dict (from your existing module)
        params = imported_params.copy()
        params["beta_C"]  = float(row["beta_C"])
        params["beta_A"]  = float(row["beta_A"])
        params["beta_DB"] = float(row["beta_DB"])

        print(f"[{idx+1}/{len(beta_df)}] "
              f"beta_C={params['beta_C']}, beta_A={params['beta_A']}, beta_DB={params['beta_DB']}")

        t0 = time.time()
        sol = run_one(params, y0_base, T_EVAL)
        dt = time.time() - t0

        if not sol.success:
            print(f"  Solver failed: {sol.message}")
            continue

        # Indices
        final_idx = -1
        prev_idx  = - (prev_year_steps + 1)

        # Prevalence in children
        SC, EC, ICv = sol.y[0, :], sol.y[1, :], sol.y[2, :]
        tot_children_final = SC[final_idx] + EC[final_idx] + ICv[final_idx]
        tot_children_prev  = SC[prev_idx]  + EC[prev_idx]  + ICv[prev_idx]
        prev_I_C_final = (ICv[final_idx] / tot_children_final) if tot_children_final > 0 else 0.0
        prev_I_C_prev  = (ICv[prev_idx]  / tot_children_prev)  if tot_children_prev  > 0 else 0.0
        change_C = abs(prev_I_C_final - prev_I_C_prev)
        within_C = int(change_C <= TOL_PREV)

        # Prevalence in adults
        SA, EA, IA = sol.y[3, :], sol.y[4, :], sol.y[5, :]
        tot_adults_final = SA[final_idx] + EA[final_idx] + IA[final_idx]
        tot_adults_prev  = SA[prev_idx]  + EA[prev_idx]  + IA[prev_idx]
        prev_I_A_final = (IA[final_idx] / tot_adults_final) if tot_adults_final > 0 else 0.0
        prev_I_A_prev  = (IA[prev_idx]  / tot_adults_prev)  if tot_adults_prev  > 0 else 0.0
        change_A = abs(prev_I_A_final - prev_I_A_prev)
        within_A = int(change_A <= TOL_PREV)

        # Prevalence in dogs (lineage B)
        SD, EDB, IDB = sol.y[6, :], sol.y[7, :], sol.y[8, :]
        tot_dogs_final = SD[final_idx] + EDB[final_idx] + IDB[final_idx]
        tot_dogs_prev  = SD[prev_idx]  + EDB[prev_idx]  + IDB[prev_idx]
        prev_I_DB_final = (IDB[final_idx] / tot_dogs_final) if tot_dogs_final > 0 else 0.0
        prev_I_DB_prev  = (IDB[prev_idx]  / tot_dogs_prev)  if tot_dogs_prev  > 0 else 0.0
        change_DB = abs(prev_I_DB_final - prev_I_DB_prev)
        within_DB = int(change_DB <= TOL_PREV)

        # Larvae equilibrium (A & B)
        L_A_final, L_A_prev = sol.y[9, final_idx],  sol.y[9, prev_idx]
        L_B_final, L_B_prev = sol.y[10, final_idx], sol.y[10, prev_idx]
        within_LA = int(L_A_final != 0 and abs(L_A_final - L_A_prev) <= TOL_PREV * abs(L_A_final))
        within_LB = int(L_B_final != 0 and abs(L_B_final - L_B_prev) <= TOL_PREV * abs(L_B_final))

        results.append({
            "Particle": idx + 1,
            "beta_C": params["beta_C"],
            "beta_A": params["beta_A"],
            "beta_DB": params["beta_DB"],
            "prevalence_C_within_1pct": within_C,
            "prevalence_A_within_1pct": within_A,
            "prevalence_DB_within_1pct": within_DB,
            "L_A_within_1pct": within_LA,
            "L_B_within_1pct": within_LB,
            "final_prevalence_I_C": prev_I_C_final,
            "final_prevalence_I_A": prev_I_A_final,
            "final_prevalence_I_DB": prev_I_DB_final,
            "L_A_final": L_A_final,
            "L_B_final": L_B_final,
            "solve_seconds": dt,
        })

    # 4) Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved results → {OUT_CSV}")
    print(f"Total wall time: {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()
