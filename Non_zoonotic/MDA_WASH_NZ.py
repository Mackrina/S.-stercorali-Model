# solve_MDA_WASH.py
# Run MDA pulses + WASH intervention using legacy Model_NZ (11-state system)
# State order: [S_C, E_C, I_C, S_A, E_A, I_A, S_D, E_DB, I_DB, L_A, L_B]

import time
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.integrate import solve_ivp

from Model_NZ import model_ode                  
from Params_and_IC import params as base_params
from Params_and_IC import IC

# --------------------------
# File I/O
# --------------------------
EQUIL_CSV = 'final_derivatives_results.csv'
OUT_PREV_CSV = 'model_prevalence_MDA_WASH_NZ.csv'
OUT_STATE_CSV = 'model_population_states_MDA_WASH_NZ.csv'

# --------------------------
# Simulation window
# --------------------------
YEARS = 50
TOTAL_DAYS = YEARS * 365
# daily RHS evaluations (solver still adapts but returns these points)
T_DAILY = np.arange(0, TOTAL_DAYS + 1, 1, dtype=int)  # inclusive of TOTAL_DAYS

# --------------------------
# Interventions
# --------------------------

# Below keeps your original 1y,2y,3y,4y,5y pulses:
MDA_PULSES_DAYS = [365 * i for i in range(1, 6)]

# WASH: start day
WASH_START_DAY = 365  # day 365 (year 1)

# Default WASH reductions if not provided in params:
# Reductions are FRACTIONS (e.g., 0.25 means 25% reduction in that quantity once WASH starts)
DEFAULT_WASH = {
    'wash_beta_red_C': 0.25,  # reduce beta_C
    'wash_beta_red_A': 0.25,  # reduce beta_A
    'wash_f_red_C':   0.00,   # reduce f_C (shedding/contribution)
    'wash_f_red_A':   0.00,   # reduce f_A
}

# --------------------------
# State index map (for clarity)
# --------------------------
IDX = {
    'S_C': 0, 'E_C': 1, 'I_C': 2,
    'S_A': 3, 'E_A': 4, 'I_A': 5,
    'S_D': 6, 'E_DB': 7, 'I_DB': 8,
    'L_A': 9, 'L_B': 10,
}

# ==========================
# Helper: MDA pulse
# ==========================
def apply_mda_pulse(y, params):
    """
    Instantaneous MDA:
      - Humans: move fraction (c_H * eff_H) of E_C and I_C into S_C, and of E_A/I_A into S_A
      - Dogs:   move fraction (c_D * eff_D) of E_DB and I_DB into S_D
    Susceptibles are not changed (treatment has no effect on S).
    """
    y = np.maximum(y, 0.0)

    c_H = float(params.get('c_H', 0.0))
    eff_H = float(params.get('eff_H', 1.0))
    frac_H = max(0.0, min(1.0, c_H * eff_H))

    c_D = float(params.get('c_D', 0.0))
    eff_D = float(params.get('eff_D', 1.0))
    frac_D = max(0.0, min(1.0, c_D * eff_D))

    # Humans: children
    move_EC = frac_H * y[IDX['E_C']]
    move_IC = frac_H * y[IDX['I_C']]
    y[IDX['E_C']] -= move_EC
    y[IDX['I_C']] -= move_IC
    y[IDX['S_C']] += (move_EC + move_IC)

    # Humans: adults
    move_EA = frac_H * y[IDX['E_A']]
    move_IA = frac_H * y[IDX['I_A']]
    y[IDX['E_A']] -= move_EA
    y[IDX['I_A']] -= move_IA
    y[IDX['S_A']] += (move_EA + move_IA)

    # Dogs
    move_EDB = frac_D * y[IDX['E_DB']]
    move_IDB = frac_D * y[IDX['I_DB']]
    y[IDX['E_DB']] -= move_EDB
    y[IDX['I_DB']] -= move_IDB
    y[IDX['S_D']]  += (move_EDB + move_IDB)

    return np.maximum(y, 0.0)

# ==========================
# Wrapper: WASH-adjusted RHS
# ==========================
def rhs_with_wash(t, y, params):
    """
    Model_NZ.model_ode signature: model_ode(t, y, params).
    WASH: after WASH_START_DAY, reduce beta_C/A and/or f_C/A according to params or DEFAULT_WASH.
    """
    y = np.maximum(y, 0.0)

    tmp = params.copy()
    if t >= WASH_START_DAY:
        wash_beta_red_C = tmp.get('wash_beta_red_C', DEFAULT_WASH['wash_beta_red_C'])
        wash_beta_red_A = tmp.get('wash_beta_red_A', DEFAULT_WASH['wash_beta_red_A'])
        wash_f_red_C    = tmp.get('wash_f_red_C',    DEFAULT_WASH['wash_f_red_C'])
        wash_f_red_A    = tmp.get('wash_f_red_A',    DEFAULT_WASH['wash_f_red_A'])

        # reduce acquisition (beta) and/or contribution (f)
        tmp['beta_C'] *= (1.0 - wash_beta_red_C)
        tmp['beta_A'] *= (1.0 - wash_beta_red_A)
        tmp['f_C']    *= (1.0 - wash_f_red_C)
        tmp['f_A']    *= (1.0 - wash_f_red_A)

    # If model_ode expects attribute access (p.beta_C), provide a namespace
    p = SimpleNamespace(**tmp)
    dydt = model_ode(t, y, p)  
    # Non-negativity projection (simple guard)
    return np.where(y + dydt < 0.0, -y, dydt)

# ==========================
# Main
# ==========================
def main():
    t_wall0 = time.time()

    # Load equilibria / starting states
    eq = pd.read_csv(EQUIL_CSV)

    # Prepare output files (append mode with header on first particle)
    write_header = True

    for idx, row in eq.iterrows():
        # Copy base params and set per-particle betas
        params = base_params.copy()
        params['beta_C']  = float(row['beta_C'])
        params['beta_A']  = float(row['beta_A'])
        params['beta_DB'] = float(row['beta_DB'])

        # Build y0 from equilibrium row (matches 11-state order)
        y0 = np.array([
            row['S_C_final'], row['E_C_final'], row['I_C_final'],
            row['S_A_final'], row['E_A_final'], row['I_A_final'],
            row['S_D_final'], row['E_DB_final'], row['I_DB_final'],
            row['L_A_final'], row['L_B_final']
        ], dtype=float)

        print(f"Running Particle {idx+1}/{len(eq)}: "
              f"beta_C={params['beta_C']}, beta_A={params['beta_A']}, beta_DB={params['beta_DB']}")

        # step through each MDA pulse; integrate to pulse, apply, continue
        current_t = 0
        current_state = y0.copy()

        prevalence_rows = []
        state_rows = []

        # Build the sequence of segment ends (all pulses + final horizon)
        segment_ends = list(MDA_PULSES_DAYS) + [TOTAL_DAYS]

        for seg_end in segment_ends:
            # time grid for this segment (daily) — include seg_end so pulse uses state at the pulse time
            t_eval_seg = np.arange(current_t, seg_end + 1, 1, dtype=int)
            if t_eval_seg.size == 0:
                pass
            else:
                sol = solve_ivp(
                    fun=lambda t, y: rhs_with_wash(t, y, params),
                    t_span=(float(current_t), float(seg_end)),
                    y0=current_state,
                    method='LSODA',
                    t_eval=t_eval_seg,
                    vectorized=False,
                    max_step=1.0,
                )

                if not sol.success:
                    print(f"  Solver failed: {sol.message}")
                    break

                # Record every ~30 days
                for j in range(sol.t.size):
                    if (int(sol.t[j]) % 30) != 0:
                        continue
                    state = sol.y[:, j]

                    # Totals (correct indices)
                    tot_children = state[IDX['S_C']] + state[IDX['E_C']] + state[IDX['I_C']]
                    tot_adults   = state[IDX['S_A']] + state[IDX['E_A']] + state[IDX['I_A']]
                    tot_humans   = tot_children + tot_adults
                    tot_dogs     = state[IDX['S_D']] + state[IDX['E_DB']] + state[IDX['I_DB']]

                    # Prevalences
                    prev_I_H  = ((state[IDX['I_C']] + state[IDX['I_A']]) / tot_humans) if tot_humans > 0 else 0.0
                    prev_I_DB = (state[IDX['I_DB']] / tot_dogs) if tot_dogs > 0 else 0.0

                    prevalence_rows.append({
                        'Time_day': float(sol.t[j]),
                        'Particle': idx + 1,
                        'beta_C': params['beta_C'],
                        'beta_A': params['beta_A'],
                        'beta_DB': params['beta_DB'],
                        'prevalence_I_H': prev_I_H,
                        'prevalence_I_DB': prev_I_DB,
                    })

                    state_rows.append({
                        'Time_day': float(sol.t[j]),
                        'Particle': idx + 1,
                        'S_C': state[IDX['S_C']], 'E_C': state[IDX['E_C']], 'I_C': state[IDX['I_C']],
                        'S_A': state[IDX['S_A']], 'E_A': state[IDX['E_A']], 'I_A': state[IDX['I_A']],
                        'S_D': state[IDX['S_D']], 'E_DB': state[IDX['E_DB']], 'I_DB': state[IDX['I_DB']],
                        'L_A': state[IDX['L_A']], 'L_B': state[IDX['L_B']],
                    })

                # advance current state to segment end
                current_state = sol.y[:, -1]

            # Apply MDA pulse at seg_end (unless seg_end == TOTAL_DAYS)
            if seg_end in MDA_PULSES_DAYS:
                current_state = apply_mda_pulse(current_state, params)

            current_t = seg_end

        # Append results to CSVs
        pd.DataFrame(prevalence_rows).to_csv(
            OUT_PREV_CSV, mode='a', header=write_header, index=False
        )
        pd.DataFrame(state_rows).to_csv(
            OUT_STATE_CSV, mode='a', header=write_header, index=False
        )
        write_header = False  # only write headers once

    print(f"Simulation complete. Total time: {time.time() - t_wall0:.2f} s")

if __name__ == "__main__":
    main()
