"""

Parameter dictionary (`params`) and initial conditions (`IC`) for the
human–dog transmission model with two larval lineages.

This file is intentionally simple (plain dicts) so other scripts can
`import params, IC` without extra dependencies or types.
"""

# ---------------------------
# Parameters (per day unless noted)
# ---------------------------
params = {
    # Human demography
    'b_H': 1 / (70 * 365),   # human birth rate
    'd_H': 1 / (70 * 365),   # human mortality rate
    'a'  : 1 / (10 * 365),   # ageing rate (children -> adults)

    # Transmission (beta values typically overridden by LHS_Samples.csv in solve.py)
    # 'beta_C': 0.00009295464,   # children contact rate
    # 'beta_A': 0.0001928778,    # adults contact rate
    # 'beta_DA': 1.0,            # dogs contact rate
    # 'beta_DB': 1.0,            # dogs contact rate
    'K': 1e5,                   # half-saturation larvae density (larvae)

    # Progression
    'alpha_H': 1 / 28,          # larvae maturity in humans
    'alpha_D': 1 / 14,          # larvae maturity in dogs

    # Fecal larval output
    'n_C': 8.5,  'f_C': 225,    # children: larvae per gram, grams/day
    'n_A': 8.5,  'f_A': 385,    # adults
    'n_D': 58,   'f_D': 186,    # dogs

    # Dog demography
    'b_D': 1 / (5 * 365),       # dog birth rate
    'd_D': 1 / (5 * 365),       # dog mortality rate

    # Environment
    'mu_L': 1 / 7,              # larvae mortality in environment

    'c_H' : 1, # effective coverage of mass drug administration in humans 
    'c_D' : 1, # effective coverage of mass drug administration in dogs
}

# ---------------------------
# Initial conditions (state order used by model_ode):
# [S_C, E_C, I_C, S_A, E_A, I_A, S_D, E_DB, I_DB, L_A, L_B]
# ---------------------------
IC = {
    "S_C":  {"name": "constant", "args": {"value": 249}},   # Susceptible Children
    "E_C":  {"name": "constant", "args": {"value": 0}},     # Exposed Children
    "I_C":  {"name": "constant", "args": {"value": 0}},     # Infectious Children

    "S_A":  {"name": "constant", "args": {"value": 1014}},  # Susceptible Adults
    "E_A":  {"name": "constant", "args": {"value": 0}},     # Exposed Adults
    "I_A":  {"name": "constant", "args": {"value": 0}},     # Infectious Adults

    "S_D":  {"name": "constant", "args": {"value": 421}},   # Susceptible Dogs
    "E_DA": {"name": "constant", "args": {"value": 0}},     # Exposed Dogs (Lineage A)
    "I_DA": {"name": "constant", "args": {"value": 0}},     # Infectious Dogs (Lineage A)
    "E_DB": {"name": "constant", "args": {"value": 0}},     # Exposed Dogs (Lineage B)
    "I_DB": {"name": "constant", "args": {"value": 0}},     # Infectious Dogs (Lineage B)

    "L_A":  {"name": "constant", "args": {"value": 10}},    # Environmental larvae — Lineage A
    "L_B":  {"name": "constant", "args": {"value": 10}},    # Environmental larvae — Lineage B
}
