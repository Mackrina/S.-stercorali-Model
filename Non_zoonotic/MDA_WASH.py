import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from Model_NZ import model
from Params_and_IC_MDA import IC
from Params_and_IC_MDA import params as imported_params
import time

# Start timing
time_start = time.time()

# Read equilibrium file
final_results_df = pd.read_csv(
    '/media/ubuntu/f3df3264-97ef-43bf-922c-957a3c7d28f41/NZ/final_derivative_200_years_NZ_95.csv')

# Simulation settings
years = 50
total_days = years * 365
t_eval = np.arange(0, total_days, 1)  # Daily output

# Define MDA schedule (semiannual for 2.5 years)
mda_times_days = [365 * i for i in range(1, 6)]

# Define WASH intervention start day
WASH_START_DAY = 365  # start at 1 year

# --- MDA Treatment Function ---
def apply_mda(y, params):
    y = np.maximum(y, 0)

    # Human MDA
    
    e_coverage_H = params['c_H']
    total_H = sum(y[IDX[key]] for key in ['S_C', 'S_A', 'E_C', 'E_A', 'I_C', 'I_A'])

    if total_H > 0:
        proportions = {key: y[IDX[key]] / total_H for key in ['S_C', 'S_A', 'E_C', 'E_A', 'I_C', 'I_A']}
    else:
        proportions = {key: 0 for key in ['S_C', 'S_A', 'E_C', 'E_A', 'I_C', 'I_A']}

    treated_H = total_H * e_coverage_H

    for key in ['S_C', 'S_A', 'E_C', 'E_A', 'I_C', 'I_A']:
        y[IDX[key]] -= treated_H * proportions[key]

    y[IDX['T_C']] += treated_H * (proportions['S_C'] + proportions['E_C'] + proportions['I_C'])
    y[IDX['T_A']] += treated_H * (proportions['S_A'] + proportions['E_A'] + proportions['I_A'])

    # Dog MDA
    
    e_coverage_D = params['c_D']

    total_D = y[IDX['S_D']] + y[IDX['E_DB']] + y[IDX['I_DB']]
    treated_D = total_D * e_coverage_D

    prop_S_D = y[IDX['S_D']] / total_D if total_D > 0 else 1
    
    prop_E_DB = y[IDX['E_DB']] / total_D if total_D > 0 else 1
    
    prop_I_DB = y[IDX['I_DB']] / total_D if total_D > 0 else 1

    # Apply treatment to dogs
    treated_S_D = treated_D * prop_S_D
    
    treated_E_DB = treated_D * prop_E_DB
    
    treated_I_DB = treated_D * prop_I_DB

    y[IDX['S_D']] +=  treated_E_DB + treated_I_DB  # move treated dogs to S_D
    
    y[IDX['E_DB']] -= treated_E_DB
    
    y[IDX['I_DB']] -= treated_I_DB

    return np.maximum(y, 0)

# --- ODE Wrapper with WASH ---
def model_wrapper(t, y, params):
    temp_params = params.copy()

    if t >= WASH_START_DAY:
        # Years since WASH started
        years_since_start = (t - WASH_START_DAY) / 365.0

        # Cap years_since_start at 7
        effective_years = min(years_since_start, 9)

        # Calculate capped dynamic reduction
        base_reduction = 0.1
        increase_per_year = 0.1
        dynamic_reduction = base_reduction + increase_per_year * effective_years  # Max is 0.8 but capped at 0.7 below
        dynamic_reduction = min(dynamic_reduction, 0.8)

        # Apply reduction to parameters
        temp_params['beta_C'] *= (1 - dynamic_reduction)
        temp_params['beta_A'] *= (1 - dynamic_reduction)
        temp_params['f_C'] *= (1 - dynamic_reduction)
        temp_params['f_A'] *= (1 - dynamic_reduction)

    dydt = model(t, np.maximum(y, 0), temp_params)
    return np.where(y + dydt < 0, -y, dydt)

# --- Main Simulation Loop ---
for index, row in final_results_df.iterrows():
    params = imported_params.copy()
    params['beta_C'] = row['beta_C']
    params['beta_A'] = row['beta_A']
    
    params['beta_DB'] = row['beta_DB']

    y0 = [
        row['S_C_final'], row['E_C_final'], row['I_C_final'], 0,
        row['S_A_final'], row['E_A_final'], row['I_A_final'], 0,
        row['S_D_final'], row['E_DB_final'], row['I_DB_final'],
        row['L_Z_final'], row['L_NZ_final']
    ]

    print(f"Running Particle {index + 1}")
    current_t = 0
    current_state = y0
    prevalence_data = []
    population_states_data = []

    for next_t in mda_times_days + [total_days]:
        sol = solve_ivp(
            fun=model_wrapper,
            t_span=(current_t, next_t),
            y0=current_state,
            args=(params,),
            method='LSODA',
            t_eval=np.arange(current_t, next_t, 1),
            max_step=1
        )

        for i in range(len(sol.t)):
            if int(sol.t[i]) % 30 != 0:
                continue
            state = sol.y[:, i]
            total_children = state[0] + state[1] + state[2] + state[3]
            total_adults = state[4] + state[5] + state[6] + state[7]
            total_humans = total_children + total_adults
            total_dogs = state[8] + state[9] +  state[10] 

            prevalence_data.append({
                'Time_day': sol.t[i],
                'Particle': index + 1,
                'prevalence_I_H': (state[2] + state[6]) / total_humans if total_humans > 0 else 0,
                
                'prevalence_I_DB': state[10] / total_dogs if total_dogs > 0 else 0,
                
            })

            population_states_data.append({
                'Time_day': sol.t[i],
                'Particle': index + 1,
                'S_C': state[0], 'E_C': state[1], 'I_C': state[2], 'T_C': state[3],
                'S_A': state[4], 'E_A': state[5], 'I_A': state[6], 'T_A': state[7],
                'S_D': state[8], 'E_DB': state[9], 'I_DB': state[10],
            })

        if next_t != total_days:
            current_state = apply_mda(sol.y[:, -1], params)
        else:
            current_state = sol.y[:, -1]

        current_t = next_t

    base_path = '/media/ubuntu/f3df3264-97ef-43bf-922c-957a3c7d28f41/WASH_NZ/'
    pd.DataFrame(prevalence_data).to_csv(
        base_path + 'model_prevalence_DW_NZ_P.csv',
        mode='a', header=(index == 0), index=False
    )
    pd.DataFrame(population_states_data).to_csv(
        base_path + 'model_population_states_DW_NZ_P.csv',
        mode='a', header=(index == 0), index=False
    )

print(f"Simulation complete. Total time: {time.time() - time_start:.2f} seconds.")
