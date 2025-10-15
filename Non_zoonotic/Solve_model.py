import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from Model import model
from Params_and_IC import IC
from Params_and_IC import params as imported_params
import time

# Read beta values from CSV file
start_time = time.time()
beta_df = pd.read_csv('/media/ubuntu/f3df3264-97ef-43bf-922c-957a3c7d28f4/LHS_samples.csv')
read_time = time.time() - start_time
print(f"Time taken to read beta values: {read_time:.4f} seconds")

# Total duration in years
years = 200
total_days = years * 365

# Time array for evaluation
start_time = time.time()
t_eval = np.linspace(0, total_days, total_days + 1)
generate_time = time.time() - start_time
print(f"Time taken to generate time array: {generate_time:.4f} seconds")

# Model wrapper
def model_wrapper(t, y, params):
    return model(y, t, params)

# Prepare results list
final_derivatives = []

# Tolerances
tolerance_prevalence = 0.01

# Time tracking
start_time = time.time()

for index, row in beta_df.iterrows():
    # Load and override parameters
    params = imported_params.copy()
    params['beta_C'] = row['beta_C']
    params['beta_A'] = row['beta_A']
    params['beta_DB'] = row['beta_DB']
    
    print(f"Running simulation {index + 1}: beta_C={params['beta_C']}, beta_A={params['beta_A']}, beta_DB={params['beta_DB']}")

    # Initial conditions
    y0 = [
        IC['S_C']['args']['value'],
        IC['E_C']['args']['value'],
        IC['I_C']['args']['value'],
        IC['S_A']['args']['value'],
        IC['E_A']['args']['value'],
        IC['I_A']['args']['value'],
        IC['S_D']['args']['value'],
        IC['E_DB']['args']['value'],
        IC['I_DB']['args']['value'],
        IC['L_Z']['args']['value'],
        IC['L_NZ']['args']['value']
    ]
        
    # Solve ODE
    start_time_solver = time.time()
    solution = solve_ivp(
        fun=model_wrapper,
        t_span=(0, total_days),
        y0=y0,
        args=(params,),
        method='LSODA',
        t_eval=t_eval
    )
    solve_time = time.time() - start_time_solver

    if not solution.success:
        print(f"Solver failed for beta values {row['beta_C']}, {row['beta_A']}, {row['beta_DB']}: {solution.message}")
        continue


    # Time indices for final and one year before
    final_index = -1
    prev_year_index = -366  # 365 days before final

    # Prevalence in Children
    total_children_final = sum(solution.y[i, final_index] for i in [0, 1, 2])
    prevalence_I_C_final = solution.y[2, final_index] / total_children_final if total_children_final != 0 else 0

    total_children_prev = sum(solution.y[i, prev_year_index] for i in [0, 1, 2])
    prevalence_I_C_prev = solution.y[2, prev_year_index] / total_children_prev if total_children_prev != 0 else 0

    change_C = abs(prevalence_I_C_final - prevalence_I_C_prev)
    prevalence_C_within_tol = change_C <= tolerance_prevalence

    # Prevalence in Adults
    total_adults_final = sum(solution.y[i, final_index] for i in [3, 4, 5])
    prevalence_I_A_final = solution.y[5, final_index] / total_adults_final if total_adults_final != 0 else 0

    total_adults_prev = sum(solution.y[i, prev_year_index] for i in [3, 4, 5])
    prevalence_I_A_prev = solution.y[5, prev_year_index] / total_adults_prev if total_adults_prev != 0 else 0

    change_A = abs(prevalence_I_A_final - prevalence_I_A_prev)
    prevalence_A_within_tol = change_A <= tolerance_prevalence

    # Prevalence in Dogs
    total_dogs_final = sum(solution.y[i, final_index] for i in [6, 7, 8])
    prevalence_I_DB_final = solution.y[8, final_index] / total_dogs_final if total_dogs_final != 0 else 0

    total_dogs_prev = sum(solution.y[i, prev_year_index] for i in [6, 7, 8])
    prevalence_I_DB_prev = solution.y[8, prev_year_index] / total_dogs_prev if total_dogs_prev != 0 else 0

    change_DB = abs(prevalence_I_DB_final - prevalence_I_DB_prev)
    prevalence_DB_within_tol = change_DB <= tolerance_prevalence

    # Larvae equilibrium check (1% change over 1 year)
    LZ_final = solution.y[9, final_index]
    LZ_prev = solution.y[9, prev_year_index]
    LZ_change_within_tol = abs(LZ_final - LZ_prev) <= tolerance_prevalence * LZ_final if LZ_final != 0 else False

    LNZ_final = solution.y[10, final_index]
    LNZ_prev = solution.y[10, prev_year_index]
    LNZ_change_within_tol = abs(LNZ_final - LNZ_prev) <= tolerance_prevalence * LNZ_final if LNZ_final != 0 else False

    # Store results
    final_derivatives.append({
        'Particle': index + 1,
        'beta_C': row['beta_C'],
        'beta_A': row['beta_A'],
        'beta_DB': row['beta_DB'],
        'prevalence_C_within_1pct': int(prevalence_C_within_tol),
        'prevalence_A_within_1pct': int(prevalence_A_within_tol),
        'prevalence_DB_within_1pct': int(prevalence_DB_within_tol),
        'LZ_within_1pct': int(LZ_change_within_tol),
        'LNZ_within_1pct': int(LNZ_change_within_tol),
        'final_prevalence_I_C': prevalence_I_C_final,
        'final_prevalence_I_A': prevalence_I_A_final,
        'final_prevalence_I_DB': prevalence_I_DB_final,
        'L_Z_final': LZ_final,
        'L_NZ_final': LNZ_final 
        
    })

# Save output
results_df = pd.DataFrame(final_derivatives)
output_file_path = '/media/ubuntu/f3df3264-97ef-43bf-922c-957a3c7d28f4/final_derivatives_results_NZ.csv'
results_df.to_csv(output_file_path, index=False)
print(f"Results saved to {output_file_path}")

# Total time
total_time = time.time() - start_time
print(f"Total time taken for the entire process: {total_time:.4f} seconds")
