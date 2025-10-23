import pandas as pd
import numpy as np


# Load CSV file
df = pd.read_csv('final_derivatives_results_NZ.csv') # upload model outputs

# Define the range for final_prevalence_I_C and final_prevalence_I_D
prevalence_I_C_range = (0.032, 0.21) #95% CI of prevalence in children
prevalence_I_DB_range = (0.16, 0.43) #95% CI of prevalence in dogs

# Filter the DataFrame based on the specified ranges
df_filtered = df[
    (df['final_prevalence_I_C'] >= prevalence_I_C_range[0]) & 
    (df['final_prevalence_I_C'] <= prevalence_I_C_range[1]) &
    (df['final_prevalence_I_A'] >= 1.7 * df['final_prevalence_I_C']) &
    (df['final_prevalence_I_A'] <= 2.5 * df['final_prevalence_I_C']) &
    (df['final_prevalence_I_DB'] >= prevalence_I_DB_range[0]) &
    (df['final_prevalence_I_DB'] <= prevalence_I_DB_range[1])
]
# Compute I_D/I_C ratio
df_filtered[ 'ratio'] = df_filtered['final_prevalence_I_DB'] / df_filtered['final_prevalence_I_C']

print('num_rows', len(df_filtered))

# Apply ratio filter
df_filtered = df_filtered[(df_filtered['ratio'] >= 1.23) & (df_filtered['ratio'] <= 11.45)]

# Print the number of rows after filtering
print('num_rows', len(df_filtered))

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('filtered_data_NZ.csv', index=False)

print('Results saved to filtered_data_NZ.csv')
