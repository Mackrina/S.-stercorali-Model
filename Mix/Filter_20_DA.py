import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('final_derivatives_results_Z_and_NZ.csv')


# Define the range for final_prevalence_I_C and final_prevalence_I_D
prevalence_I_C_range = (0.032, 0.21)
prevalence_I_DA_20_range =  (0.0128, 0.168)
prevalence_I_DB_20_range =  (0.117,0.366)
prevalence_I_D_range = (0.16, 0.43)

# Filter the DataFrame based on the specified ranges
filtered_df = df[
    (df['final_prevalence_I_C'] >= prevalence_I_C_range[0]) & 
    (df['final_prevalence_I_C'] <= prevalence_I_C_range[1]) &
    (df['final_prevalence_I_A'] >= 1.7 * df['final_prevalence_I_C']) &
    (df['final_prevalence_I_A'] <= 2.5 * df['final_prevalence_I_C']) &
    (df['final_prevalence_I_D'] >= prevalence_I_D_range[0]) & 
    (df['final_prevalence_I_D'] <= prevalence_I_D_range[1]) &
    (df['final_prevalence_I_DA'] >= prevalence_I_DA_20_range[0]) &
    (df['final_prevalence_I_DA'] <= prevalence_I_DA_20_range[1]) &
    (df['final_prevalence_I_DB'] >= prevalence_I_DB_20_range[0]) &
    (df['final_prevalence_I_DB'] <= prevalence_I_DB_20_range[1])
    
    
]

# Compute I_D/I_C ratio
filtered_df[ 'ratio'] = filtered_df['final_prevalence_I_D'] / filtered_df['final_prevalence_I_C']

print('num_rows', len(filtered_df))

# Apply ratio filter
filtered_df = filtered_df[(filtered_df['ratio'] >= 1.23) & (filtered_df['ratio'] <=11.45)]

# Print the number of rows after filtering
print('num_rows', len(filtered_df))


# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('Filetred_20.csv', index=False)


print('results saved to filtered_data.csv')
