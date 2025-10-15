
# Parameters
params = {
    'b_H': 1/(70*365),  #Human birth rate (day^(-1))
    'd_H': 1/(70*365), # Human mortality rate (day^(-1))
    'a': 1/(10*365), # Ageing rate from children to adults (day^(-1))
    #'beta_C':0.00009295464, # Effective transmission rate of children (day^(-1))
    #'beta_A': 0.0001928778, # Effective transmission rate of adult (day^(-1))
    'alpha_H': 1/28, # Larvae maturity rate (in-human host) (day^(-1))
    'n_C': 8.5, # Number of Larvae per gram of feces in children (LPG)
    'f_C': 225, # Gram of fecal output per child per day (gram * person^(-1) * (day^(-1)))
    'n_A': 8.5, # Number of Larvae per gram of feces in adults (LPG)
    'f_A': 385, # Gram of fecal output per adult per day (gram * person^(-1) * (day^(-1)))
    'b_D': 1/(5*365), # Dog birth rate (day^(-1))
    'd_D': 1/(5*365),  # Dog mortality rate (day^(-1))
    #'beta_DA': 1, # Effective transmission rate of dog (day^(-1))
    'alpha_D': 1/14,  # Larvae maturity rate (in-Dog host) (day^(-1))
    'n_D': 58, # Number of Larvae per gram of feces in dogs (LPG)
    'f_D': 186,  # Gram of fecal output per dog per day (gram * dog^(-1) * (day^(-1)))
    'K': 1e5, # The number of larvae in the reservoir at which the force of infection is 1/2 of its maximum value (larvae)
    'mu_L': 1/7,  # Larvae mortality rate in the environment (day^(-1))
  
}

# Initial conditions
IC = {
    "S_C": {"name": "constant", "args": {"value": 249}},  # Susceptible Children
    "E_C": {"name": "constant", "args": {"value": 0}},  # Exposed Children
    "I_C": {"name": "constant", "args": {"value": 0}},  # Infectious Children
    "S_A": {"name": "constant", "args": {"value": 1014}},  # Susceptible Adults
    "E_A": {"name": "constant", "args": {"value": 0}},  # Exposed Adults
    "I_A": {"name": "constant", "args": {"value": 0}},  # Infectious Adults
    "S_D": {"name": "constant", "args": {"value": 421}},  # Susceptible Dogs
    "E_DB": {"name": "constant", "args": {"value": 0}},  # Non Zoonotic Exposed Dogs
    "I_DB": {"name": "constant", "args": {"value": 0}},  # Non Zoonotic Infectious Dogs
    "L_Z": {"name": "constant", "args": {"value": 10}},  # Zoonotic Infective Larvae in the Environment
    "L_NZ": {"name": "constant", "args": {"value": 10}},  # Zoonotic Infective Larvae in the Environment
    
}
