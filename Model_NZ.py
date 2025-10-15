import numpy as np


def model(y, t, params):
    # Unpacking the state variables from the input
    S_C = y[0] # Susceptible Children
    E_C = y[1] # Exposed Children
    I_C = y[2] # Infectious Children
    S_A = y[3] # Susceptible Adults
    E_A = y[4] # Exposed Adults
    I_A = y[5] # Infectious Adults
    S_D = y[6] # Susceptible Dogs
    E_DB = y[7] # Non-Zoonotic Exposed Dogs
    I_DB = y[8] #Non-Zoonotic Infectious Dogs
    L_Z = y[9] # Zoonotic Infective Larvae in the Environment
    L_NZ = y[10] # Non-Zoonotic Infective Larvae in the Environment
   

    # Unpacking parameters from the input dictionary
    b_H = params['b_H'] # Human birth rate (day^(-1))
    d_H = params['d_H'] # Human mortality rate (day^(-1))
    a = params['a'] # Ageing rate from children to adults (day^(-1))
    beta_C = params['beta_C'] # Effective transmission rate of children (day^(-1))
    beta_A = params['beta_A'] # Effective transmission rate of adult (day^(-1))
    alpha_H =params['alpha_H'] # Larvae maturity rate (in-human host) (day^(-1))
    n_C = params['n_C'] # Number of Larvae per gram of feces in children (LPG)
    f_C = params['f_C'] # Gram of fecal output per child per day (gram * person^(-1) * (day^(-1)))
    n_A = params['n_A'] # Number of Larvae per gram of feces in adults (LPG)
    f_A = params['f_A'] # Gram of fecal output per adult per day (gram * person^(-1) * (day^(-1)))
    b_D = params['b_D'] # Dog birth rate (day^(-1))
    d_D = params['d_D']  # Dog mortality rate (day^(-1))
    beta_DB = params['beta_DB'] # Effective transmission rate of dog (day^(-1))
    alpha_D = params['alpha_D'] # Larvae maturity rate (in-Dog host) (day^(-1))
    n_D = params['n_D']  # Number of Larvae per gram of feces in dogs (LPG)
    f_D = params['f_D']  # Gram of fecal output per dog per day (gram * dog^(-1) * (day^(-1)))
    K = params['K'] # The number of larvae in the reservoir at which the force of infection is 1/2 of its maximum value (larvae)
    mu_L =params['mu_L'] # Larvae mortality rate in the environment (day^(-1))



    # Host population sizes
    Total_Children = S_C + E_C + I_C #Suceptible Children + Exposed Children + Infectious Children
    Total_Adults = S_A + E_A + I_A #Suceptible Adults + Exposed Adults + Infectious Adults
    Total_Humans = S_C + E_C + I_C + S_A + E_A + I_A #Suceptible Children + Exposed Children + Infectious Children + Suceptible Adults + Exposed Adults + Infectious Adults
    Total_Dogs = S_D + E_DB + I_DB  #Suceptible Dogs + Exposed Dogs with Non Zoonotic Larvae + Infectious Dogs with Non Zoonotic Larvae 

    # Human birth
    human_births = b_H * Total_Humans #Birth rate of Humans * Total Dogs
    # Human death
    human_deaths_S_C = d_H * S_C #Death rate of Humans * Suceptible Children
    human_deaths_E_C = d_H * E_C #Death rate of Humans * Exposed Children
    human_deaths_I_C = d_H * I_C #Death rate of Humans * Infectious Children
    human_deaths_S_A = d_H * S_A #Death rate of Humans * Suceptible Adults
    human_deaths_E_A = d_H * E_A #Death rate of Humans * Exposed Adults
    human_deaths_I_A = d_H * I_A #Death rate of Humans * Infectious Adults

    # Dog birth
    dog_births = b_D * Total_Dogs #Birth rate of Dogs * Total Dogs
    # Dog death
    dog_deaths_S_D = d_D * S_D #Death rate of Dogs * Suceptible Dogs
    dog_deaths_E_DB = d_D * E_DB #Death rate of Dogs * Exposed Dogs with Zoonotic Larvae
    dog_deaths_I_DB = d_D * I_DB #Death rate of Dogs * Infectious Dogs with Zoonotic Larvae

    # Ageing from children to adults
    Ageing_S = a * S_C #aging rate from kids to adult * Suceptible Children
    Ageing_E = a * E_C #aging rate from kids to adult * Exposed Children
    Ageing_I = a * I_C #aging rate from kids to adult * Infectious Children

    # Larvae death rates in the environment
    larvae_deaths_env_Z = mu_L * L_Z #Larvae death rate in the environment * Zoonotic Larvae
    larvae_deaths_env_NZ = mu_L * L_NZ #Larvae death rate in the environment * Non-Zoonotic Larvae

    # Force of infection (lambda)
    lambda_C = beta_C * L_Z / (K + L_Z) #Effective transmission rate of children * Zoonotic Larvae / (CC + Zoonotic Larvae)
    lambda_A = beta_A * L_Z / (K + L_Z) #Effective transmission rate of adult * Zoonotic Larvae / (CC + Zoonotic Larvae)
    lambda_DB = beta_DB * L_NZ / (K + L_NZ) #Effective transmission rate of dogs * Zoonotic Larvae / (CC + Zoonotic Larvae)
    
    # Infection rates
    Children_infections = lambda_C * S_C #FoI of children * Susceptible Children
    Adult_infections = lambda_A * S_A #FoI of adults * Susceptible adults
    dog_infections_DB = lambda_DB * S_D #FoI of dogs with zoonotic larvae * Susceptible dogs
    

    # Progression from exposed to infectious
    infectious_Children = alpha_H * E_C #Larvae maturity rate (in-human host) * Exposed Children
    infectious_Adult = alpha_H * E_A #Larvae maturity rate (in-human host) * Exposed Adults
    infectious_Dogs_DB = alpha_D * E_DB #Larvae maturity rate (in-dog host) * Exposed dogs with zoonotic larvae
   

    # Contribution to environmental larvae
    human_larvae_output_C = n_C * f_C * I_C #Number of Larvae per gram of feces in children * Gram of fecal output per child per day * Infectious Children
    human_larvae_output_A = n_A * f_A * I_A #Number of Larvae per gram of feces in adults * Gram of fecal output per adult per day * Infectious adult
    dog_larvae_output_DB = n_D * f_D * I_DB #Number of Larvae per gram of feces in dogs * Gram of fecal output per dog per day * Infectious dog
    
    # Differential equations
    dS_C = human_births - Children_infections - Ageing_S - human_deaths_S_C
    dE_C = Children_infections - Ageing_E - infectious_Children - human_deaths_E_C
    dI_C = infectious_Children - Ageing_I - human_deaths_I_C
    dS_A = Ageing_S - Adult_infections - human_deaths_S_A
    dE_A = Adult_infections + Ageing_E - infectious_Adult - human_deaths_E_A
    dI_A = infectious_Adult + Ageing_I - human_deaths_I_A
    dS_D = dog_births - dog_deaths_S_D - dog_infections_DB 
    dE_DB = dog_infections_DB - dog_deaths_E_DB - infectious_Dogs_DB
    dI_DB = infectious_Dogs_DB - dog_deaths_I_DB
    dL_Z = human_larvae_output_C + human_larvae_output_A - larvae_deaths_env_Z
    dL_NZ = dog_larvae_output_DB - larvae_deaths_env_NZ
 
 

    return np.array([dS_C, dE_C, dI_C, dS_A, dE_A, dI_A, dS_D, dE_DB, dI_DB, dL_Z, dL_NZ])
