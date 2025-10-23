"""
Model for mix transmission with two larval lineages. This file only contains the model structure — the numerical solver is handled
in a separate script solve_Z_NZ.py.

This module provides:
- A `Params` dataclass containing all model parameters with units documented.
- A `model_ode` function (right-hand side of the ODE system).

Usage (CLI):
    $ python model.py

Dependencies:
    numpy, pandas, scipy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    """
    Model parameters.

    All rates are per day unless otherwise specified.

    Human demography:
        b_H : float  Human birth rate (day^-1)
        d_H : float  Human mortality rate (day^-1)
        a   : float  Ageing rate from children to adults (day^-1)

    Transmission (force of infection uses saturating term L/(K+L)):
        beta_C  : float  Effective contact rate for children (day^-1)
        beta_A  : float  Effective contact rate for adults (day^-1)
        beta_DA : float  Effective contact rate for dogs via lineage A (day^-1)
        beta_DB : float  Effective contact rate for dogs (day^-1)
        K       : float  Half-saturation (larval density at half max FOI) (larvae)

    Progression:
        alpha_H : float  Larval maturity rate in humans (day^-1)
        alpha_D : float  Larval maturity rate in dogs (day^-1)

    Fecal larval output:
        n_C : float  Larvae per gram (children) (LPG)
        f_C : float  Grams fecal output per child per day (g person^-1 day^-1)
        n_A : float  Larvae per gram (adults) (LPG)
        f_A : float  Grams fecal output per adult per day (g person^-1 day^-1)
        n_D : float  Larvae per gram (dogs) (LPG)
        f_D : float  Grams fecal output per dog per day (g dog^-1 day^-1)

    Dog demography:
        b_D : float  Dog birth rate (day^-1)
        d_D : float  Dog mortality rate (day^-1)

    Environmental larvae:
        mu_L : float  Larval mortality in the environment (day^-1)

    Notes on coupling:
        - Humans are infected by lineage A larvae (L_A).
        - Dogs are infected by lineage B larvae (L_B).
        - L_A accrues from infectious humans (children + adults).
        - L_B accrues from infectious dogs.
    """
    # Human demography
    b_H: float
    d_H: float
    a: float

    # Transmission
    beta_C: float
    beta_A: float
    beta_DA: float
    beta_DB: float
    K: float

    # Progression
    alpha_H: float
    alpha_D: float

    # Fecal larval output
    n_C: float
    f_C: float
    n_A: float
    f_A: float
    n_D: float
    f_D: float

    # Dog demography
    b_D: float
    d_D: float

    # Environmental larvae
    mu_L: float


# --------------------------------------------------------------------------------------
# ODE right-hand side
# --------------------------------------------------------------------------------------

def model_ode(
    t: float,
    y: np.ndarray,
    p: Params,
) -> np.ndarray:
    """
    Right-hand side of the ODE system dy/dt = f(t, y; p).

    State vector y (length 11):
        0  S_C   Susceptible Children
        1  E_C   Exposed Children
        2  I_C   Infectious Children
        3  S_A   Susceptible Adults
        4  E_A   Exposed Adults
        5  I_A   Infectious Adults
        6  S_D   Susceptible Dogs
        7  E_DA  Exposed Dogs (Lineage A)
        8  I_DA  Infectious Dogs (Lineage A)
        9  E_DB  Exposed Dogs (Lineage B)
        10  I_DB  Infectious Dogs (Lineage B)
        11  L_A   Environmental larvae (Lineage A)
        12  L_B   Environmental larvae (Lineage B)

    Coupling (corrected and consistent):
        lambda_C  = beta_C  * L_A / (K + L_A)  # children FOI
        lambda_A = beta_A  * L_A / (K + L_A)  # adults FOI
        lambda_DA = beta_DA * L_A / (K + L_A)  # dogs_A FOI
        lambda_DB = beta_DB * L_B / (K + L_B)  # dogs_B FOI

    Returns:
        dydt: np.ndarray of shape (11,)
    """
    # Unpack state
    S_C, E_C, I_C, S_A, E_A, I_A, S_D, E_DA, I_DA, E_DB, I_DB, L_A, L_B = y

    # Totals
    Total_Children = S_C + E_C + I_C
    Total_Adults = S_A + E_A + I_A
    Total_Humans = Total_Children + Total_Adults
    Total_Dogs = S_D + E_DA + I_DA + E_DB + I_DB

    # Births & deaths (humans)
    human_births = p.b_H * Total_Humans
    human_deaths_S_C = p.d_H * S_C
    human_deaths_E_C = p.d_H * E_C
    human_deaths_I_C = p.d_H * I_C
    human_deaths_S_A = p.d_H * S_A
    human_deaths_E_A = p.d_H * E_A
    human_deaths_I_A = p.d_H * I_A

    # Births & deaths (dogs)
    dog_births = p.b_D * Total_Dogs
    dog_deaths_S_D = p.d_D * S_D
    dog_deaths_E_DA = p.d_D * E_DA
    dog_deaths_I_DA = p.d_D * I_DA
    dog_deaths_E_DB = p.d_D * E_DB
    dog_deaths_I_DB = p.d_D * I_DB

    # Ageing (children -> adults)
    Ageing_S = p.a * S_C
    Ageing_E = p.a * E_C
    Ageing_I = p.a * I_C

    # Environmental larval mortality
    larvae_deaths_env_A = p.mu_L * L_A
    larvae_deaths_env_B = p.mu_L * L_B

    # Forces of infection (saturating larval pressure)
    lambda_C = p.beta_C * L_A / (p.K + L_A) 
    lambda_A = p.beta_A * L_A / (p.K + L_A) 
    lambda_DA = p.beta_DA * L_A / (p.K + L_A) 
    lambda_DB = p.beta_DB * L_B / (p.K + L_B) 

    # New infections
    Children_infections = lambda_C * S_C
    Adult_infections = lambda_A * S_A
    dog_infections_DA = lambda_DA * S_D
    dog_infections_DB = lambda_DB * S_D

    # Progression exposed -> infectious
    infectious_Children = p.alpha_H * E_C
    infectious_Adult = p.alpha_H * E_A
    infectious_Dogs_DA = p.alpha_D * E_DA
    infectious_Dogs_DB = p.alpha_D * E_DB

    # Larval output into the environment
    human_larvae_output_C = p.n_C * p.f_C * I_C
    human_larvae_output_A = p.n_A * p.f_A * I_A
    dog_larvae_output_DA = p.n_D * p.f_D * I_DA
    dog_larvae_output_DB = p.n_D * p.f_D * I_DB

    # ODEs: humans
    dS_C = human_births - Children_infections - Ageing_S - human_deaths_S_C
    dE_C = Children_infections - Ageing_E - infectious_Children - human_deaths_E_C
    dI_C = infectious_Children - Ageing_I - human_deaths_I_C

    dS_A = Ageing_S - Adult_infections - human_deaths_S_A
    dE_A = Adult_infections + Ageing_E - infectious_Adult - human_deaths_E_A
    dI_A = infectious_Adult + Ageing_I - human_deaths_I_A

    # ODEs: dogs
    dS_D = dog_births - dog_deaths_S_D - dog_infections_DB
    dE_DA = dog_infections_DA - dog_deaths_E_DA - infectious_Dogs_DA
    dI_DA = infectious_Dogs_DA - dog_deaths_I_DA
    dE_DB = dog_infections_DB - dog_deaths_E_DB - infectious_Dogs_DB
    dI_DB = infectious_Dogs_DB - dog_deaths_I_DB

    # ODEs: environmental larvae
    dL_A = human_larvae_output_C + human_larvae_output_A + dog_larvae_output_DA - larvae_deaths_env_A
    dL_B = dog_larvae_output_DB - larvae_deaths_env_B

    return np.array([dS_C, dE_C, dI_C, dS_A, dE_A, dI_A, dS_D, dE_DA, dI_DA, dE_DB, dI_DB, dL_A, dL_B], dtype=float)
