"""
Model for non-zoonotic transmission with two larval lineages. This file only contains the model structure — the numerical solver is handled
in a separate script (e.g., solve_model.py).

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
        7  E_DB  Exposed Dogs (Lineage B)
        8  I_DB  Infectious Dogs (Lineage B)
        9  L_A   Environmental larvae (Lineage A)
        10 L_B   Environmental larvae (Lineage B)

    Coupling (corrected and consistent):
        lambda_C  = beta_C  * L_A / (K + L_A)  # children FOI
        lambda_AH = beta_A  * L_A / (K + L_A)  # adults FOI
        lambda_DB = beta_DB * L_B / (K + L_B)  # dogs FOI

    Returns:
        dydt: np.ndarray of shape (11,)
    """
    # Unpack state
    S_C, E_C, I_C, S_A, E_A, I_A, S_D, E_DB, I_DB, L_A, L_B = y

    # Totals
    Total_Children = S_C + E_C + I_C
    Total_Adults = S_A + E_A + I_A
    Total_Humans = Total_Children + Total_Adults
    Total_Dogs = S_D + E_DB + I_DB

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
    lambda_C = p.beta_C * L_A / (p.K + L_A) if (p.K + L_A) 
    lambda_AH = p.beta_A * L_A / (p.K + L_A) if (p.K + L_A) 
    lambda_DB = p.beta_DB * L_B / (p.K + L_B) if (p.K + L_B)

    # New infections
    Children_infections = lambda_C * S_C
    Adult_infections = lambda_AH * S_A
    dog_infections_DB = lambda_DB * S_D

    # Progression exposed -> infectious
    infectious_Children = p.alpha_H * E_C
    infectious_Adult = p.alpha_H * E_A
    infectious_Dogs_DB = p.alpha_D * E_DB

    # Larval output into environment
    human_larvae_output_C = p.n_C * p.f_C * I_C
    human_larvae_output_A = p.n_A * p.f_A * I_A
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
    dE_DB = dog_infections_DB - dog_deaths_E_DB - infectious_Dogs_DB
    dI_DB = infectious_Dogs_DB - dog_deaths_I_DB

    # ODEs: environmental larvae
    dL_A = human_larvae_output_C + human_larvae_output_A - larvae_deaths_env_A
    dL_B = dog_larvae_output_DB - larvae_deaths_env_B

    return np.array([dS_C, dE_C, dI_C, dS_A, dE_A, dI_A, dS_D, dE_DB, dI_DB, dL_A, dL_B], dtype=float)


# --------------------------------------------------------------------------------------
# Simulation helpers
# --------------------------------------------------------------------------------------

def simulate(
    y0: Iterable[float],
    t_span: Tuple[float, float],
    params: Params,
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    method: str = "RK45",
) -> solve_ivp:
    """
    Integrate the ODE system.

    Args:
        y0: Initial conditions, iterable of length 11.
        t_span: (t0, tf) in days.
        params: Params dataclass.
        t_eval: Optional array of time points (days) at which to store the solution.
        rtol, atol: Solver tolerances.
        method: Any method accepted by `solve_ivp` (e.g., "RK45", "BDF").

    Returns:
        SciPy `OdeResult` as returned by `solve_ivp`.
    """
    y0 = np.asarray(list(y0), dtype=float)
    assert y0.shape == (11,), "y0 must have 11 elements."

    # Wrap RHS with params closed over
    def rhs(t, y):
        return model_ode(t, y, params)

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )
    return sol


def results_to_dataframe(
    sol: solve_ivp,
    particle: int = 1,
    beta_C: Optional[float] = None,
    beta_A: Optional[float] = None,
    beta_DB: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert a SciPy OdeResult into a tidy DataFrame with totals and prevalences.

    Columns produced:
        Time_day, Particle, beta_C, beta_A, beta_DB,
        S_C, E_C, I_C, S_A, E_A, I_A, S_D, E_DB, I_DB, L_A, L_B,
        Total_Children, Total_Adults, Total_Humans, Total_Dogs,
        prevalence_I_C, prevalence_I_A, prevalence_I_H, prevalence_I_DA

    Args:
        sol: SciPy OdeResult with `t` and `y`.
        particle: An identifier for parameter/particle index (for stacking runs).
        beta_C, beta_A, beta_DB: Optional copies of beta values to store per row.

    Returns:
        pd.DataFrame
    """
    if sol.t is None or sol.y is None:
        raise ValueError("Invalid OdeResult: missing solution arrays.")

    t = sol.t
    Y = sol.y  # shape (11, len(t))

    df = pd.DataFrame(
        {
            "Time_day": t,
            "S_C": Y[0],
            "E_C": Y[1],
            "I_C": Y[2],
            "S_A": Y[3],
            "E_A": Y[4],
            "I_A": Y[5],
            "S_D": Y[6],
            "E_DB": Y[7],
            "I_DB": Y[8],
            "L_A": Y[9],
            "L_B": Y[10],
        }
    )

    # Totals
    df["Total_Children"] = df["S_C"] + df["E_C"] + df["I_C"]
    df["Total_Adults"] = df["S_A"] + df["E_A"] + df["I_A"]
    df["Total_Humans"] = df["Total_Children"] + df["Total_Adults"]
    df["Total_Dogs"] = df["S_D"] + df["E_DB"] + df["I_DB"]

    # Prevalences (guard against zero totals)
    df["prevalence_I_C"] = np.where(df["Total_Children"] > 0, df["I_C"] / df["Total_Children"], 0.0)
    df["prevalence_I_A"] = np.where(df["Total_Adults"] > 0, df["I_A"] / df["Total_Adults"], 0.0)
    df["prevalence_I_H"] = np.where(df["Total_Humans"] > 0, (df["I_C"] + df["I_A"]) / df["Total_Humans"], 0.0)
    df["prevalence_I_DA"] = np.where(df["Total_Dogs"] > 0, df["I_DB"] / df["Total_Dogs"], 0.0)

    # Metadata / identifiers
    df["Particle"] = int(particle)
    df["beta_C"] = beta_C
    df["beta_A"] = beta_A
    df["beta_DB"] = beta_DB

    return df
