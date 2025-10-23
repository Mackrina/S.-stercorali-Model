
---

## ⚙️ Model Overview

Each model represents compartments for **Susceptible (S)**, **Exposed (E)**, and **Infectious (I)** hosts, and includes environmental larval stages.

- **Humans:** Children and adults with age progression
- **Dogs:** Separate infection lineages (A, B) in zoonotic models
- **Environment:** Larval pools (L_A, L_B)

The ODEs are solved using `scipy.integrate.solve_ivp` with the LSODA method.

---

## 🧩 Key Scripts

| Script | Description |
|--------|-------------|
| `Model_*.py` | Defines the system of ODEs and parameter structure. |
| `Params_and_IC*.py` | Contains initial conditions and parameter dictionaries. |
| `solve_*.py` | Runs baseline equilibrium simulations. |
| `solve_MDA_WASH_*.py` | Runs MDA/WASH intervention. |

---

## 💊 Interventions

### Mass Drug Administration (MDA)
- Simulated as instantaneous pulses at years 1–5.
- Clears a fraction of exposed and infectious individuals in both humans and dogs.

### Water, Sanitation, and Hygiene (WASH)
- Introduced at year 1.
- Reduces transmission and/or environmental contamination via fractional reductions in:
  - `beta_C`, `beta_A` (exposure)
  - `f_C`, `f_A` (shedding)

---


