# Hubbard Model: RPA Analysis and DQMC Validation

A comprehensive computational study of the two-dimensional Hubbard model combining analytical Random Phase Approximation (RPA) with exact Determinantal Quantum Monte Carlo (DQMC) simulations.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
   - [The Hubbard Model](#the-hubbard-model)
   - [Green's Functions](#greens-functions)
   - [Random Phase Approximation](#random-phase-approximation)
   - [Spin Susceptibility](#spin-susceptibility)
   - [Stoner Criterion](#stoner-criterion)
   - [D-Normalization Scheme](#d-normalization-scheme)
3. [Computational Methods](#computational-methods)
   - [DQMC Algorithm](#dqmc-algorithm)
   - [RPA Implementation](#rpa-implementation)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Repository Structure](#repository-structure)
7. [Results](#results)
8. [Citation](#citation)

---

## Overview

This project investigates magnetic properties of the 2D Hubbard model by:

- **Exact simulations**: DQMC for finite-temperature quantum many-body systems
- **Analytical theory**: RPA for spin response and magnetic fluctuations
- **Validation**: Novel D-normalization scheme for quantitative RPA-DQMC comparison
- **Analysis**: Stoner criterion for magnetic instabilities and finite-size scaling

**Key Features:**
- MPI-parallelized DQMC with numerical stabilization
- Complete RPA calculation from bare susceptibility to structure factors
- D-normalization matching exact thermodynamic sum rules
- Comprehensive validation across parameter space (L, β, U)

---

## Theoretical Background

### The Hubbard Model

The Hubbard Hamiltonian describes interacting electrons on a lattice:

$$\hat{H} = -t \sum_{\langle i,j \rangle, \sigma} \left( c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.} \right) - \mu \sum_{i,\sigma} n_{i\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}$$

**Parameters:**
- **t**: Nearest-neighbor hopping amplitude (sets energy scale, typically t = 1)
- **U**: On-site Coulomb repulsion (interaction strength)
- **μ**: Chemical potential (controls particle density)
- **β = 1/T**: Inverse temperature

**Lattice geometry:**
- 2D square lattice with periodic boundary conditions
- Size: Nx × Ny sites (N = Nx·Ny total)
- Momentum: **k** = (2πm/Nx, 2πn/Ny) where m, n are integers

**Non-interacting dispersion:**

$$\varepsilon(\mathbf{k}) = -2t[\cos(k_x) + \cos(k_y)]$$

**Half-filling condition:**
- At U = 0: μ = 0 gives ⟨n⟩ = 1 (one electron per site)
- At U > 0 (particle-hole symmetry): μ = U/2

---

### Green's Functions

#### Single-Particle Green's Function

The imaginary-time Green's function describes fermion propagation:

$$G(i, j, \tau) = -\langle T_\tau \, c_i(\tau) c_j^\dagger(0) \rangle$$

where **Tτ** is the imaginary-time ordering operator for τ ∈ [0, β].

**Key properties:**
1. **Anti-periodicity**: G(τ + β) = -G(τ)
2. **Matsubara representation**:
   $$G(\mathbf{k}, \tau) = \frac{1}{\beta} \sum_n G(\mathbf{k}, i\omega_n) e^{-i\omega_n \tau}$$
   where ωₙ = (2n+1)π/β are fermionic Matsubara frequencies

3. **Spectral decomposition**:
   $$G(\mathbf{k}, i\omega_n) = \int_{-\infty}^{\infty} d\omega \frac{A(\mathbf{k}, \omega)}{i\omega_n - \omega}$$

#### Non-Interacting Green's Function

For U = 0, we can solve exactly:

$$G_0(\mathbf{k}, i\omega_n) = \frac{1}{i\omega_n - \xi(\mathbf{k})}$$

where ξ(**k**) = ε(**k**) - μ.

In imaginary time (0 ≤ τ < β):

$$G_0(\mathbf{k}, \tau) = -[1 - f(\xi(\mathbf{k}))] e^{-\xi(\mathbf{k})\tau}$$

where f(ξ) = 1/[1 + exp(βξ)] is the Fermi-Dirac distribution.

**Physical meaning:**
- f(ξ) = occupation probability
- 1 - f(ξ) = hole probability
- G₀ interpolates between particle/hole propagation

---

### Random Phase Approximation

#### Bare Susceptibility

The **non-interacting (bare) susceptibility** measures the system's response to external perturbations:

$$\chi_0(\mathbf{q}, T) = \frac{1}{N} \sum_{\mathbf{k}} \frac{f(\xi_{\mathbf{k}}) - f(\xi_{\mathbf{k}+\mathbf{q}})}{\xi_{\mathbf{k}+\mathbf{q}} - \xi_{\mathbf{k}}}$$

**Derivation:**
Starting from linear response theory, the density-density correlation function is:

$$\chi_0(\mathbf{q}, i\Omega_m) = -\frac{1}{N\beta} \sum_{\mathbf{k},n} G_0(\mathbf{k}, i\omega_n) G_0(\mathbf{k}+\mathbf{q}, i\omega_n + i\Omega_m)$$

For static (Ωₘ = 0) and taking the zero-frequency limit, this reduces to the form above.

**Numerical handling of degeneracies:**

When ξ(**k**+**q**) ≈ ξ(**k**), we use l'Hôpital's rule:

$$\lim_{\Delta\xi \to 0} \frac{f(\xi) - f(\xi + \Delta\xi)}{\Delta\xi} = -\frac{\partial f}{\partial \xi} = \beta f(\xi)[1 - f(\xi)]$$

**Implementation:**
```
For each momentum q:
  sum = 0
  For each k in Brillouin zone:
    ε_k = -2t[cos(kx) + cos(ky)] - μ
    ε_k+q = -2t[cos(kx+qx) + cos(ky+qy)] - μ
    Δε = ε_k+q - ε_k
    
    If |Δε| < tolerance:
      sum += β × f(ε_k) × [1 - f(ε_k)]
    Else:
      sum += [f(ε_k) - f(ε_k+q)] / Δε
  
  χ₀(q) = sum / N
```

**Physical interpretation:**
- χ₀(**q**) measures tendency for density/spin fluctuations at wavevector **q**
- Maximum at **q*** indicates dominant instability
- For square lattice: typically **q*** = (π, π) → antiferromagnetic

#### RPA Resummation

RPA includes **bubble diagram resummation** to all orders:

$$\chi(\mathbf{q}) = \chi_0(\mathbf{q}) + \chi_0(\mathbf{q}) U \chi_0(\mathbf{q}) + \chi_0(\mathbf{q}) U \chi_0(\mathbf{q}) U \chi_0(\mathbf{q}) + \cdots$$

This geometric series sums to:

$$\chi(\mathbf{q}) = \frac{\chi_0(\mathbf{q})}{1 - U \chi_0(\mathbf{q})}$$

**Diagrammatic interpretation:**
```
χ = χ₀ + χ₀ U χ₀ + χ₀ U χ₀ U χ₀ + ...

    ○        ○   ○        ○   ○   ○
χ = ─── + ───╱─\─── + ───╱─\─╱─\─── + ...
            U            U   U

where ○──○ represents χ₀ (bubble)
```

**Physical meaning:**
- RPA captures **screening** of bare interaction
- Includes polarization effects
- Valid for weak-to-moderate coupling

---

### Spin Susceptibility

#### Spin Structure Factor

The equal-time **spin structure factor** measures magnetic correlations:

$$S^z(\mathbf{q}) = \langle S^z_{\mathbf{q}} S^z_{-\mathbf{q}} \rangle = \frac{1}{4} \sum_{i,j} e^{i\mathbf{q}\cdot(\mathbf{R}_i - \mathbf{R}_j)} \langle (n_{i\uparrow} - n_{i\downarrow})(n_{j\uparrow} - n_{j\downarrow}) \rangle$$

**Connection to susceptibility:**

Via fluctuation-dissipation theorem:

$$S^z(\mathbf{q}) = \beta \chi(\mathbf{q})$$

**RPA prediction:**

$$S^z_{\text{RPA}}(\mathbf{q}) = \beta \chi_{\text{RPA}}(\mathbf{q}) = \frac{\beta \chi_0(\mathbf{q})}{1 - U \chi_0(\mathbf{q})}$$

#### Sum Rule

The structure factor satisfies an exact sum rule:

$$\sum_{\mathbf{q}} S^z(\mathbf{q}) = \frac{1}{4} \sum_{i,j} \langle (n_{i\uparrow} - n_{i\downarrow})(n_{j\uparrow} - n_{j\downarrow}) \rangle$$

For a uniform system, this simplifies to:

$$\sum_{\mathbf{q}} S^z(\mathbf{q}) = \frac{N}{4} \langle (n_\uparrow - n_\downarrow)^2 \rangle = \frac{N}{4}(1 - 2D)$$

where **D = ⟨n↑n↓⟩** is the **double occupancy** (probability of two electrons on same site).

**Derivation:**
```
⟨(n↑ - n↓)²⟩ = ⟨n↑² + n↓² - 2n↑n↓⟩
             = ⟨n↑⟩ + ⟨n↓⟩ - 2⟨n↑n↓⟩    [since n²_σ = n_σ for fermions]
             = 1 - 2D                    [at half-filling: ⟨n↑⟩ + ⟨n↓⟩ = 1]
```

**Importance:** This sum rule provides a thermodynamic constraint for validation.

#### Local Magnetic Moment

The **on-site magnetic moment squared**:

$$m_s^2(z) = \frac{1}{N} \sum_i \langle (n_{i\uparrow} - n_{i\downarrow})^2 \rangle = \frac{S^z(\pi,\pi)}{N}$$

For **SU(2) spin symmetry**:

$$m_{\text{total}}^2 = 3 m_s^2(z)$$

**Heisenberg limit:**

At large U and low T, system approaches S = 1/2 Heisenberg model:

$$\hat{H}_{\text{eff}} = J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j$$

where **J = 4t²/U** (from second-order perturbation theory).

For 2D square lattice Heisenberg antiferromagnet:
- T = 0 staggered magnetization: m ≈ 0.307
- Therefore: **m²ₛ(z) ≈ (0.307)²/3 ≈ 0.0314**

---

### Stoner Criterion

#### Ferromagnetic Instability

The **Stoner criterion** predicts magnetic instability when the RPA denominator vanishes:

$$1 - U \chi_0(\mathbf{q}^*) = 0$$

where **q*** is the ordering wavevector.

**For ferromagnetic order** (q* = 0):
$$U_c^{\text{FM}} = \frac{1}{\chi_0(0)}$$

**For antiferromagnetic order** (q* = (π,π) on square lattice):
$$U_c^{\text{AFM}} = \frac{1}{\chi_0(\pi,\pi)}$$

**Physical interpretation:**
- Interaction U competes with kinetic energy (∝ 1/χ₀)
- When U exceeds critical value, magnetic order becomes favorable
- RPA diverges at instability (signals breakdown)

#### Temperature Dependence

The bare susceptibility increases with decreasing temperature:

$$\chi_0(\mathbf{q}, T) \nearrow \text{ as } T \searrow$$

This gives a **temperature-dependent phase boundary**:

$$T_c(U) : 1 - U \chi_0(\mathbf{q}^*, T_c) = 0$$

**Phase diagram structure:**
```
Temperature T
    ↑
    |   Paramagnetic
    |   (1 - Uχ₀ > 0)
 Tc |___________________
    |                   
    | Magnetically      
    | Ordered           
    |   (1 - Uχ₀ < 0)  
    └──────────────────→ U
           Uc
```

#### Stoner Indicator Function

We define:

$$I(U, \beta, \mathbf{q}) = 1 - U \chi_0(\mathbf{q}, \beta)$$

**Interpretation:**
- **I > 0**: Paramagnetic, stable
- **I → 0**: Approaching instability
- **I < 0**: Ordered phase (RPA breaks down)

For antiferromagnetic order:

$$I_{\text{AFM}} = 1 - U \chi_0(\pi, \pi, \beta)$$

**Practical use:**
- Plot I vs β for fixed U: see approach to instability
- Plot I vs U for fixed β: find critical Uc
- 2D map I(U, β): visualize phase boundary

---

### D-Normalization Scheme

#### Motivation

**Problem:** RPA is approximate, doesn't exactly satisfy sum rules.

**Solution:** Rescale RPA results to match exact thermodynamic constraint from DQMC.

#### Normalization Procedure

1. **Run DQMC** to measure double occupancy D

2. **Calculate RPA** structure factor S^z_RPA(q)

3. **Compute target sum** from DQMC:
   $$\text{Target} = \frac{1}{4}(1 - 2D_{\text{DQMC}}) N$$

4. **Find normalization factor α**:
   $$\alpha = \frac{\text{Target}}{\sum_{\mathbf{q}} S^z_{\text{RPA}}(\mathbf{q})} = \frac{(1 - 2D_{\text{DQMC}}) N}{4 \sum_{\mathbf{q}} S^z_{\text{RPA}}(\mathbf{q})}$$

5. **Apply rescaling**:
   $$S^z_{\text{norm}}(\mathbf{q}) = \alpha \cdot S^z_{\text{RPA}}(\mathbf{q})$$

#### Properties

**What is preserved:**
- Momentum dependence of S(**q**)
- Qualitative physics (peak locations, etc.)
- Relative magnitudes between different **q**

**What is corrected:**
- Overall normalization
- Sum rule satisfaction
- Quantitative comparison with DQMC

**Why it works:**
- RPA captures correct **functional form** S(**q**)
- D-normalization corrects **overall scale**
- Exact sum rule: ∑ S = (1-2D)N/4 is fundamental thermodynamic constraint

#### Validation Metrics

After normalization, compare:

1. **Channel-by-channel**: S^z_norm(**q**) vs S^z_DQMC(**q**) at key momenta
   - (π,π): Antiferromagnetic
   - (π,0), (0,π): Intermediate
   - (0,0): Ferromagnetic

2. **Magnetic moment**: m²ₛ(z) = S(π,π)/N

3. **Finite-size scaling**: Extrapolate to N → ∞

4. **βJ scaling**: Approach to Heisenberg limit

---

## Computational Methods

### DQMC Algorithm

#### Auxiliary Field Decomposition

The Hubbard interaction is decoupled using **discrete Hubbard-Stratonovich transformation**:

$$e^{-\Delta\tau U n_{i\uparrow} n_{i\downarrow}} = \frac{1}{2} \sum_{s_i = \pm 1} e^{\lambda s_i (n_{i\uparrow} - n_{i\downarrow})} e^{-\Delta\tau U/2}$$

where:
$$\lambda = \text{arccosh}(e^{\Delta\tau U/2})$$

**Effect:** Replaces quartic interaction with quadratic coupling to auxiliary Ising field sᵢ.

#### Propagator Matrices

Define the **single-time-slice propagator**:

$$B_l(s) = e^{-\Delta\tau K/2} \cdot \text{diag}(e^{\lambda s_i}) \cdot e^{-\Delta\tau K/2}$$

where **K** is the kinetic energy matrix:
$$K_{ij} = \begin{cases}
-t & \text{if } i,j \text{ nearest neighbors} \\
-\mu & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

The **full propagator** from τ = 0 to β:

$$B = \prod_{l=1}^L B_l(s)$$

where L = β/Δτ is the number of time slices (Trotter discretization).

#### Equal-Time Green's Function

The equal-time Green's function:

$$G = [1 + B]^{-1}$$

**Observables:**
- Particle number: n_σ = 1 - Tr(G_σ)/N
- Double occupancy: D = ⟨(1 - G↑)(1 - G↓)⟩/N
- Spin correlations: Compute in momentum space for S(**q**)

#### Numerical Stabilization

**Challenge:** Product ∏ Bₗ causes exponential growth/decay of singular values.

**Solution - QR decomposition every k steps:**

```python
Q = I; R = I
for l in 1...L:
    C = B_l × Q
    Q, R_new = QR(C)
    R = R_new × R
    
    if l mod k == 0:  # Stabilize
        Q_r, R = QR(R)
        Q = Q × Q_r

G = [I + R×Q]^{-1} = I - Q × [I + R]^{-1} × R
```

This prevents numerical overflow while preserving G exactly.

#### Monte Carlo Sampling

**Metropolis updates** of auxiliary field configuration {sᵢ,ₗ}:

1. Propose flip: sᵢ,ₗ → -sᵢ,ₗ

2. Compute acceptance ratio:
   $$r = \left| \det(1 + B'_\uparrow) \det(1 + B'_\downarrow) \right| / \left| \det(1 + B_\uparrow) \det(1 + B_\downarrow) \right|$$

3. Accept with probability min(1, r)

**Fast update formula:**
Can compute r from rank-1 update without full determinant:
$$r = [1 + (1 - G_{ii})(e^{2\lambda s_{i,l}} - 1)]_\uparrow \times [1 + (1 - G_{ii})(e^{-2\lambda s_{i,l}} - 1)]_\downarrow$$

#### Parameters

**Typical settings:**
- Time slices: L = 120 (Δτ = β/120)
- Warm-up sweeps: 200 (thermalization)
- Measurement sweeps: 800 (statistics)
- Stabilization frequency: every 16 steps
- MPI parallelization: 4-8 independent runs

---

### RPA Implementation

#### k-Space Grid

Generate momentum grid for Brillouin zone integration:

```python
def grid_k(Nx, Ny):
    kx = 2π × [0, 1, ..., Nx-1] / Nx
    ky = 2π × [0, 1, ..., Ny-1] / Ny
    return meshgrid(kx, ky)
```

**Resolution:** Use fine grid (e.g., 128×128) for accurate χ₀ calculation.

#### Bare Susceptibility Calculation

```python
def chi0(qx, qy, beta, Nx_grid=128, Ny_grid=128):
    kx, ky = grid_k(Nx_grid, Ny_grid)
    
    # Dispersion
    eps_k = -2t*(cos(kx) + cos(ky)) - μ
    eps_kq = -2t*(cos(kx+qx) + cos(ky+qy)) - μ
    
    # Fermi functions
    f_k = 1/(1 + exp(beta*eps_k))
    f_kq = 1/(1 + exp(beta*eps_kq))
    
    # Handle degeneracies
    delta_eps = eps_kq - eps_k
    term = zeros_like(delta_eps)
    
    degenerate = (abs(delta_eps) < 1e-12)
    term[degenerate] = beta * f_k * (1 - f_k)
    term[~degenerate] = (f_k - f_kq) / delta_eps
    
    return mean(term)
```

#### RPA Susceptibility

```python
def chi_RPA(qx, qy, beta, U):
    chi0_val = chi0(qx, qy, beta)
    
    denominator = 1 - U*chi0_val
    
    if abs(denominator) < 1e-10:
        return inf  # Stoner instability
    else:
        return chi0_val / denominator
```

#### Structure Factor Map

```python
def compute_S_RPA(Nx, Ny, beta, U):
    S = zeros((Ny, Nx))
    
    for m in range(Nx):
        for n in range(Ny):
            qx = 2π*m/Nx
            qy = 2π*n/Ny
            
            chi = chi_RPA(qx, qy, beta, U)
            S[n, m] = beta * chi
    
    return S
```

#### D-Normalization

```python
def apply_normalization(S_RPA, D_DQMC, Nx, Ny):
    N = Nx * Ny
    
    # Target from DQMC
    target = 0.25 * (1 - 2*D_DQMC) * N
    
    # Current RPA sum
    sum_RPA = S_RPA.sum()
    
    # Normalization factor
    alpha = target / sum_RPA
    
    # Rescale
    S_normalized = alpha * S_RPA
    
    return S_normalized, alpha
```

---

## Installation

### Prerequisites

- Python 3.8+
- MPI (OpenMPI or MPICH)
- C compiler (for mpi4py)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/hubbard-rpa-dqmc.git
cd hubbard-rpa-dqmc

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install numpy scipy pandas matplotlib mpi4py jupyter

# Or use requirements file (if provided)
pip install -r requirements.txt

# Verify MPI installation
mpirun --version
```

### System-Specific MPI Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install open-mpi
```

**Conda:**
```bash
conda install -c conda-forge mpi4py openmpi
```

---

## Usage

### Running DQMC Simulations

#### Single Parameter Set

```bash
# Example: L=4, β=6, U=4
mpirun -np 4 python src/Smartv7_SpinMPI_v2.py \
    --nx 4 --ny 4 \
    --beta 6.0 \
    --U 4.0 \
    --nwarm 200 \
    --nmeas 800 \
    --L 120
```

**Output:**
- Console: Real-time progress and measurements
- `results/summary.csv`: Appended row with D, S(π,π), occupancies
- `maps/S_q_mean_N4x4_U4.0_beta6.0.csv`: Full S(**q**) map

#### Parameter Sweep

Create a sweep script:

```bash
#!/bin/bash
for L in 2 4 6; do
  for BETA in 1.0 4.0 6.0; do
    for U in 1.0 4.0 10.0; do
      echo "Running L=$L, β=$BETA, U=$U"
      mpirun -np 4 python src/Smartv7_SpinMPI_v2.py \
        --nx $L --ny $L \
        --beta $BETA --U $U \
        --nwarm 200 --nmeas 800
    done
  done
done
```

### RPA Analysis

#### Using Jupyter Notebooks

```bash
jupyter notebook notebooks/RPAv2.ipynb
```

**Key notebooks:**
- `RPAv1.ipynb`: RPA with Padé analytic continuation
- `RPAv2.ipynb`: RPA with D-normalization (main)
- `StonerMap.ipynb`: Stoner criterion visualization
- `post.ipynb`: Post-processing and figure generation

#### Python Script Example

```python
import numpy as np
import pandas as pd
from rpa_utils import chi0_at_q, compute_S_RPA, apply_normalization

# Load DQMC data
df = pd.read_csv('data/summary.csv')
row = df[(df['Nx']==4) & (df['beta']==6.0) & (df['U']==4.0)].iloc[0]

# Extract parameters
Nx, Ny = int(row['Nx']), int(row['Ny'])
beta, U = float(row['beta']), float(row['U'])
D_DQMC = float(row['D_mean'])

# Compute RPA
S_RPA = compute_S_RPA(Nx, Ny, beta, U)

# Apply D-normalization
S_norm, alpha = apply_normalization(S_RPA, D_DQMC, Nx, Ny)

# Compare at (π,π)
S_pipi_RPA = S_norm[Ny//2, Nx//2]
S_pipi_DQMC = float(row['S_pi_pi'])

print(f"S(π,π) DQMC: {S_pipi_DQMC:.6f}")
print(f"S(π,π) RPA:  {S_pipi_RPA:.6f}")
print(f"Agreement: {100*(1-abs(S_pipi_RPA-S_pipi_DQMC)/S_pipi_DQMC):.2f}%")
```

### Stoner Analysis

```python
# Compute Stoner indicator
def stoner_indicator(U, beta):
    chi0_pipi = chi0_at_q(np.pi, np.pi, beta)
    return 1 - U*chi0_pipi

# Critical U
beta_vals = np.linspace(1, 10, 50)
U_crit = []
for beta in beta_vals:
    chi0_max = chi0_at_q(np.pi, np.pi, beta)
    U_crit.append(1.0/chi0_max)

plt.plot(beta_vals, U_crit)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$U_c$')
plt.title('Critical U for AFM Instability')
```

---

## Repository Structure

```
.
├── README.md                    # This file (complete theory + usage)
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
│
├── src/                         # Source code
│   └── Smartv7_SpinMPI_v2.py   # DQMC simulation (MPI-parallelized)
│
├── notebooks/                   # Jupyter notebooks
│   ├── RPAv1.ipynb             # RPA with Padé continuation
│   ├── RPAv2.ipynb             # RPA with D-normalization
│   ├── StonerMap.ipynb         # Stoner criterion analysis
│   └── post.ipynb              # Post-processing and figures
│
├── data/                        # Data files
│   ├── summary.csv             # DQMC results summary
│   ├── ms2_summary_dnorm.csv   # D-normalized magnetic moments
│   └── combined_S_q_mean.txt   # Full S(q) maps
│
├── results/                     # Generated DQMC outputs
│   └── .gitkeep
│
├── figures/                     # Generated plots
│   └── .gitkeep
│
└── docs/                        # Additional documentation
```

---

## Results

### Parameter Space

**Explored:**
- Lattice sizes: L ∈ {2, 4, 6} (N = 4, 16, 36 sites)
- Temperatures: β ∈ {1, 4, 6}
- Interactions: U ∈ {1, 4, 10}
- Total runs: 27 combinations

### Key Findings

#### 1. RPA Accuracy

| Coupling | Agreement | Notes |
|----------|-----------|-------|
| Weak (U=1) | >95% | Excellent across all β |
| Moderate (U=4) | >97% | D-normalization essential |
| Strong (U=10) | >98% at low T | Heisenberg regime |

#### 2. Stoner Analysis

- **U_crit(β)** increases with decreasing T
- System paramagnetic for all studied U, β (I > 0)
- U = 4 approaches instability at β = 6 (I ≈ 0.2)

#### 3. Magnetic Moments

Approach to Heisenberg T=0 value (m²ₛ ≈ 0.0314):

| L | β | U  | m²ₛ DQMC | m²ₛ RPA | Heisenberg |
|---|---|----|----------|---------|------------|
| 6 | 6 | 4  | 0.0289   | 0.0282  | 0.0314     |
| 6 | 6 | 10 | 0.0312   | 0.0318  | 0.0314     |
| 4 | 6 | 10 | 0.0321   | 0.0327  | 0.0314     |

#### 4. Finite-Size Effects

- Systematic approach to thermodynamic limit
- Larger L shows smoother S(**q**)
- Extrapolation: m²ₛ(L→∞) consistent with theory

#### 5. βJ Scaling

Universal scaling with **βJ** (J = 4t²/U):
- All U values collapse onto single curve
- Validates effective Heisenberg description
- Smooth approach to T=0 limit

### Physical Interpretation

**Antiferromagnetic Fluctuations:**
- S(π,π) dominates at all parameters
- Enhanced at low T and large U
- Signature of AFM ground state

**Temperature Evolution:**
- Increasing β sharpens momentum-space structure
- Suppresses thermal fluctuations
- Approaches ordered state (but no transition at finite T, finite N)

**Interaction Strength:**
- U = 1: Weakly correlated, itinerant electrons
- U = 4: Crossover regime, strong correlations
- U = 10: Localized moments, Heisenberg physics

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{hubbard_rpa_dqmc_2025,
  author = {Yuewen Sun},
  title = {Hubbard Model: RPA Analysis and DQMC Validation},
  year = {2025},
  url = {https://github.com/jerrysun8416/hubbard-rpa-dqmc},
  version = {1.0}
}
```

### References

**Theoretical Background:**
1. Hubbard, J. (1963). Proc. R. Soc. Lond. A 276, 238
2. Scalapino, D. J. et al. (1989). Phys. Rev. B 40, 197

**DQMC Method:**
3. Blankenbecler, R., Scalapino, D. J., & Sugar, R. L. (1981). Phys. Rev. D 24, 2278
4. White, S. R. et al. (1989). Phys. Rev. B 40, 506
5. Assaad, F. F. & Evertz, H. G. (2008). Lect. Notes Phys. 739, 277

---

## License

MIT License - see LICENSE file for details.

## Contact

**Issues:** https://github.com/jerrysun8416/hubbard-rpa-dqmc/issues  
**Email:** sunyw@shanghaitech.edu.cn

---

**Last Updated:** January 2026  
**Version:** 1.0.0
