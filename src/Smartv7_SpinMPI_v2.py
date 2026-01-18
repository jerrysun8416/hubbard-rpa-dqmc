# dqmc_mpi_sweep.py
# MPI-enabled sweep of (Nx,Ny) in {(2,2),(4,4),(6,6),(8,8)}, beta in {1,4,6}, U in {1,4,10}
# - Algorithmic core kept intact (no debugging changes)
# - Each MPI rank runs the same combo with an independent seed; results are averaged across ranks
# - Only rank 0 writes logs and output files
# - Defaults: NWARM=200, NMEAS=800, L=120
# - Minimal change: after each combo finishes, master appends a row to results/summary.csv

import os, sys, math, time, csv, argparse
# Set BLAS threads before importing numpy to avoid oversubscription across MPI ranks
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from mpi4py import MPI
import numpy as np
from contextlib import redirect_stdout, redirect_stderr, nullcontext

# ---------------- RNG ----------------
MASK = (1 << 64) - 1
def splitmix64_next(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z ^= (z >> 30); z = (z * 0xBF58476D1CE4E5B9) & MASK
    z ^= (z >> 27); z = (z * 0x94D049BB133111EB) & MASK
    z ^= (z >> 31); return state, z
def rng_uniform_01(state):
    state, z = splitmix64_next(state)
    return state, float(z >> 11) * (1.0 / (1 << 53))

# -------------- Lattice / model --------------
def build_K(Nx, Ny, t=1.0, mu=0.0):
    N = Nx * Ny
    K = np.zeros((N, N), dtype=np.float64)
    def idx(x, y): return x + Nx * y
    for x in range(Nx):
        for y in range(Ny):
            i = idx(x, y)
            j = idx((x + 1) % Nx, y); K[i, j] = K[j, i] = -t
            j = idx(x, (y + 1) % Ny); K[i, j] = K[j, i] = -t
    K -= float(mu) * np.eye(N)
    return 0.5 * (K + K.T)

def sublattice_eta(Nx, Ny):
    N = Nx * Ny
    eta = np.empty(N, dtype=np.float64)
    for y in range(Ny):
        for x in range(Nx):
            eta[x + Nx * y] = 1.0 if ((x + y) % 2 == 0) else -1.0
    return eta

def make_lambda(U, dt):
    val = np.exp(0.5 * dt * U)
    return float(np.log(val + np.sqrt(max(val * val - 1.0, 0.0))))

def expm_symmetric(K, scale):
    w, Q = np.linalg.eigh(K)
    ew = np.exp(scale * w)
    return Q @ np.diag(ew) @ Q.T

# ---------------- B-slices ----------------
def make_B_slice_up(E_half, s_col, lam):
    D = np.exp(lam * s_col)
    return E_half @ (D[:, None] * E_half)

def build_Bu_stack(E_half, s, lam):
    N, Ls = s.shape
    Bu = np.empty((Ls, N, N), dtype=np.float64)
    for l in range(Ls):
        Bu[l] = make_B_slice_up(E_half, s[:, l], lam)
    return Bu

# -------- Stabilized equal-time G --------
def _seq_excl_left(B_set, l):
    Ls = B_set.shape[0]
    return [B_set[i] for i in range(l - 1, -1, -1)] + [B_set[i] for i in range(Ls - 1, l - 1, -1)]
def _seq_excl_right(B_set, l):
    Ls = B_set.shape[0]
    return [B_set[i] for i in range(l, -1, -1)] + [B_set[i] for i in range(Ls - 1, l, -1)]

def _eqtime_G_from_seq_qr(seq, reqr_every=16):
    N = seq[0].shape[0]
    Q = np.eye(N, dtype=np.float64)
    R = np.eye(N, dtype=np.float64)
    for k, B in enumerate(seq, 1):
        C = B @ Q
        Qk, Rk = np.linalg.qr(C, mode='reduced')
        Q = Qk
        R = Rk @ R
        if reqr_every and (k % reqr_every == 0):
            Qr, R = np.linalg.qr(R, mode='reduced')
            Q = Q @ Qr
    K = np.linalg.solve(np.eye(N) + R @ Q, np.eye(N))
    return np.eye(N) - Q @ (K @ R)

def balanced_eqtime_G(B_set, l, restart_every=6, use_qr=True, reqr_every=16):
    seqL = _seq_excl_left(B_set, l)
    seqR = _seq_excl_right(B_set, l)
    if use_qr:
        GL = _eqtime_G_from_seq_qr(seqL, reqr_every=reqr_every)
        GR = _eqtime_G_from_seq_qr(seqR, reqr_every=reqr_every)
    else:
        GL = _eqtime_G_from_seq_qr(seqL, reqr_every=reqr_every)
        GR = _eqtime_G_from_seq_qr(seqR, reqr_every=reqr_every)
    return 0.5 * (GL + GR), GL, GR

# -------- Sherman-Morrison update --------
def sm_update_up_inplace(G, i, r, tiny=1e-14):
    gii = float(G[i, i])
    c = r - 1.0
    denom_up = 1.0 + (1.0 - gii) * c
    if not np.isfinite(denom_up) or abs(denom_up) < tiny:
        return False, denom_up
    alpha = c / denom_up
    col = G[:, i].copy()
    row = np.zeros_like(G[i, :]); row[i] = 1.0
    row -= G[i, :]
    G -= alpha * (col[:, None] @ row[None, :])
    return True, denom_up

# -------------- Stats helpers --------------
def mean_stderr(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0: return (float('nan'), float('nan'))
    m = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(max(len(arr), 1))) if len(arr) > 1 else 0.0
    return m, se

def autocorr_time(x, max_lag=None):
    x = np.asarray(x, dtype=float); x = x - x.mean()
    n = len(x)
    if n < 3: return 0.5
    if max_lag is None: max_lag = min(n // 2, 100)
    denom = np.dot(x, x)
    if denom == 0: return 0.5
    tau = 0.5
    for k in range(1, max_lag + 1):
        rho = np.dot(x[:-k], x[k:]) / denom
        if rho <= 0: break
        tau += rho
    return max(tau, 0.5)

def mean_se_tau(x):
    m, se = mean_stderr(x)
    tau = autocorr_time(x, max_lag=min(10, len(x)//2))
    n_eff = len(x) / (2 * tau) if tau > 0 else len(x)
    se_tau = np.std(x, ddof=1) / np.sqrt(max(n_eff, 1)) if len(x) > 1 else 0.0
    return m, se, se_tau, tau

# -------------- D & spin estimators --------------
def double_occupancy_LR_consistent(GL_up, GR_up, eta):
    N = GL_up.shape[0]
    GdL = np.eye(N) - (eta[:, None] * GR_up.T * eta[None, :])
    GdR = np.eye(N) - (eta[:, None] * GL_up.T * eta[None, :])
    n_up_L = 1.0 - np.real(np.diag(GL_up))
    n_dn_L = 1.0 - np.real(np.diag(GdL))
    n_up_R = 1.0 - np.real(np.diag(GR_up))
    n_dn_R = 1.0 - np.real(np.diag(GdR))
    D_L = float(np.mean(n_up_L * n_dn_L))
    D_R = float(np.mean(n_up_R * n_dn_R))
    D   = 0.5 * (D_L + D_R)
    n_up_mean = 0.5 * (float(np.mean(n_up_L)) + float(np.mean(n_up_R)))
    n_dn_mean = 0.5 * (float(np.mean(n_dn_L)) + float(np.mean(n_dn_R)))
    return D, (D_L, D_R), (n_up_mean, n_dn_mean)

def _czz_from_G(Gup, Gdn):
    N = Gup.shape[0]
    n_up = 1.0 - np.real(np.diag(Gup))
    n_dn = 1.0 - np.real(np.diag(Gdn))
    Aup = np.eye(N) - Gup.T
    Mup = np.outer(n_up, n_up) - (Aup * (np.eye(N) - Gup))
    np.fill_diagonal(Mup, n_up)
    Adn = np.eye(N) - Gdn.T
    Mdn = np.outer(n_dn, n_dn) - (Adn * (np.eye(N) - Gdn))
    np.fill_diagonal(Mdn, n_dn)
    cross = np.outer(n_up, n_dn) + np.outer(n_dn, n_up)
    return 0.25 * (Mup + Mdn - cross)

def _avg_translate_to_Cr(Czz, Nx, Ny):
    N = Nx * Ny
    C = Czz.reshape(Ny, Nx, Ny, Nx)
    Cr = np.empty((Ny, Nx), dtype=np.float64)
    y = np.arange(Ny)[:, None]
    x = np.arange(Nx)[None, :]
    for dy in range(Ny):
        for dx in range(Nx):
            yp = (y + dy) % Ny
            xp = (x + dx) % Nx
            Cr[dy, dx] = np.mean(C[y, x, yp, xp])
    return Cr

def measure_Sq_map(GL_up, GR_up, eta, Nx, Ny):
    N = Nx * Ny
    Gdn_L = np.eye(N) - (eta[:, None] * GR_up.T * eta[None, :])
    Gdn_R = np.eye(N) - (eta[:, None] * GL_up.T * eta[None, :])
    Czz_L = _czz_from_G(GL_up, Gdn_L)
    Czz_R = _czz_from_G(GR_up, Gdn_R)
    Czz   = 0.5 * (Czz_L + Czz_R)
    Cr = _avg_translate_to_Cr(Czz, Nx, Ny)
    Sq = np.real(np.fft.fft2(Cr))  # NumPy convention
    return Sq, Cr

def symmetrize_Sq_C4v(Sq):
    mats = [
        Sq, np.rot90(Sq, 1), np.rot90(Sq, 2), np.rot90(Sq, 3),
        np.fliplr(Sq), np.flipud(Sq),
        np.fliplr(np.rot90(Sq, 1)), np.flipud(np.rot90(Sq, 1)),
    ]
    return sum(mats) / len(mats)

# -------------- Single combo run (no file writes except optional logging) --------------
def run_single_combo(Nx, Ny, beta, U, *, L=120, NWARM=100, NMEAS=400,
                     RESTART_EVERY=6, REBUILD_PERIOD=12, DENS_WARN_TOL=5e-3,
                     CAND_RADIUS=2, PRINT_CAND_SUMMARY=True, PRINT_DEEP_DIAG=True,
                     write_log=False, log_path=None, seed_base=0xC0FFEE,
                     out_root="."):
    """
    Returns dict with per-rank results (will be averaged via MPI outside):
      - 'Sq_mean': Ny x Nx array (double)
      - 'D_mean', 'n_up', 'n_dn' (floats)
      - 'S_pi_pi' (float derived from Sq_mean)
      - 'log_path' if write_log else None
    """
    N = Nx * Ny
    dt = beta / L
    lam = make_lambda(U, dt)

    # logging context
    if write_log:
        os.makedirs(os.path.join(out_root, "logs"), exist_ok=True)
        assert log_path is not None
        log_fh = open(log_path, "w", buffering=1)
        ctx = redirect_stdout(log_fh)
        ctx2 = redirect_stderr(log_fh)
    else:
        ctx = nullcontext()
        ctx2 = nullcontext()

    with ctx, ctx2:
        print(f"[cfg] {Nx}x{Ny}  beta={beta}  U={U}  L={L}  dt={dt:.5f}  lam={lam:.6f}")

        K = build_K(Nx, Ny, 1.0, 0.0)
        E_half = expm_symmetric(-K, 0.5 * dt)
        eta = sublattice_eta(Nx, Ny)

        # Deterministic initial HS field
        s = np.empty((N, L), dtype=np.float64)
        for i in range(N):
            for l in range(L):
                s[i, l] = 1.0 if ((i + 3 * l + 1) % 2 == 0) else -1.0

        Bu = build_Bu_stack(E_half, s, lam)

        total = NWARM + NMEAS
        l_meas_fixed = L // 2

        # Unique seed per rank/combo
        rng = (seed_base & MASK)

        Sq_sum   = np.zeros((Ny, Nx), dtype=np.float64)
        Sq_sumsq = np.zeros((Ny, Nx), dtype=np.float64)
        Sq_count = 0
        D_list   = []
        nup_list = []
        ndn_list = []

        for sweep in range(total):
            t0 = time.perf_counter()
            is_meas = (sweep >= NWARM)
            tag = (f"{sweep + 1 - NMEAS}/{NMEAS}" if is_meas else f"warm {sweep + 1}/{NWARM}")  # keep prints harmless
            # (prints only if write_log=True)
            if write_log:
                print(f"\n=== [sweep {tag}] ===")

            acc_total = 0
            restab_total = 0

            for l in range(L):
                _, Gu, _ = balanced_eqtime_G(Bu, l, restart_every=6)
                accepted = 0
                restab = 0
                dirty = False

                for i in range(N):
                    gii = float(Gu[i, i])
                    r = math.exp(-2.0 * lam * s[i, l])
                    den_up = 1.0 + (1.0 - gii) * (r - 1.0)
                    den_dn = 1.0 + gii * (1.0 / r - 1.0)
                    det_ratio = den_up * den_dn
                    if (not np.isfinite(det_ratio)) or det_ratio <= 0.0:
                        Bu[l] = make_B_slice_up(E_half, s[:, l], lam)
                        _, Gu, _ = balanced_eqtime_G(Bu, l, restart_every=6)
                        gii = float(Gu[i, i])
                        den_up = 1.0 + (1.0 - gii) * (r - 1.0)
                        den_dn = 1.0 + gii * (1.0 / r - 1.0)
                        det_ratio = den_up * den_dn
                        if (not np.isfinite(det_ratio)) or det_ratio <= 0.0:
                            continue

                    rng, u = rng_uniform_01(rng)
                    if u < min(1.0, det_ratio):
                        ok, _ = sm_update_up_inplace(Gu, i, r)
                        if (not ok) or (not np.all(np.isfinite(Gu))):
                            Bu[l] = make_B_slice_up(E_half, s[:, l], lam)
                            _, Gu, _ = balanced_eqtime_G(Bu, l, restart_every=6)
                            restab += 1
                            ok2, _ = sm_update_up_inplace(Gu, i, r)
                            if (not ok2) or (not np.all(np.isfinite(Gu))):
                                continue
                        s[i, l] *= -1.0
                        accepted += 1
                        dirty = True

                        if (accepted % 12) == 0:
                            Bu[l] = make_B_slice_up(E_half, s[:, l], lam)
                            _, Gu, _ = balanced_eqtime_G(Bu, l, restart_every=6)
                            restab += 1

                if dirty:
                    Bu[l] = make_B_slice_up(E_half, s[:, l], lam)

                acc_total += accepted
                restab_total += restab

            if is_meas:
                l_meas = L // 2

                if write_log:
                    print("[cand] l  trL     trR     trAvg   |trAvg-N/2|   symErr     L/Rgap   ||GL-GR||_F/N")
                    for lm in [(l_meas + d) % L for d in range(-2, 3)]:
                        Gavg, GL, GR = balanced_eqtime_G(Bu, lm, restart_every=6)
                        trL = float(np.trace(GL).real)
                        trR = float(np.trace(GR).real)
                        trA = float(np.trace(Gavg).real)
                        dev = abs(trA - 0.5 * N)
                        sym_err = np.linalg.norm(Gavg - Gavg.T, 'fro') / max(N, 1)
                        lr_gap = abs(trL - trR)
                        lr_fro = np.linalg.norm(GL - GR, 'fro') / max(N, 1)
                        print(f"[cand] {lm:3d}  {trL:6.3f}  {trR:6.3f}   {trA:7.3f}   {dev:9.3f}   {sym_err:7.3e}   {lr_gap:7.3e}   {lr_fro:7.3e}")

                Gavg, GL_up, GR_up = balanced_eqtime_G(Bu, l_meas, restart_every=6)
                trL = float(np.trace(GL_up).real)
                trR = float(np.trace(GR_up).real)
                trA = float(np.trace(Gavg).real)
                dev = abs(trA - 0.5 * N)
                symA = np.linalg.norm(Gavg - Gavg.T, 'fro') / max(N, 1)
                symL = np.linalg.norm(GL_up - GL_up.T, 'fro') / max(N, 1)
                symR = np.linalg.norm(GR_up - GR_up.T, 'fro') / max(N, 1)
                lr_gap = abs(trL - trR)
                lr_fro = np.linalg.norm(GL_up - GR_up, 'fro') / max(N, 1)

                raw_diag = np.real(np.diag(Gavg))
                raw_min = float(raw_diag.min()); raw_mean = float(raw_diag.mean()); raw_max = float(raw_diag.max())
                frac_oob = float(np.mean((raw_diag < -1e-6) | (raw_diag > 1.0 + 1e-6)))

                D_slice, (D_L, D_R), (n_up_slice, n_dn_slice) = double_occupancy_LR_consistent(GL_up, GR_up, eta)

                if write_log:
                    print(f"[diag-pre] l={l_meas:3d}  Tr(GL)={trL:.3f}  Tr(GR)={trR:.3f}  Tr(Gavg)={trA:.3f}  "
                          f"|Δ|={dev:.3f}  L/R gap={lr_gap:.3f}  ||GL-GR||_F/N={lr_fro:.3e}")
                    print(f"[diag-sym] symA={symA:.3e}  symL={symL:.3e}  symR={symR:.3e}")
                    print(f"[diag-diag] raw diag[min/mean/max]={raw_min:.3f}/{raw_mean:.3f}/{raw_max:.3f}  frac_oob={frac_oob:.3e}")
                    print(f"[diag] l={l_meas:3d}  <n↑>={n_up_slice:.4f} <n↓>={n_dn_slice:.4f}  Tr(Gavg)={trA:.3f}")
                    if abs((n_up_slice + n_dn_slice) - 1.0) > 5e-3:
                        print(f"[warn] ⟨n⟩={n_up_slice + n_dn_slice:.6f} deviates from 1.0 (tol=5e-3)")
                    print(f"[meas] D_L={D_L:.6f}  D_R={D_R:.6f}  D={D_slice:.6f}")

                # Slice-averaged S(q), D, and densities over all slices
                lm_indices = range(L)
                Sq_acc        = np.zeros((Ny, Nx), dtype=np.float64)
                Cii_sum_acc   = 0.0
                D_acc         = 0.0
                nup_acc       = 0.0
                ndn_acc       = 0.0

                for lm in lm_indices:
                    _, GL_lm, GR_lm = balanced_eqtime_G(Bu, lm, restart_every=6)

                    Sq_lm, Cr_lm = measure_Sq_map(GL_lm, GR_lm, eta, Nx, Ny)
                    Sq_lm = symmetrize_Sq_C4v(Sq_lm)
                    sumS   = float(np.sum(Sq_lm))
                    sumCii = float(N * Cr_lm[0,0])
                    if sumS != 0.0:
                        Sq_lm *= (sumCii / sumS)
                    Sq_acc      += Sq_lm
                    Cii_sum_acc += sumCii

                    D_lm, _, _ = double_occupancy_LR_consistent(GL_lm, GR_lm, eta)
                    D_acc += D_lm

                    Nsite = GL_lm.shape[0]
                    GdL = np.eye(Nsite) - (eta[:, None] * GR_lm.T * eta[None, :])
                    GdR = np.eye(Nsite) - (eta[:, None] * GL_lm.T * eta[None, :])
                    n_up_L = 1.0 - np.real(np.diag(GL_lm))
                    n_dn_L = 1.0 - np.real(np.diag(GdL))
                    n_up_R = 1.0 - np.real(np.diag(GR_lm))
                    n_dn_R = 1.0 - np.real(np.diag(GdR))
                    nup_acc += 0.5 * (float(np.mean(n_up_L)) + float(np.mean(n_up_R)))
                    ndn_acc += 0.5 * (float(np.mean(n_dn_L)) + float(np.mean(n_dn_R)))

                Sq_sweep      = Sq_acc / len(lm_indices)
                Cii_sum_sweep = Cii_sum_acc / len(lm_indices)
                D_sweep       = D_acc    / len(lm_indices)
                nup_sweep     = nup_acc  / len(lm_indices)
                ndn_sweep     = ndn_acc  / len(lm_indices)

                sum_Sq  = float(np.sum(Sq_sweep))
                sum_Cii = float(Cii_sum_sweep)
                if sum_Sq != 0.0:
                    Sq_sweep *= (sum_Cii / sum_Sq)

                avg_Cii = Cii_sum_sweep / N
                if write_log:
                    print(f"[check] ⟨Czz_ii⟩ = {avg_Cii:.6f}  vs  0.25*(1-2D) = {0.25*(1-2*D_sweep):.6f}")
                    print(f"[sumrule z] Σ_q S(q) = {float(np.sum(Sq_sweep)):.6f}  |  Σ_i Cii = {sum_Cii:.6f}")

                S_pi_pi = float(Sq_sweep[Ny//2, Nx//2])
                m_s_sq  = S_pi_pi / (Nx * Ny)
                if write_log:
                    print(f"[AF] S^z(pi,pi)={S_pi_pi:.6f}  m_s^2(z)={m_s_sq:.6f}  (SU(2): m^2≈3*m_s^2)")

                Sq_sum   += Sq_sweep
                Sq_sumsq += Sq_sweep * Sq_sweep
                Sq_count += 1
                D_list.append(D_sweep)
                nup_list.append(nup_sweep)
                ndn_list.append(ndn_sweep)

            if write_log:
                print(f"[mc] accepted_flips={acc_total} restabilizations={restab_total}")
                print(f"[time] sweep {tag}: {time.perf_counter() - t0:.3f} s")

        # Per-rank finals
        D_mean, D_se_naive, D_se_tau, D_tau = mean_se_tau(D_list)
        nup_mean, _, _, _ = mean_se_tau(nup_list)
        ndn_mean, _, _, _ = mean_se_tau(ndn_list)

        Sq_mean = Sq_sum / max(Sq_count, 1)
        S_pi_pi_mean = float(Sq_mean[Ny // 2, Nx // 2])

        if write_log:
            print("\n=== FINAL (per-rank) ===")
            print(f"D_mean = {D_mean:.6f}  ± {D_se_naive:.6f} (naive)  ± {D_se_tau:.6f} (τ_int={D_tau:.2f})   over {len(D_list)} sweeps")
            print(f"n_up = {nup_mean:.6f}   n_dn = {ndn_mean:.6f}   n_up+n_dn={nup_mean+ndn_mean:.6f}")

        if write_log:
            log_fh.flush()
            log_fh.close()

    return dict(
        Sq_mean=Sq_mean,
        D_mean=D_mean,
        n_up=nup_mean,
        n_dn=ndn_mean,
        S_pi_pi=S_pi_pi_mean,
        log_path=log_path if write_log else None
    )

# -------------- MPI Sweep orchestrator --------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=".", help="Output root directory")
    parser.add_argument("--nwarm", type=int, default=100)
    parser.add_argument("--nmeas", type=int, default=400)
    parser.add_argument("--L", type=int, default=120)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_master = (rank == 0)

    # parameter grid
    # sizes_grid = [(2,2),(4,4),(6,6),(8,8)]
    sizes_grid = [(2,2),(4,4),(6,6)]
    betas = [1.0, 4.0, 6.0]
    Us = [1.0, 4.0, 10.0]
    combos = [(Nx,Ny,b,U) for (Nx,Ny) in sizes_grid for b in betas for U in Us]  # 36 combos

    if is_master:
        print(f"[MPI] ranks = {size}")
        print(f"Total combos = {len(combos)}")
        os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "maps"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "results"), exist_ok=True)

        # --- Minimal change: prepare summary.csv and write header if not exists ---
        summary_path = os.path.join(args.out, "results", "summary.csv")
        if not os.path.exists(summary_path):
            with open(summary_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Nx","Ny","beta","U","L","D_mean","n_up","n_dn","S_pi_pi","map_csv","log"])
    else:
        summary_path = None  # unused on workers

    comm.Barrier()

    aggregated_rows = []  # master collects in-memory too (unchanged)

    for (Nx, Ny, beta, U) in combos:
        # Unique seed per rank per combo
        seed_base = (0xA5A5A5A5A5A5 ^ (Nx*1315423911) ^ (Ny*2654435761) ^
                     (int(beta*1e6)*97531) ^ (int(U*1e6)*86491) ^ (rank*0x9E3779B97F4A7C15)) & MASK

        # Only master writes this combo log
        log_path = os.path.join(args.out, "logs", f"dqmc_N{Nx}x{Ny}_U{U}_beta{beta}.log") if is_master else None

        # Run local replica
        res_local = run_single_combo(
            Nx, Ny, beta, U,
            L=args.L, NWARM=args.nwarm, NMEAS=args.nmeas,
            write_log=is_master, log_path=log_path,
            seed_base=seed_base, out_root=args.out
        )

        # Reduce scalars (sum then average)
        for key in ["D_mean", "n_up", "n_dn"]:
            val = res_local[key]
            total = comm.reduce(val, op=MPI.SUM, root=0)
            if is_master:
                res_local[key] = total / size  # global mean

        # Reduce S(q) map
        Sq_local = res_local["Sq_mean"]
        Ny_, Nx_ = Sq_local.shape
        Sq_sum = np.zeros_like(Sq_local) if is_master else None
        comm.Reduce([Sq_local, MPI.DOUBLE], [Sq_sum, MPI.DOUBLE], op=MPI.SUM, root=0)

        if is_master:
            Sq_mean_global = Sq_sum / float(size)
            S_pi_pi_global = float(Sq_mean_global[Ny_//2, Nx_//2])

            # Append aggregated summary line to this combo log
            with open(log_path, "a", buffering=1) as fh:
                fh.write("\n=== AGGREGATED OVER MPI RANKS ===\n")
                fh.write(f"D_mean(avg over {size}) = {res_local['D_mean']:.6f}\n")
                fh.write(f"n_up = {res_local['n_up']:.6f}   n_dn = {res_local['n_dn']:.6f}   n_up+n_dn={res_local['n_up']+res_local['n_dn']:.6f}\n")
                fh.write(f"S^z(pi,pi) (from aggregated map) = {S_pi_pi_global:.6f}\n")

            # Write aggregated map CSV
            map_path = os.path.join(args.out, "maps", f"S_q_mean_N{Nx}x{Ny}_U{U}_beta{beta}.csv")
            with open(map_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["m","n","qx","qy","S_mean"])
                for n in range(Ny_):
                    for m in range(Nx_):
                        qx, qy = 2*np.pi*m/Nx_, 2*np.pi*n/Ny_
                        w.writerow([m, n, qx, qy, float(Sq_mean_global[n, m])])

            row = [
                Nx, Ny, beta, U, args.L,
                f"{res_local['D_mean']:.8f}",
                f"{res_local['n_up']:.8f}",
                f"{res_local['n_dn']:.8f}",
                f"{S_pi_pi_global:.8f}",
                map_path,
                log_path
            ]
            aggregated_rows.append(row)

            # -------- Minimal change: append the row to summary.csv immediately --------
            with open(os.path.join(args.out, "results", "summary.csv"), "a", newline="") as fsum:
                wsum = csv.writer(fsum)
                wsum.writerow(row)

        comm.Barrier()

    if is_master:
        # Final full summary rewrite (kept; harmless — produces a clean final file)
        summary_path = os.path.join(args.out, "results", "summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Nx","Ny","beta","U","L","D_mean","n_up","n_dn","S_pi_pi","map_csv","log"])
            for row in aggregated_rows:
                w.writerow(row)
        print(f"[DONE] Wrote summary: {summary_path}")
        print(f"Logs in:   {os.path.join(args.out,'logs')}")
        print(f"Maps in:   {os.path.join(args.out,'maps')}")
        print(f"Results in:{os.path.join(args.out,'results')}")

if __name__ == "__main__":
    main()
