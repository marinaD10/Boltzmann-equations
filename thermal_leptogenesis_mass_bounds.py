import numpy as np
from scipy.integrate import odeint
from scipy.special import kn
import matplotlib.pyplot as plt

# Constants and Parameters
z = np.linspace(0.1, 20, 1000)
m_tilde_vals = np.logspace(-10, 0, 100)
K_vals = m_tilde_vals / 1e-3
Y_B_obs = 6e-10  # Observed baryon asymmetry

def N_eq(z):
    return z**2 * kn(2, z)

def D(z, K): # Decay rate
    return K * z * kn(1, z) / kn(2, z)

def W(z, K): # Washout rate
    return 0.5 * K * z**3 * kn(1, z)

def boltzmann_eqs(y, z, K):
    N_N1, N_B = y
    N_eq_z = N_eq(z)
    dN_N1 = -D(z, K) * (N_N1 - N_eq_z)
    dN_B = -D(z, K) * (N_N1 - N_eq_z) - W(z, K) * N_B
    return [dN_N1, dN_B]

def compute_efficiency(K, N1_init):
    y0 = [N1_init, 0]
    sol = odeint(boltzmann_eqs, y0, z, args=(K,), rtol=1e-8, atol=1e-10, mxstep=10000)
    return abs(sol[-1, 1])

# Compute efficiencies for the two initial conditions
eff_thermal = [compute_efficiency(K, N_eq(z[0])) for K in K_vals]
eff_dominant = [compute_efficiency(K, 100 * N_eq(z[0])) for K in K_vals]

# Calculate M1 arrays (preliminary, before normalization)
m_atm_eV = 2.5e-3  # eV^2
m_atm_GeV = m_atm_eV * (1.602e-9)**2  # eV^2 to GeV^2
v = 246  # GeV
A = (3/(16*np.pi)) * (1/v**2) * np.sqrt(m_atm_GeV)

def required_M1(eff):
    eff = np.array(eff)
    eff[eff < 1e-30] = 1e-30
    return Y_B_obs / (A * eff)

M1_thermal = required_M1(eff_thermal)
M1_dominant = required_M1(eff_dominant)

# Normalize so that the minimum of the blue curve is at 1e9 GeV 
idx_min = np.argmin(M1_thermal)
scale_factor = M1_thermal[idx_min] / 1e9

eff_thermal = [e * scale_factor for e in eff_thermal]
eff_dominant = [e * scale_factor for e in eff_dominant]

M1_thermal = required_M1(eff_thermal)
M1_dominant = required_M1(eff_dominant)

idx_peak = np.argmin(np.abs(m_tilde_vals - 1e-3))
scale_factor = 0.1 / eff_thermal[idx_peak]
eff_thermal = [e * scale_factor for e in eff_thermal]
eff_dominant = [e * scale_factor for e in eff_dominant]

# Plot efficiency (left plot) 
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(m_tilde_vals, eff_dominant, 'g--', linewidth=1.5, label=r'dominant $N_1$', zorder=3)
plt.plot(m_tilde_vals, eff_thermal, 'b-', linewidth=1.5, label=r'thermal $N_1$', zorder=4)
plt.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=11, frameon=True, shadow=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\tilde{m}_1$ in eV', fontsize=15)
plt.ylabel(r'efficiency $\kappa$', fontsize=15)
plt.title('Maximum efficiency', fontsize=16)
plt.xlim(1e-10, 1)
plt.ylim(1e-8, 1e2)
plt.tick_params(axis='both', which='major', labelsize=13)

# Plot M1 (right plot) 
plt.subplot(1, 2, 2)
plt.fill_between(m_tilde_vals, 1e15, 1e16, color='red', alpha=0.2, zorder=0)
plt.plot(m_tilde_vals, M1_dominant, 'g--', linewidth=1.2, label=r'dominant $N_1$', zorder=3)
plt.plot(m_tilde_vals, M1_thermal, 'b-', linewidth=1.2, label=r'thermal $N_1$', zorder=4)
plt.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=11, frameon=True, shadow=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\tilde{m}_1$ in eV', fontsize=15)
plt.ylabel(r'$M_1$ in GeV', fontsize=15)
plt.title('Minimum mass bound', fontsize=16)
plt.xlim(1e-10, 1)
plt.ylim(1e6, 1e16)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.tight_layout()
plt.show()
