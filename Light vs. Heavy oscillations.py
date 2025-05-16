import numpy as np
import matplotlib.pyplot as plt

# Light neutrino oscillation (undamped)
def neutrino_oscillation_probability(L_over_E, delta_m_squared, theta):
    osc_arg = 1.267 * delta_m_squared * L_over_E
    return np.sin(2*theta)**2 * np.sin(osc_arg)**2

# Heavy neutrino oscillation (with decay and decoherence) 
def heavy_neutrino_oscillation(L_over_E, energy, M, delta_m_squared, theta, Gamma, L_coh):
    L = L_over_E * energy
    phase = 1.267 * delta_m_squared * L_over_E
    gamma = energy / M
    decay_damping = np.exp(-Gamma * L / gamma)
    coherence_damping = np.exp(-(L / L_coh)**2)
    P = (np.sin(2*theta)**2 * np.sin(phase)**2 
         * decay_damping 
         * coherence_damping)
    return P

# Parameters
# Light neutrino
delta_m2_light = 2.5e-3  # eV^2
theta_light = np.pi/4    # 45 degrees

# Heavy neutrino
M = 1.0                # GeV
delta_m2_heavy = 1e-2  # eV^2
theta_heavy = np.pi/4
Gamma = 1e-5           # 1/km
L_coh = 1000           # km
energy = 2.0           # GeV

# L/E range (km/GeV)
L_over_E_values = np.linspace(0, 2000, 1000)

# --- Calculate probabilities ---
P_light = [neutrino_oscillation_probability(L_over_E, delta_m2_light, theta_light)
           for L_over_E in L_over_E_values]

P_heavy = [heavy_neutrino_oscillation(L_over_E, energy, M, delta_m2_heavy, theta_heavy, Gamma, L_coh)
           for L_over_E in L_over_E_values]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Light neutrino plot
axes[0].plot(L_over_E_values, P_light, color='b', linewidth=2)
axes[0].set_title(r'Light Neutrino Oscillation', fontsize=14)
axes[0].set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
axes[0].set_ylabel('Oscillation Probability', fontsize=12)
axes[0].set_xlim(0, 2000)
axes[0].set_ylim(0, 1.05)
axes[0].grid(True)

# Heavy neutrino plot
axes[1].plot(L_over_E_values, P_heavy, color='r', linewidth=2)
axes[1].set_title(r'Heavy Neutrino Oscillation (Damped)', fontsize=14)
axes[1].set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
axes[1].set_xlim(0, 2000)
axes[1].grid(True)

plt.tight_layout()
plt.show()
