import numpy as np
import matplotlib.pyplot as plt

def heavy_neutrino_oscillation(L, energy, M, delta_m_squared, theta, Gamma, L_coh):
    phase = 1.267 * delta_m_squared * L / energy
    gamma = energy / M
    decay_damping = np.exp(-Gamma * L / gamma)
    coherence_damping = np.exp(-(L / L_coh)**2)
    P = (np.sin(2 * theta)**2 * np.sin(phase)**2 
         * decay_damping 
         * coherence_damping)
    return P

# Fixed parameters
M = 1.0                # GeV
delta_m_squared = 1e-2 # eV^2
theta = np.pi / 4
energy = 2.0           # GeV
L_values = np.linspace(0, 1000, 1000)

# Left plot: Varying Gamma
Gamma_values = [0, 1e-6, 1e-5, 5e-5, 1e-4]  # in 1/km
L_coh_fixed = 1000  # km
colors_gamma = ['k', 'b', 'g', 'orange', 'r']

# Right plot: Varying L_coh 
L_coh_values = [200, 500, 1000, 3000, 1e10]  # 1e10 ~ no decoherence
Gamma_fixed = 1e-5  # 1/km
colors_coh = ['r', 'orange', 'green', 'blue', 'black']

# Subplots 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Left: Varying Gamma
for Gamma, color in zip(Gamma_values, colors_gamma):
    probs = [heavy_neutrino_oscillation(L, energy, M, delta_m_squared, theta, Gamma, L_coh_fixed)
             for L in L_values]
    label = rf'$\Gamma = {Gamma:.0e}\,\mathrm{{km^{{-1}}}}$'
    ax1.plot(L_values / energy, probs, label=label, color=color, linewidth=2)
ax1.set_title(r'Varying $\Gamma$ (Decay Width)', fontsize=14)
ax1.set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
ax1.set_ylabel('Oscillation Probability', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 1.05)

# Right: Varying L_coh
for L_coh, color in zip(L_coh_values, colors_coh):
    probs = [heavy_neutrino_oscillation(L, energy, M, delta_m_squared, theta, Gamma_fixed, L_coh)
             for L in L_values]
    label = r'$L_\mathrm{coh} = $' + (f'{L_coh:.0f} km' if L_coh < 1e5 else r'$\infty$ (no decoh.)')
    ax2.plot(L_values / energy, probs, label=label, color=color, linewidth=2)
ax2.set_title(r'Varying $L_\mathrm{coh}$ (Coherence Length)', fontsize=14)
ax2.set_xlabel(r'$L/E$ (km/GeV)', fontsize=12)
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 500)

plt.tight_layout()
plt.show()
