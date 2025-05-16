import numpy as np
import matplotlib.pyplot as plt

def PMNS_matrix(theta12, theta13, theta23, delta_cp):
    c12 = np.cos(theta12)
    s12 = np.sin(theta12)
    c13 = np.cos(theta13)
    s13 = np.sin(theta13)
    c23 = np.cos(theta23)
    s23 = np.sin(theta23)
    e_minus_idelta = np.exp(-1j * delta_cp)
    U = np.array([
        [c12*c13, s12*c13, s13*e_minus_idelta],
        [-s12*c23 - c12*s23*s13*e_minus_idelta, c12*c23 - s12*s23*s13*e_minus_idelta, s23*c13],
        [s12*s23 - c12*c23*s13*e_minus_idelta, -c12*s23 - s12*c23*s13*e_minus_idelta, c23*c13]
    ], dtype=complex)
    return U

def oscillation_probability_3flavor(alpha, beta, L, E, delta_m_squared, theta12, theta13, theta23, delta_cp):
    U = PMNS_matrix(theta12, theta13, theta23, delta_cp)
    m1_sq = 0
    m2_sq = delta_m_squared[0]
    m3_sq = delta_m_squared[1]
    phases = np.array([0, 1.267 * m2_sq * L / E, 1.267 * m3_sq * L / E])
    amplitude = 0
    for i in range(3):
        amplitude += U[beta, i] * np.exp(-1j * phases[i]) * np.conj(U[alpha, i])
    P = np.abs(amplitude)**2
    return P

def oscillation_probability_3flavor_heavy(alpha, beta, L, E, delta_m_squared, theta12, theta13, theta23, delta_cp, Gamma3=0, L_coh3=np.inf, M3=1.0):
    U = PMNS_matrix(theta12, theta13, theta23, delta_cp)
    m1_sq = 0
    m2_sq = delta_m_squared[0]
    m3_sq = delta_m_squared[1]
    phases = np.array([0, 1.267 * m2_sq * L / E, 1.267 * m3_sq * L / E])
    gamma3 = E / M3
    decay_damping3 = np.exp(-Gamma3 * L / gamma3)
    coherence_damping3 = np.exp(-(L / L_coh3)**2)
    dampings = np.array([1, 1, decay_damping3 * coherence_damping3])
    amplitude = 0
    for i in range(3):
        amplitude += U[beta, i] * dampings[i] * np.exp(-1j * phases[i]) * np.conj(U[alpha, i])
    P = np.abs(amplitude)**2
    return P

# Parameters
theta12 = 0.59  # radians
theta13 = 0.15
theta23 = 0.79
delta_cp = 0
delta_m_squared = [7.53e-5, 2.44e-3]  # eV^2
Gamma3 = 1e-4  # decay width for heavy state, in 1/km
L_coh3 = 500   # coherence length for heavy state, in km
M3 = 1.0       # GeV

L_over_E = np.linspace(0, 1000, 1000)  # km/GeV
E = 1.0  # GeV

# Standard 3-flavor oscillation
probs_standard = [oscillation_probability_3flavor(1, 0, L_over_E[i]*E, E, delta_m_squared, theta12, theta13, theta23, delta_cp)
                  for i in range(len(L_over_E))]

# Heavy (damped) 3-flavor oscillation
probs_heavy = [oscillation_probability_3flavor_heavy(1, 0, L_over_E[i]*E, E, delta_m_squared, theta12, theta13, theta23, delta_cp, Gamma3, L_coh3, M3)
               for i in range(len(L_over_E))]

plt.figure(figsize=(10, 6))
plt.plot(L_over_E, probs_standard, label=r'Standard $P(\nu_\mu \rightarrow \nu_e)$', color='deepskyblue', linewidth=2)
plt.plot(L_over_E, probs_heavy, label=r'Heavy (damped) $P(\nu_\mu \rightarrow \nu_e)$', color='crimson', linewidth=2)
plt.title('3-Flavor Neutrino Oscillation Probability vs L/E')
plt.xlabel(r'$L/E$ (km/GeV)')
plt.ylabel('Oscillation Probability')
plt.grid(True)
plt.legend()
plt.show()
