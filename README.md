# Leptogenesis and Neutrino Oscillations Simulation

This repository contains Python codes used to simulate the generation of the baryon asymmetry of the universe via thermal leptogenesis alongside modeling neutrino oscillation probabilities, including effects of decay and coherence damping relevant for heavy neutrinos. It includes numerical solutions to Boltzmann equations under different initial conditions for the lightest heavy neutrino N₁. 

## 📄 Contents

- `Leptogenesis Mass Bounds.py`: Main script for solving the Boltzmann equations and plotting results.
- `Light vs. Heavy oscillations.py`: Plotting the neutrino oscillation probabilities for the light vs. heavy neutrinos.
- `Damping effects.py`: Includes damping effects.
- `Plots/`: A folder with all generated plots including efficiency vs. effective neutrino mass and required M₁ vs. effective neutrino mass and neutrino oscillations probability.

## 🔧 Features

- Supports different initial conditions:
  - Thermal abundance
  - Dominant N₁
- Computes:
  - Efficiency factor κ
  - Lower bound on M₁ 
- Log-log plots for visual clarity

## 📊 Example Output

- Efficiency vs. m̃₁
- Minimum M₁ for successful baryogenesis
- Side-by-side comparison of light and heavy neutrino oscillations as a function of L/E.
- Impact of decay widths and coherence lengths on heavy neutrino oscillation probabilities

## ▶️ How to Run

python Leptogenesis Mass Bounds.py
python Light vs. Heavy oscillations.py
python Damping effects.py
