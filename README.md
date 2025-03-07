# SrPolarizabilities

Polarizabilities of strontium 5s² 1S0 and 5s5p 3P1 levels for bosonic isotopes only (no hyperfine states).

![plot](./polarizability-curves-600-1600nm.horiz.png)

The predictions are not state-of-the-art, as we simply gathered transitions RDMEs and core polarizabilities scattered in the existing (2024) litterature. It does not use any atomic model unlike Safronova's predictions. For the same reason, we cannot really compute uncertainties. The predictions are not fined tuned to reproduce the magic conditions nor measured (differential) polarizabilities. For instance, our prediction of the linear polarization magic wavelength (measured to be ~914nm) is quite off. Taking into account the latest RDME for the 1S0↔1P1 transition (https://arxiv.org/abs/2501.07395 which gives D=5.271) yields even worse results.

The code can be used to predict polarizabilities for other levels, but the transition table is incomplete and should be completed to accurate prediction.

For details and references, see https://doi.org/10.1103/PhysRevA.110.032819 ("Differential polarizability of the strontium intercombination transition at 1064.7 nm"). For generalities about polarization and for wavelength plots/comparisons, see my PhD dissertation (https://www.xif.fr/public/phys/phd/ chapter "Polarizabilities of strontium 1S0 and 3P1 levels").

Usage : either run directly `ODT_new_transitions.py` and enter the wavelength, or use the notebook `plot-polarizabilities.ipynb`.
