"""
Example 3 -- The rho_star principle: two windows on one system.

A single underlying system (exponential correlator C(r) = exp(-r/tau))
observed through two different analytical windows:
   bare   : O_1(r) = C(r)                     -> rho_star = 1
   gamma2 : O_2(r) = sigma * tau^2/(sigma+tau)^2  -> rho_star = 2 - sqrt(3)

The bridge predicts: sigma_c[O_i] / rho_star_i = tau, for both probes.
The probes "disagree" on sigma_c (by a factor 1/(2-sqrt(3)) ~ 3.73) but
they "agree" on tau exactly. This is the rho_star separation principle
the foundation paper isolates as the central interpretive result
(Proposition 4.1).

Aha effect:
  Probe-dependent sigma_c is not a measurement error -- it's a
  convention factor that the framework subtracts out cleanly.
"""
from pathlib import Path
import math
import numpy as np

from sigma_c_v4 import analyze, two_probe_test, bare, gamma_k


HERE = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 60)
    print("EXAMPLE 3 -- Two windows on one system")
    print("  Different sigma_c values, same tau.")
    print("=" * 60)

    tau_true = 5.0

    # --- Probe 1: bare correlator ---
    sigma = np.geomspace(0.05, 100.0, 600)
    O_1 = np.exp(-sigma / tau_true)
    r_1 = analyze(sigma, O_1, window=bare(),
                  label="Probe 1: bare correlator")
    r_1.x_name = "lag sigma"
    r_1.y_name = "chi_{O_1}(sigma)"

    # --- Probe 2: Gamma-2 windowed correlator ---
    # The analytic windowed observable is sigma * tau^2 / (sigma + tau)^2,
    # which the gamma2 window's known Mellin transform produces.
    O_2 = sigma * tau_true ** 2 / (sigma + tau_true) ** 2
    r_2 = analyze(sigma, O_2, window=gamma_k(2),
                  label="Probe 2: Gamma-2 windowed correlator")
    r_2.x_name = "lag sigma"
    r_2.y_name = "chi_{O_2}(sigma)"

    print("\n--- Probe 1 (bare) ---")
    print(r_1.summary())
    print("\n--- Probe 2 (Gamma-2) ---")
    print(r_2.summary())

    # Non-circular two-probe test
    test = two_probe_test(r_1, r_2, delta_threshold=0.05)
    print("\n--- Non-circular two-probe test ---")
    print(test.summary())

    print(f"\nTheoretical tau            : {tau_true:.4f}")
    print(f"tau from probe 1           : {r_1.tau:.4f}")
    # Probe 2 picks the rising-side peak only if the geometric classifier
    # restricts; v4 currently sees both peaks and reports regime II for the
    # gamma2 correlator -- by paper convention O_*^+ the rising side is used.
    if r_2.tau is not None:
        print(f"tau from probe 2           : {r_2.tau:.4f}")
    else:
        if isinstance(r_2.sigma_c, list):
            rising = min(r_2.sigma_c)
            print(f"probe 2 sigma_c (rising)   : {rising:.4f}  (Onice^+ convention)")
            print(f"-> recovered tau           : {rising / r_2.rho_star:.4f}")

    card_1 = HERE / "card_03a_probe_bare.png"
    card_2 = HERE / "card_03b_probe_gamma2.png"
    r_1.card(save_to=card_1)
    r_2.card(save_to=card_2)
    print(f"\n--> Cards saved: {card_1.name}, {card_2.name}")


if __name__ == "__main__":
    main()
