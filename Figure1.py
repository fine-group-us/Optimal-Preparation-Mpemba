import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# Single-bang and limiting two-bang preparations.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.usetex": True,
})
d = 3


def integrate_bang(T_start, T_end, chi, a2_s, B):
    def rhs(T, y):
        a2 = y[0]
        T_power = T**1.5
        dT = chi*(1 + 3*a2_s/16) - T_power*(1 + 3*a2/16)
        da2 = 2/T*((T_power - chi)*a2 + B*T_power*(a2_s - a2))
        return [da2/dT]

    sol = solve_ivp(rhs, (T_start, T_end), [a2_s], method="Radau",
                    dense_output=True, rtol=1e-10, atol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol


for alpha, initial_temperatures, Tf in [
    (0.4, (3.0, 4.5, 6.0), 7.2),
    (0.9, (9.5, 8.8, 8.0), 7.0),
]:
    denominator = 73 + 56*d - 24*d*alpha - 105*alpha + 30*(1 - alpha)*alpha**2
    a2_s = 16*(1 - alpha)*(1 - 2*alpha**2)/denominator
    B = denominator/(16*(1 - alpha)*(3 + 2*d + 2*alpha**2))

    # Both panels use stationary temperatures 1 and 10.
    chi = 10**1.5 if alpha == 0.4 else 1
    T_start = 1 if alpha == 0.4 else 10
    T_stationary = chi**(2/3)
    T_end = T_stationary + 1e-5*(T_start - T_stationary)
    limiting = integrate_bang(T_start, T_end, chi, a2_s, B)
    temperatures = np.linspace(T_start, T_end, 1500)
    a2_limiting = limiting(temperatures)[0]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(a2_s, color="black", ls=":", lw=1.2)
    ax.plot(temperatures, a2_limiting, color="#0072B2", lw=1.8,
            label="Two bangs")
    for j, Ti in enumerate(initial_temperatures, start=1):
        direct = integrate_bang(Ti, T_end, chi, a2_s, B)
        temperatures = np.linspace(Ti, T_end, 900)
        ax.plot(temperatures, direct(temperatures)[0], color="#D55E00",
                ls="--", lw=1.5, label="Single bang" if j == 1 else None)
        ax.plot(Ti, a2_s, "o", color="#D55E00", ms=5.5,
                markerfacecolor="white", markeredgewidth=1.2)
        ax.plot(Tf, direct(Tf)[0], "o", color="#D55E00", ms=4,
                markeredgewidth=0.6, zorder=5)

    ax.axvline(Tf, color="black", ls=":", lw=1)
    for T in (1, 10):
        ax.axvline(T, color="black", ls=":", lw=0.9)
    ax.plot(T_start, a2_s, "s", color="#0072B2", ms=6)
    ax.plot(Tf, limiting(Tf)[0], "o", color="#0072B2", ms=4,
            markeredgewidth=0.6, zorder=5)

    y_span = a2_s - a2_limiting.min()
    ax.set_ylim(a2_limiting.min() - 0.55*y_span, a2_s + 0.30*y_span)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks([1, Tf, 10],
                  [r"$\chi_{\min}^{2/3}$", r"$T_{\rm f}$", r"$\chi_{\max}^{2/3}$"])
    ax.set_yticks([a2_s], [r"$a_2^{\rm s}$"])
    for j, Ti in enumerate(initial_temperatures, start=1):
        ax.text(Ti, a2_s + 0.06*y_span, rf"$T_{{\rm i,{j}}}$",
                ha="center", va="bottom", fontsize=14)
    ax.set_xlabel(r"$T$", fontsize=20)
    ax.set_ylabel(r"$a_2$", fontsize=20)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, labelsize=16, pad=5.5)
    if alpha == 0.4:
        ax.legend(loc="lower left", fontsize=11, frameon=True)

    fig.tight_layout()
    fig.savefig(f"bang_comparison_alpha{alpha}.pdf", bbox_inches="tight")
    plt.close(fig)
