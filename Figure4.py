import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Run DataFig4.py first. The symbols show the DSMC data.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.usetex": True,
})
theory = np.loadtxt("DataFig4.dat")
simulation = np.loadtxt("DataFig4DSMC.dat")
d = 3
alpha_c = 1/np.sqrt(2)
alpha = np.sort(np.r_[np.linspace(theory[:, 1].min(), theory[:, 1].max(), 2000),
                      alpha_c])


def stationary_kurtoses(alpha):
    numerator = 16*(1 - alpha)*(1 - 2*alpha**2)
    a2_s = numerator/(73 + 56*d - 24*d*alpha - 105*alpha
                      + 30*(1 - alpha)*alpha**2)
    a2_hcs = numerator/(25 + 2*alpha**2*(alpha - 1) + 24*d + alpha*(8*d - 57))
    return a2_s, a2_hcs


a2_s, a2_hcs = stationary_kurtoses(alpha)
stationary_color = "#52616b"
bang_colors = {"max": "#fbe5e5", "min": "#e4effa"}

for column, extremum in [(2, "min"), (3, "max")]:
    fig, ax = plt.subplots(figsize=(6, 4.09))
    if extremum == "min":
        ideal = np.minimum(0, a2_hcs)
        bangs = ("max", "min")
        heating_in_lower_regime = 1
        positions = [(0.35, 0.12), (0.85, 0.48)]
        labels = [(r"$a_2^{\rm s}$", 0.18, 0, (8, 12), stationary_color, "center"),
                  (r"$a_2^{\rm HCS}$", 0.86, 1, (0, -6), "black", "top")]
    else:
        ideal = np.maximum(0, a2_hcs)
        bangs = ("min", "max")
        heating_in_lower_regime = 0
        positions = [(0.35, 0.80), (0.85, 0.80)]
        labels = [(r"$a_2^{\rm s}$", 0.18, 0, (0, -14), stationary_color, "center"),
                  (r"$a_2^{\rm HCS}$", 0.18, 1, (8, 14), "black", "center")]

    ax.axvspan(0, alpha_c, color=bang_colors[bangs[0]], lw=0, zorder=0)
    ax.axvspan(alpha_c, 1, color=bang_colors[bangs[1]], lw=0, zorder=0)
    ax.plot(alpha, ideal, color="black", ls=(0, (6, 3)), lw=1.8, zorder=1)
    ax.plot(alpha, a2_s, color=stationary_color, ls=(0, (1, 2)),
            lw=1.8, dash_capstyle="round", zorder=1)

    legend_handles = []
    for chi_max, style, color, marker in [
        (2, "-", "#0072B2", "s"),
        (10, ":", "#E69F00", "o"),
        (100, "--", "#009E73", "^"),
        (1000, "-.", "#7B3294", "v"),
    ]:
        data = theory[theory[:, 0] == chi_max]
        ax.plot(data[:, 1], data[:, column], color=color, ls=style,
                lw=1.4, zorder=2)
        for heating in (1, 0):
            data = simulation[(simulation[:, 0] == chi_max) &
                              (simulation[:, 1] == heating)]
            if heating == heating_in_lower_regime:
                data = data[data[:, 2] < alpha_c]
            else:
                data = data[data[:, 2] > alpha_c]
            ax.plot(data[:, 2], data[:, 3], ls="None", marker=marker, markersize=7,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1, zorder=3)

        if extremum == "min":
            legend_handles.append(Line2D(
                [], [], color=color, ls=style, lw=1.4, marker=marker, markersize=7,
                markerfacecolor="white", markeredgecolor=color,
                label=rf"$\chi_{{\max}}={chi_max}$",
            ))

    ax.set_xlim(0, 1)
    ax.margins(x=0, y=0.06)
    if extremum == "min":
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=bottom - 0.08*(top - bottom), top=top)
        ax.legend(handles=legend_handles, loc="upper right", ncol=2,
                  fontsize=13, frameon=True, columnspacing=1, handlelength=2)

    for bang, position in zip(bangs, positions):
        ax.text(*position, rf"$\chi=\chi_{{\{bang}}}$",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color="#8b3030" if bang == "max" else "#285b89")
    for label, x, index, offset, color, alignment in labels:
        ax.annotate(label, xy=(x, stationary_kurtoses(x)[index]), xytext=offset,
                    textcoords="offset points", ha="center", va=alignment,
                    fontsize=16, color=color, zorder=4)

    ax.set_xlabel(r"$\alpha$", fontsize=20)
    ax.set_ylabel(rf"$a_2^{{\{extremum}}}$", fontsize=20)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, labelsize=16, pad=5.5)
    fig.tight_layout()
    fig.savefig(f"a2{extremum}_chi_vs_alfa.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
