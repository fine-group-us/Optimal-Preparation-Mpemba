import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp

rc('font', family='serif', serif=['Times'])
rc('text', usetex=True)

# Reachable regions for the two restitution coefficients in Figure 2.
d = 3


def boundary(Ti, Ts, a2s, B):
    def rhs(t, y):
        T, a2 = y
        return (
            Ts**1.5*(1 + 3*a2s/16) - T**1.5*(1 + 3*a2/16),
            2*np.sqrt(T)*((1 - (Ts/T)**1.5)*a2 + B*(a2s - a2))
        )

    sol = solve_ivp(rhs, (0, 50), (Ti, a2s), method='Radau',
                    rtol=1e-10, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    T, indices = np.unique(sol.y[0], return_index=True)
    return T, sol.y[1, indices]


for panel, alpha in enumerate((0.4, 0.9), start=1):
    numerator = 16*(1 - alpha)*(1 - 2*alpha**2)
    a2s = numerator/(73 + 56*d - 24*d*alpha - 105*alpha
                     + 30*(1 - alpha)*alpha**2)
    a2hcs = numerator/(25 + 2*alpha**2*(alpha - 1) + 24*d
                       + alpha*(8*d - 57))
    B = a2hcs/(a2hcs - a2s)

    fig, ax = plt.subplots(figsize=(6, 4.09))
    for i in range(4):
        Tsmax = 10**(i + 1)
        Th, a2h = boundary(1, Tsmax, a2s, B)
        Tc, a2c = boundary(Tsmax, 1, a2s, B)
        T = np.geomspace(1, Tsmax, 2000)
        heating = np.interp(T, Th, a2h)
        cooling = np.interp(T, Tc, a2c)

        color = f'C{i}'
        ax.fill_between(T, np.minimum(heating, cooling),
                        np.maximum(heating, cooling), color=color,
                        alpha=0.12, lw=0, zorder=5-i)
        ax.plot(Th, a2h, lw=1.3, color=color, zorder=10,
                label=rf'$10^{{{i + 1}}}$')
        ax.plot(Tc, a2c, lw=1.3, color=color, zorder=10)

    ax.axhline(a2s, color='black', lw=1, zorder=9)
    ax.axhline(0, color='black', ls='--', lw=1, zorder=9)
    ax.axhline(a2hcs, color='black', ls='--', lw=1, zorder=9)
    yrange = abs(a2hcs)
    ax.text(1.4, a2s + 0.02*yrange, r'$a_2^{\rm s}$',
            fontsize=16, va='bottom')
    ax.text(1.4, a2hcs + (0.02 if panel == 1 else -0.15)*yrange,
            r'$a_2^{\rm HCS}$',
            fontsize=16, va='bottom')
    lower = min(0, a2hcs) - 0.2*yrange
    if panel == 1:
        lower -= 0.02  # Space for the legend.
    ax.set_ylim(lower, max(0, a2hcs) + 0.2*yrange)
    ax.set_xscale('log')
    ax.set_xlabel(r'$T_{\rm f}$', fontsize=20)
    ax.set_ylabel(r'$a_2$', fontsize=20)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', labelsize=16, pad=5.5)
    ax.margins(x=0)

    if panel == 1:
        ax.legend(title=r'$\chi_{\max}^{2/3}=T_{\rm s}^{\max}$', ncol=4,
                  fontsize=12, title_fontsize=14, frameon=True,
                  loc='lower center', columnspacing=1.2, handlelength=1.6)

    fig.tight_layout()
    fig.savefig(f'reachable_alfa{panel}.pdf', bbox_inches='tight')
    plt.close(fig)
