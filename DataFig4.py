import numpy as np
from scipy.integrate import solve_ivp


# Kurtosis extrema along the final bang, with chi_min = 1 and d = 3.
def extremum(initial_temperature, a2_s, B):
    # T is scaled by the final bang's stationary temperature and
    # A = (a2 - a2_s)/(a2_HCS - a2_s).
    def event(t, y):
        T, A = y
        inverse_T_power = T**(-1.5)
        return (B - 1)*(1 - inverse_T_power) - (B - 1 + inverse_T_power)*A

    event.direction = 1 if initial_temperature < 1 else -1
    event.terminal = True

    def rhs(t, y):
        T, A = y
        a2 = a2_s*(1 + A/(B - 1))
        return [1 + 3*a2_s/16 - T**1.5*(1 + 3*a2/16),
                2*np.sqrt(T)*event(t, y)]

    sol = solve_ivp(rhs, (0, 1), [initial_temperature, 0], method="LSODA",
                    events=event, rtol=1e-10, atol=1e-12)
    if not sol.t_events[0].size:
        raise RuntimeError("The integration did not reach a kurtosis extremum.")
    A = sol.y_events[0][0, 1]
    return a2_s*(1 + A/(B - 1))


d = 3
alpha_c = 1/np.sqrt(2)
alphas = np.sort(np.r_[np.linspace(0.001, 0.9999, 1000), alpha_c])
rows = []
for chi_max in (2, 10, 100, 1000):
    for alpha in alphas:
        if alpha == alpha_c:
            rows.append([chi_max, alpha, 0, 0])
            continue
        denominator = 73 + 56*d - 24*d*alpha - 105*alpha + 30*(1 - alpha)*alpha**2
        a2_s = 16*(1 - alpha)*(1 - 2*alpha**2)/denominator
        B = denominator/(16*(1 - alpha)*(3 + 2*d + 2*alpha**2))
        heating = extremum(chi_max**(-2/3), a2_s, B)
        cooling = extremum(chi_max**(2/3), a2_s, B)
        rows.append([chi_max, alpha, min(heating, cooling), max(heating, cooling)])

np.savetxt("DataFig4.dat", rows, header="chi_max alpha a2_min a2_max")
