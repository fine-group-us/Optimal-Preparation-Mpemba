import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, identity


def data(alpha, delta_max):
    d = 3
    Tsmax = 1000**(2/3)
    numerator = 16*(1 - alpha)*(1 - 2*alpha**2)
    a2s = numerator/(73 + 56*d - 24*d*alpha - 105*alpha
                     + 30*(1 - alpha)*alpha**2)
    a2hcs = numerator/(25 + 2*alpha**2*(alpha - 1) + 24*d
                       + alpha*(8*d - 57))
    B = a2hcs/(a2hcs - a2s)

    def dT(T, a2, Ts):
        return Ts**1.5*(1 + 3*a2s/16) - T**1.5*(1 + 3*a2/16)

    def da2(T, a2, Ts):
        return 2*np.sqrt(T)*((1 - (Ts/T)**1.5)*a2 + B*(a2s - a2))

    # Find the preparation boundaries once for each panel.
    def boundary(Ti, Tend, Ts):
        sol = solve_ivp(lambda T, y: da2(T, y, Ts)/dT(T, y, Ts),
                        (Ti, Tend), [a2s], method='Radau', rtol=1e-10,
                        atol=1e-14, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)

        def evaluate(T):
            T = np.asarray(T)
            return np.where(T == Ts, a2s,
                            sol.sol(np.clip(T, min(Ti, Tend), max(Ti, Tend)))[0])

        return evaluate

    heating = boundary(1, Tsmax*(1 - 1e-6), Tsmax)
    cooling = boundary(Tsmax, 1 + 1e-6, 1)
    a2min, a2max = (heating, cooling) if alpha == 0.4 else (cooling, heating)
    times = np.r_[0., np.geomspace(1e-6, 20, 6000)]
    n_hot = 200
    S = identity(n_hot + 1, format='csr')
    sparsity = bmat([[S, S], [S, S]], format='csr')

    # Integrate the 200 hot samples together with their cold sample.
    def relax(Tc):
        Th = np.linspace(Tc, min(Tc + delta_max, Tsmax), n_hot)
        T0 = np.append(Th, Tc)
        a0 = np.append(a2max(Th), a2min(Tc))

        def rhs(t, y):
            T, a2 = y[:n_hot + 1], y[n_hot + 1:]
            return np.concatenate((dT(T, a2, 1), da2(T, a2, 1)))

        sol = solve_ivp(rhs, (0, 20), np.concatenate((T0, a0)),
                        method='Radau', t_eval=times, jac_sparsity=sparsity,
                        rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError(sol.message)
        Mp = (sol.y[n_hot] - sol.y[:n_hot]).max(axis=1)
        return np.column_stack((np.full(n_hot, Tc), Th,
                                np.where(Mp > 1e-8, Mp, 0)))

    rows = Parallel(n_jobs=-1)(delayed(relax)(Tc)
                              for Tc in np.linspace(1, Tsmax, 200))
    np.savetxt(f'DataFig3_alpha{alpha}.dat', np.vstack(rows))


if __name__ == '__main__':
    data(0.4, 0.4)
    data(0.9, 0.01)
