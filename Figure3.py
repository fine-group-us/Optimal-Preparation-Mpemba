import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import griddata

rc('font', family='serif', serif=['Times'])
rc('text', usetex=True)

# Run DataFig3.py to calculate the inversion amplitudes.
for alpha, delta_max in ((0.4, 0.4), (0.9, 0.01)):
    Tc, Th, Mp = np.loadtxt(f'DataFig3_alpha{alpha}.dat', unpack=True)
    points, indices = np.unique(np.column_stack((Tc, Th - Tc)), axis=0,
                                return_index=True)
    T, delta = np.meshgrid(np.linspace(Tc.min(), Tc.max(), 1000),
                           np.linspace(0, delta_max, 1000))
    amplitude = griddata(points, Mp[indices], (T, delta), rescale=True)
    amplitude = np.where(T + delta <= 1000**(2/3),
                         np.maximum(amplitude, 0), np.nan)
    amplitude = np.ma.masked_invalid(amplitude)

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(T, delta, amplitude, shading='auto', cmap='jet',
                         vmin=0, rasterized=True)
    ax.contour(T, delta, amplitude, levels=[1e-8], colors='white',
               linewidths=2, zorder=5)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r'$\rm{Mp}(T_{\rm c,i},\delta)$', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r'$T_{\rm c,i}$', fontsize=20)
    ax.set_ylabel(r'$\delta$', fontsize=20)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=16)
    ax.set_xlim(1, 1000**(2/3))
    ax.set_ylim(0, delta_max)
    fig.tight_layout()
    fig.savefig(f'DensityChimax1000_alpha{alpha}.pdf', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
