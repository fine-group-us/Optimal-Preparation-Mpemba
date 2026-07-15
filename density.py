import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
import matplotlib.tri as mtri
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

d=3

def a2s(a):
    return (16 * (1 - a) * (1 - 2 * a**2)) / (
        73 + 56*d - 24*d*a - 105*a + 30*(1 - a)*a**2
    )

def a2hcs(a):
    return (16 * (1 - a) * (1 - 2 * a**2)) / (
        25 + 2*a*(a - 1) + 24*d + a*(8*d - 57)
    )

def B(a):
    return a2hcs(a) / (a2hcs(a) - a2s(a))

data = np.loadtxt('densityplot_Chimax.dat')
Tmax = data[:,1]
Chimax = Tmax**(3/2)
alfas_chimax = data[:,0]
a2_chimax = data[:,2]
mask = alfas_chimax < 1/np.sqrt(2)
mask1= alfas_chimax > 1/np.sqrt(2)
data = np.loadtxt('densityplot_Chimin.dat')
Tmin = data[:,1]
Chimin = Tmin**(3/2)
alfas_chimin = data[:,0]
a2_chimin = data[:,2]
mask2 = alfas_chimin > 1/np.sqrt(2)
mask3= alfas_chimin < 1/np.sqrt(2)

fig = plt.figure(figsize=(8.5, 5))

gs = fig.add_gridspec(
    1, 2,
    left=0.2,
    right=0.78,
    width_ratios=[0.71, 0.29],
    wspace=0.025
)
ax  = fig.add_subplot(gs[0, 0])
axr = fig.add_subplot(gs[0, 1])
cax_left  = fig.add_axes([0.08, ax.get_position().y0, 0.025, ax.get_position().height])
cax_right = fig.add_axes([0.88, axr.get_position().y0, 0.025, axr.get_position().height])

triang = mtri.Triangulation(
	np.concatenate((alfas_chimax[mask], np.linspace(0, 1/np.sqrt(2), 50))),
	np.concatenate((1/Chimax[mask], np.zeros(50)))
)
cf = ax.tripcolor(
    triang,
    np.concatenate((a2_chimax[mask], np.zeros(50))),
    cmap='jet', shading='gouraud'
)
triang_r = mtri.Triangulation(
	np.concatenate((alfas_chimin[mask2], np.linspace(1/np.sqrt(2), 1, 50))),
	np.concatenate((Chimin[mask2], np.zeros(50)))
)
cf_r = axr.tripcolor(
    triang_r,
    np.concatenate((-a2_chimin[mask2], -a2hcs(np.linspace(1/np.sqrt(2), 1, 50)))),
    cmap='jet', shading='gouraud'
)
ax.set_xlabel(r'$\alpha$', fontsize=20)
ax.xaxis.set_label_coords(0.7, -0.07)
ax.set_ylabel(r'$\chi_{\max}^{–1}$', fontsize=20)
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelleft=True, labelsize=16,pad=6)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(0, 1)
ax.set_xlim(0, 1/np.sqrt(2))
cbar = fig.colorbar(cf, cax=cax_left,location='left')
cbar.set_label(r'$a_2^{\min}$', fontsize=16, loc='top', rotation=0, labelpad=-35)
cbar.ax.tick_params(labelsize=16)
ax.margins(x=0,y=0)

axr.set_ylabel(r'$\chi_{\min}$', fontsize=20)
axr.yaxis.set_label_position('right')
axr.set_xticks(np.arange(0, 1.01, 0.2))
axr.set_yticks(np.arange(0, 1.01, 0.2))
axr.set_ylim(0, 1)
axr.set_xlim(1/np.sqrt(2), 1)
axr.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=True, labelleft=False, labelsize=16,pad=6)

cbarr = fig.colorbar(cf_r, cax=cax_right)
cbarr.set_label(r'$-a_2^{\min}$', fontsize=16, rotation=0)
cbarr.ax.yaxis.set_label_coords(2, 1.06)

cbarr.ax.tick_params(labelsize=16)
axr.margins(x=0,y=0)
fig.savefig(
    'densityplot_min.pdf',
    format='pdf',
)

fig = plt.figure(figsize=(8.5, 5))

gs = fig.add_gridspec(
    1, 2,
    left=0.2,
    right=0.78,
    width_ratios=[0.71, 0.29],
    wspace=0.025
)
ax  = fig.add_subplot(gs[0, 0])
axr = fig.add_subplot(gs[0, 1])
cax_left  = fig.add_axes([0.08, ax.get_position().y0, 0.025, ax.get_position().height])
cax_right = fig.add_axes([0.88, axr.get_position().y0, 0.025, axr.get_position().height])

triang = mtri.Triangulation(
	np.concatenate((alfas_chimin[mask3], np.linspace(0, 1/np.sqrt(2), 50))),
	np.concatenate((Chimin[mask3], np.zeros(50)))
)
cf = ax.tripcolor(
    triang,
    np.concatenate((a2_chimin[mask3], a2hcs(np.linspace(0, 1/np.sqrt(2), 50)))),
    cmap='jet', shading='gouraud'
)
triang_r = mtri.Triangulation(
	np.concatenate((alfas_chimax[mask1], np.linspace(1/np.sqrt(2), 1, 50))),
	np.concatenate((1/Chimax[mask1], np.zeros(50)))
)
cf_r = axr.tripcolor(
    triang_r,
    np.concatenate((-a2_chimax[mask1],np.zeros(50))),
    cmap='jet', shading='gouraud'
)
ax.set_xlabel(r'$\alpha$', fontsize=20)
ax.xaxis.set_label_coords(0.7, -0.07)
ax.set_ylabel(r'$\chi_{\min}$', fontsize=20)
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelleft=True, labelsize=16,pad=6)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(0, 1)
ax.set_xlim(0, 1/np.sqrt(2))
cbar = fig.colorbar(cf, cax=cax_left,location='left')
cbar.set_label(r'$a_2^{\max}$', fontsize=16, loc='top', rotation=0, labelpad=-35)
cbar.ax.tick_params(labelsize=16)
ax.margins(x=0,y=0)

axr.set_ylabel(r'$\chi_{\max}^{-1}$', fontsize=20)
axr.yaxis.set_label_position('right')
axr.set_xticks(np.arange(0, 1.01, 0.2))
axr.set_yticks(np.arange(0, 1.01, 0.2))
axr.set_ylim(0, 1)
axr.set_xlim(1/np.sqrt(2), 1)
axr.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=True, labelleft=False, labelsize=16,pad=6)

cbarr = fig.colorbar(cf_r, cax=cax_right)
cbarr.set_label(r'$-a_2^{\max}$', fontsize=16, rotation=0)
cbarr.ax.yaxis.set_label_coords(2, 1.06)

cbarr.ax.tick_params(labelsize=16)
axr.margins(x=0,y=0)
fig.savefig(
    'densityplot_max.pdf',
    format='pdf',
)
