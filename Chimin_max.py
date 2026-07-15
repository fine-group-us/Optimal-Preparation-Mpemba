import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

data = np.loadtxt("Chimaxvsalfa10.0.dat")
alphas = data[:, 0]
a2mins  = data[:, 1]

data = np.loadtxt("Chimaxvsalfa50.0.dat")
alphas2 = data[:, 0]
a2mins2 = data[:, 1]

data = np.loadtxt("Chimaxvsalfa100.0.dat")
alphas3 = data[:, 0]
a2mins3 = data[:, 1]

data = np.loadtxt("Chimaxvsalfa1000.0.dat")
alphas4 = data[:, 0]
a2mins4 = data[:, 1]

data = np.loadtxt("Chiminvsalfa0.1.dat")
alphas5 = data[:, 0]
a2mins5 = data[:, 1]

data = np.loadtxt("Chiminvsalfa0.05.dat")
alphas6 = data[:, 0]
a2mins6 = data[:, 1]

data = np.loadtxt("Chiminvsalfa0.01.dat")
alphas7 = data[:, 0]
a2mins7 = data[:, 1]

data = np.loadtxt("Chiminvsalfa0.001.dat")
alphas8 = data[:, 0]
a2mins8 = data[:, 1]

data = np.loadtxt("Simulation/a2opt_01Xmin.dat")
alphas_sim = data[:, 0]
a2mins_sim = data[:, 1]  

data = np.loadtxt("Simulation/a2opt_005Xmin.dat")
alphas_sim2 = data[:, 0]
a2mins_sim2 = data[:, 1]  

data = np.loadtxt("Simulation/a2opt_001Xmin.dat")
alphas_sim3 = data[:, 0]
a2mins_sim3 = data[:, 1]  

data = np.loadtxt("Simulation/a2opt_0001Xmin.dat")
alphas_sim4 = data[:, 0]
a2mins_sim4 = data[:, 1] 

data = np.loadtxt("Simulation/a2opt_10Xmax.dat")
alphas_sim5 = data[:, 0]
a2mins_sim5 = data[:, 1] 

data = np.loadtxt("Simulation/a2opt_50Xmax.dat")
alphas_sim6 = data[:, 0]
a2mins_sim6 = data[:, 1]

data = np.loadtxt("Simulation/a2opt_100Xmax.dat")
alphas_sim7 = data[:, 0]
a2mins_sim7 = data[:, 1]

data = np.loadtxt("Simulation/a2opt_1000Xmax.dat")
alphas_sim8 = data[:, 0]
a2mins_sim8 = data[:, 1]


fig, ax = plt.subplots(figsize=(6,4.1))
ax.set_xlim(0,1)
ax.set_ylim(-0.0001,0.0280)
ax.margins(x=0,y=0)
ax.plot(alphas[alphas<1/np.sqrt(2)], a2mins[alphas<1/np.sqrt(2)], 'r-')
ax.plot(alphas2[alphas2<1/np.sqrt(2)], a2mins2[alphas2<1/np.sqrt(2)], 'r:')
ax.plot(alphas3[alphas3<1/np.sqrt(2)], a2mins3[alphas3<1/np.sqrt(2)], 'r--')
ax.plot(alphas4[alphas4<1/np.sqrt(2)], a2mins4[alphas4<1/np.sqrt(2)], 'r-.')
ax.plot(alphas5[alphas5>1/np.sqrt(2)], np.abs(a2mins5[alphas5>1/np.sqrt(2)]), 'b-')
ax.plot(alphas6[alphas6>1/np.sqrt(2)], np.abs(a2mins6[alphas6>1/np.sqrt(2)]), 'b:')
ax.plot(alphas7[alphas7>1/np.sqrt(2)], np.abs(a2mins7[alphas7>1/np.sqrt(2)]), 'b--')
ax.plot(alphas8[alphas8>1/np.sqrt(2)], np.abs(a2mins8[alphas8>1/np.sqrt(2)]), 'b-.')
ax.plot(alphas_sim[alphas_sim>1/np.sqrt(2)], np.abs(a2mins_sim[alphas_sim>1/np.sqrt(2)]), linestyle='None', marker='o', markersize=5, label=r"$\chi_{\rm{min}} = 0.1$")
ax.plot(alphas_sim2[alphas_sim2>1/np.sqrt(2)], np.abs(a2mins_sim2[alphas_sim2>1/np.sqrt(2)]), linestyle='None', marker='s', markersize=5, label=r"$\chi_{\rm{min}} = 0.05$")
ax.plot(alphas_sim3[alphas_sim3>1/np.sqrt(2)], np.abs(a2mins_sim3[alphas_sim3>1/np.sqrt(2)]), linestyle='None', marker='^', markersize=5, label=r"$\chi_{\rm{min}} = 0.01$")
ax.plot(alphas_sim4[alphas_sim4>1/np.sqrt(2)], np.abs(a2mins_sim4[alphas_sim4>1/np.sqrt(2)]), linestyle='None', marker='v', markersize=5, label=r"$\chi_{\rm{min}} = 0.001$")
ax.plot(alphas_sim5[alphas_sim5<1/np.sqrt(2)], a2mins_sim5[alphas_sim5<1/np.sqrt(2)], linestyle='None', marker='D', markersize=5,  label=r"$\chi_{\rm{max}} = 10$")
ax.plot(alphas_sim6[alphas_sim6<1/np.sqrt(2)], a2mins_sim6[alphas_sim6<1/np.sqrt(2)], linestyle='None', marker='p', markersize=5, label=r"$\chi_{\rm{max}} = 50$")
ax.plot(alphas_sim7[alphas_sim7<1/np.sqrt(2)], a2mins_sim7[alphas_sim7<1/np.sqrt(2)], linestyle='None', marker='X', markersize=6, label=r"$\chi_{\rm{max}} = 100$")
ax.plot(alphas_sim8[alphas_sim8<1/np.sqrt(2)], a2mins_sim8[alphas_sim8<1/np.sqrt(2)], linestyle='None', marker='*', markersize=6, label=r"$\chi_{\rm{max}} = 1000$")
ax.set_xlabel(r"$\alpha$",fontsize=20)
ax.set_ylabel(r"$|a_2^{\min}|$",fontsize=20)
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,-2),useMathText=True)
offset = ax.yaxis.get_offset_text()
offset.set_size(12)
# ax.set_ylim(-0.02,0.19)
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
# ax.legend()
fig.tight_layout()
fig.savefig("a2min_chi_vs_alfa.pdf",format='pdf')
fig, ax = plt.subplots(figsize=(6,4.1))
ax.set_xlim(0,1)
ax.set_ylim(-0.0001,0.160)
ax.margins(x=0,y=0)
ax.plot(alphas[alphas>1/np.sqrt(2)], np.abs(a2mins[alphas>1/np.sqrt(2)]), 'r-')
ax.plot(alphas2[alphas2>1/np.sqrt(2)], np.abs(a2mins2[alphas2>1/np.sqrt(2)]), 'r:')
ax.plot(alphas3[alphas3>1/np.sqrt(2)], np.abs(a2mins3[alphas3>1/np.sqrt(2)]), 'r--')
ax.plot(alphas4[alphas4>1/np.sqrt(2)], np.abs(a2mins4[alphas4>1/np.sqrt(2)]), 'r-.')
ax.plot(alphas5[alphas5<1/np.sqrt(2)], np.abs(a2mins5[alphas5<1/np.sqrt(2)]), 'b-')
ax.plot(alphas6[alphas6<1/np.sqrt(2)], np.abs(a2mins6[alphas6<1/np.sqrt(2)]), 'b:')
ax.plot(alphas7[alphas7<1/np.sqrt(2)], np.abs(a2mins7[alphas7<1/np.sqrt(2)]), 'b--')
ax.plot(alphas8[alphas8<1/np.sqrt(2)], np.abs(a2mins8[alphas8<1/np.sqrt(2)]), 'b-.')
ax.plot(alphas_sim[alphas_sim<1/np.sqrt(2)], np.abs(a2mins_sim[alphas_sim<1/np.sqrt(2)]), linestyle='None', marker='o', markersize=5, label=r"$\chi_{\rm{min}} = 0.1$")
ax.plot(alphas_sim2[alphas_sim2<1/np.sqrt(2)], np.abs(a2mins_sim2[alphas_sim2<1/np.sqrt(2)]), linestyle='None', marker='s', markersize=5, label=r"$\chi_{\rm{min}} = 0.05$")
ax.plot(alphas_sim3[alphas_sim3<1/np.sqrt(2)], np.abs(a2mins_sim3[alphas_sim3<1/np.sqrt(2)]), linestyle='None', marker='^', markersize=5, label=r"$\chi_{\rm{min}} = 0.01$")
ax.plot(alphas_sim4[alphas_sim4<1/np.sqrt(2)], np.abs(a2mins_sim4[alphas_sim4<1/np.sqrt(2)]), linestyle='None', marker='v', markersize=5, label=r"$\chi_{\rm{min}} = 0.001$")
ax.plot(alphas_sim5[alphas_sim5>1/np.sqrt(2)], np.abs(a2mins_sim5[alphas_sim5>1/np.sqrt(2)]), linestyle='None', marker='D', markersize=5,  label=r"$\chi_{\rm{max}} = 10$")
ax.plot(alphas_sim6[alphas_sim6>1/np.sqrt(2)], np.abs(a2mins_sim6[alphas_sim6>1/np.sqrt(2)]), linestyle='None', marker='p', markersize=5, label=r"$\chi_{\rm{max}} = 50$")
ax.plot(alphas_sim7[alphas_sim7>1/np.sqrt(2)], np.abs(a2mins_sim7[alphas_sim7>1/np.sqrt(2)]), linestyle='None', marker='X', markersize=6, label=r"$\chi_{\rm{max}} = 100$")
ax.plot(alphas_sim8[alphas_sim8>1/np.sqrt(2)], np.abs(a2mins_sim8[alphas_sim8>1/np.sqrt(2)]), linestyle='None', marker='*', markersize=6, label=r"$\chi_{\rm{max}} = 1000$")
ax.set_xlabel(r"$\alpha$",fontsize=20)
ax.set_ylabel(r"$|a_2^{\max}|$",fontsize=20)
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,-2),useMathText=True)
offset = ax.yaxis.get_offset_text()
offset.set_size(12)
# ax.set_ylim(-0.02,0.19)
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
ax.legend()
fig.tight_layout()
fig.savefig("a2max_chi_vs_alfa.pdf",format='pdf')
d = 3
# ax.plot(alphas, 16 * (1 - alphas) * (1 - 2 * alphas**2) / (25+2*alphas**2*(alphas - 1) + 24*d + alphas*(8*d - 57)), 'g-')

