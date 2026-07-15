import numpy as np
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

opt = sys.argv[1]
#This code solves the system of differential equations describing the granular gas in the linear first Sonine aproximation for a constant control chi=chimax. To generate different types of plot, change the True/False parameters below. By default, it generates the plots corresponding to Fig 1 and Fig 2 of the article.
# If write_data True, it generates the data used to generate Fig 4. If density=True it generates the data used in Fig 3
graphs = True
Tgraphs = False
write_data = False
density = False

if opt == "max":
    max = True
else:    max = False
# Parámetros
d = 3
n = 4
Chimaxs = [10.0]
Tsmaxs = np.array(Chimaxs) ** (2/3)

T0 = 1.0
p20 = -1.0
if max:
    p20 = 1.0
filename = "Chimaxvsalfa" + str(Chimaxs[0]) + ".dat"

alphas = np.linspace(0.001, 1/np.sqrt(2), 1000)  # Minimisation
if max:
    alphas = np.linspace(1/np.sqrt(2)+0.001,0.9999, 1000)  # Maximisation

if Tgraphs:
    alphas = [0.3,0.5,0.8,0.9]
    Chimaxs = np.linspace(1+1e-3,10**2,1001)

Tsmaxs = np.array(Chimaxs) ** (2/3)

if density:
    alphas = np.linspace(0.001, 1/np.sqrt(2)-0.001, 100)  # Minimisation
    if max:
        alphas = np.linspace(1/np.sqrt(2)+0.001,0.9999, 50)  # Maximisation
    Chimaxs = np.linspace(1+1e-3,10**2,1001)
    Tsmaxs = np.array(Chimaxs) ** (2/3)
    filename="densityplot_Chimax.dat"
    file = open(filename, 'a')


i=0
tfs = []
a2mins = []
tfmax = 0
if graphs:
    fig_w = 6
    fig_h = 4
    axis_x = 0.175
    axis_x_width = 0.65
    axis_y = 0.15
    axis_y_height = 0.75
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
    ax1.set_position([axis_x, axis_y, axis_x_width, axis_y_height])
    ax1.set_xlabel(r"$t$",fontsize=20)
    if max:
        ax1.set_ylabel(r"$a_2(t)$",fontsize=20)
    else:
        ax1.set_ylabel(r"$a_2(t)$",fontsize=20)
    ax1.yaxis.set_label_coords(-0.18, 0.5)
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.set_position([axis_x, axis_y, axis_x_width, axis_y_height])
    ax2.set_xlabel(r"$t$",fontsize=20)
    if max:
        ax2.set_ylabel(r"$p_1(t)$",fontsize=20)
    else:
        ax2.set_ylabel(r"$p_1(t)$",fontsize=20)
    ax2.yaxis.set_label_coords(-0.18, 0.5)
    fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
    ax3.set_position([axis_x, axis_y, axis_x_width, axis_y_height])
    ax3.set_xlabel(r"$t$",fontsize=20)
    ax3.set_ylabel(r"$\bar{p}_2(t)$",fontsize=20)
    ax3.yaxis.set_label_coords(-0.18, 0.5)
    fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
    ax4.set_position([axis_x, axis_y, axis_x_width, axis_y_height])
    ax4.set_xlabel(r"$t$",fontsize=20)
    ax4.set_ylabel(r'$\Phi(t)$',fontsize=20)
    ax4.yaxis.set_label_coords(-0.18, 0.5)
if Tgraphs:
    figT, axT = plt.subplots(figsize=(6,4))

for a in alphas:
    a2minsT = []
    for Tsmax in Tsmaxs:
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

        # Sistema dinámico
        def f(T, a2, Ts):
            return Ts**(3/2) * (1 + (3 * a2s(a)) / 16) - T**(3/2) * (1 + (3 * a2) / 16)

        def g2(T, a2, Ts):
            return 2 * np.sqrt(T) * ((1 - (Ts/T)**(3/2)) * a2 + B(a) * (a2s(a) - a2))

        # Derivadas analíticas
        def dfdT(T, a2, Ts):
            return -(3/2) * np.sqrt(T) * (1 + (3 * a2) / 16)

        def dfda2(T, a2, Ts):
            return -(3/16) * T**(3/2)

        def dg2dT(T, a2, Ts):
            term1 = (1 - (Ts/T)**(3/2)) * a2 + B(a) * (a2s(a) - a2)
            term2 = a2 * (3/2) * (Ts**(3/2)) * T**(-5/2)
            return (1/np.sqrt(T)) * term1 + 2*np.sqrt(T) * term2

        def dg2da2(T, a2, Ts):
            return 2*np.sqrt(T) * ((1 - (Ts/T)**(3/2)) - B(a))

        # Condiciones iniciales
        a20 = a2s(a)
        p10 = - (p20 * g2(T0, a20, Tsmax)) / f(T0, a20, Tsmax)

        # Sistema completo (estado + coste)  
        def system(t, y):
            T, a2, psi1, psi2 = y
            
            dTdt = f(T, a2, Tsmax)
            da2dt = g2(T, a2, Tsmax)
            
            dpsi1dt = -psi1 * dfdT(T, a2, Tsmax) - psi2 * dg2dT(T, a2, Tsmax)
            dpsi2dt = -psi1 * dfda2(T, a2, Tsmax) - psi2 * dg2da2(T, a2, Tsmax)
            
            return [dTdt, da2dt, dpsi1dt, dpsi2dt]

        # Evento para detener cuando psi2 = 0
        def event_psi2_zero(t, y):
            return y[2]

        event_psi2_zero.terminal = True
        event_psi2_zero.direction = 0

        # Integración
        t_span = (0, 4)
        y0 = [T0, a20, p10, p20]
        sol = solve_ivp(system, t_span, y0, method='Radau', dense_output=True, events=event_psi2_zero)
        tf = sol.t_events[0][0]
        tfs.append(tf)
        a2mins.append(sol.sol(tf)[1])
        a2minsT.append(sol.sol(tf)[1])
        if density:
            file.write(f"{a} {Tsmax} {sol.sol(tf)[1]}\n")
        t_vals = np.linspace(0, tf, 500)
        T_vals, a2_vals, psi1_vals, psi2_vals = sol.sol(t_vals)
        def switching_function(T, a2, psi1, psi2):
            return psi1 * (1 + (3/16) * a2s(a)) - 2/T * a2 * psi2
        sw_vals = switching_function(T_vals, a2_vals, psi1_vals, psi2_vals)
        if graphs and i in [250,500,750]:
            if tf > tfmax:
                tfmax = tf
            alpha_str = f"{a:.2f}".rstrip('0').rstrip('.')
            if max:
                ax1.plot(t_vals, a2_vals,'-', label=rf"$\alpha={alpha_str}$")
                ax1.plot(t_vals[-1], a2_vals[-1], '.k')
                ax1.ticklabel_format(style='sci', axis='y', scilimits=(-2,-2),useMathText=True)
                offset = ax1.yaxis.get_offset_text()
                offset.set_size(12)
            else:
                ax1.plot(t_vals, a2_vals,'-', label=rf"$\alpha={alpha_str}$")
                ax1.plot(t_vals[-1], a2_vals[-1], '.k')

            if max:
                ax2.plot(t_vals, psi1_vals,'-', label=rf"$\alpha={alpha_str}$")
                ax2.plot(t_vals[-1], psi1_vals[-1], '.k')
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2,-2),useMathText=True)
                offset = ax2.yaxis.get_offset_text()
                offset.set_size(12)
            else:
                ax2.plot(t_vals, psi1_vals,'-', label=rf"$\alpha={alpha_str}$")
                ax2.plot(t_vals[-1], psi1_vals[-1], '.k')

            ax3.plot(t_vals, psi2_vals,'--', label=rf"$\alpha={alpha_str}$")
            ax4.plot(t_vals, sw_vals,'-', label=rf"$\alpha={alpha_str}$")
            ax3.plot(t_vals[-1], psi2_vals[-1], '.k')
            ax4.plot(t_vals[-1], sw_vals[-1], '.k')


        i = i+1
    if Tgraphs:
        if a < 1/np.sqrt(2):
            axT.plot(Chimaxs, a2minsT, '-')
        else :
            axT.plot(Chimaxs, a2minsT, '--')
        axT.set_xlabel(r"$\chi_{\max}$",fontsize=20)
        axT.set_ylabel(r"$a_2^{opt}$",fontsize=20)
        axT.margins(x=0,y=0)
        axT.set_xlim(1,100)
        axT.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
        figT.tight_layout()
        figT.savefig("a2opt_Chimax.pdf",format='pdf')
if density:
    file.close()
if graphs:
    ax1.set_xlim(0,tfmax+0.05*tfmax)
    ax2.set_xlim(0,tfmax+0.05*tfmax)
    ax3.set_xlim(0,tfmax+0.05*tfmax)
    ax4.set_xlim(0,tfmax+0.05*tfmax)


    if max:
        # fig1.legend(fontsize=16)
        ax1.margins(x=0,y=0)
        ax1.set_ylim(-0.0115, -0.0034)
        ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
        # fig1.tight_layout()
        fig1.savefig("Imagesmax/1bangChimax_a2max.pdf",format='pdf')
        # fig2.legend(fontsize=16)
        ax2.margins(x=0,y=0)
        ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
        ax3_twin = ax2.twinx()
        ax3_twin.set_position(ax2.get_position())
        ax3_twin.set_ylim(0,10.0)
        ax3_twin.margins(x=0,y=0)
        ax3_twin.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=True, labelsize=16)
        ax3_twin.set_ylabel(r"$\bar{p}_2(t)$",fontsize=20)
        ax3_twin.yaxis.set_label_coords(1.1, 0.5)
        for line in ax3.get_lines():
            ax3_twin.plot(
                line.get_xdata(),
                line.get_ydata(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                color=line.get_color(),
                label=line.get_label()
            )
        # ax2.spines['top'].set_visible(False)
        # ax3_twin.spines['top'].set_visible(False)
        ax2.set_ylim(-0.023,0.0003)
        fig2.savefig("Imagesmax/1bangChimax_p1p2max.pdf",format='pdf')
        # fig4.legend(fontsize=16)
        ax4.margins(x=0,y=0)
        ax4.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
        # fig4.tight_layout()
        ax4.set_ylim(0,0.0576)
        fig4.savefig("Imagesmax/1bangChimax_switchmax.pdf",format='pdf')
        # fig5, ax5 = plt.subplots(figsize=(6,4))
        # ax5.plot(alphas, tfs, '-')
        # ax5.set_xlabel(r"$\alpha$",fontsize=20)
        # ax5.set_ylabel(r"$t_f$",fontsize=20)
        # ax5.margins(x=0,y=0)
        # ax5.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
        # fig5.tight_layout()
        # fig5.savefig("Imagesmax/1bangChimax_tf_vs_alphamax.pdf",format='pdf')
        # fig6, ax6 = plt.subplots(figsize=(6,4))
        # ax6.plot(alphas, a2mins, '-')
        # ax6.set_xlabel(r"$\alpha$",fontsize=20)
        # ax6.set_ylabel(r"$a_2^{min}$",fontsize=20)
        # ax6.margins(x=0,y=0)
        # ax6.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
        # fig6.tight_layout()
        # fig6.savefig("Imagesmax/1bangChimax_a2_vs_alphamax.pdf",format='pdf')


    else:
        # fig1.legend(fontsize=16)
        ax1.set_ylim(0, 0.08)
        ax1.margins(x=0,y=0)
        ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
        # fig1.tight_layout()
        fig1.savefig("Images/1bangChimax_a2min.pdf",format='pdf')
        # fig2.legend(fontsize=16)
        ax2.margins(x=0,y=0)
        ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
        ax3_twin = ax2.twinx()
        ax3_twin.set_position(ax2.get_position())
        ax3_twin.margins(x=0,y=0)
        ax3_twin.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=True, labelsize=16)
        ax3_twin.set_ylabel(r"$\bar{p}_2(t)$",fontsize=20)
        ax3_twin.yaxis.set_label_coords(1.15, 0.5)
        for line in ax3.get_lines():
            ax3_twin.plot(
                line.get_xdata(),
                line.get_ydata(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                color=line.get_color(),
                label=line.get_label()
            )
        # fig2.tight_layout()
        # ax2.spines['top'].set_visible(False)
        # ax3_twin.spines['top'].set_visible(False)
        ax2.set_ylim(-0.115,0.002)
        ax3_twin.set_ylim(-7.25,0)
        fig2.savefig("Images/1bangChimax_p1p2min.pdf",format='pdf')
        # fig4.legend(fontsize=16)
        ax4.margins(x=0,y=0)
        ax4.set_yticks([0.04, 0.08, 0.12, 0.16])
        ax4.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16,pad=5.5)
        ax4.set_ylim(0,0.175)
        # fig4.tight_layout()
        fig4.savefig("Images/1bangChimax_switchmin.pdf",format='pdf')
        # fig5, ax5 = plt.subplots(figsize=(6,4))
        # ax5.plot(alphas, tfs, '-')
        # ax5.set_xlabel(r"$\alpha$",fontsize=20)
        # ax5.set_ylabel(r"$t_f$",fontsize=20)
        # ax5.margins(x=0,y=0)
        # ax5.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
        # fig5.tight_layout()
        # fig5.savefig("Images/1bangChimax_tf_vs_alphamin.pdf",format='pdf')
        # fig6, ax6 = plt.subplots(figsize=(6,4))
        # ax6.plot(alphas, a2mins, '-')
        # ax6.set_xlabel(r"$\alpha$",fontsize=20)
        # ax6.set_ylabel(r"$a_2^{min}$",fontsize=20)
        # ax6.margins(x=0,y=0)
        # ax6.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
        # fig6.tight_layout()
        # fig6.savefig("Images/1bangChimax_a2_vs_alphamin.pdf",format='pdf')
if write_data:
    with open(filename, 'a') as f:
        for alpha, a2min in zip(alphas, a2mins):
            f.write(f"{alpha} {a2min}\n")

if False:
    fig6, ax6 = plt.subplots(figsize=(6,4))
    ax6.plot(Tsmaxs, tfs, '-')
    ax6.set_xlabel(r"$T_{s,max}$",fontsize=20)
    ax6.set_ylabel(r"$t_f$",fontsize=20)
    ax6.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=16)
    fig6.tight_layout()
    fig6.savefig("Imagesmax/1bangTmax_tf_vs_tsmax.pdf",format='pdf')
    # plt.plot(t_vals, psi1_vals, label="psi1(t)")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$p_1(t)$")

    # plt.figure()
    # plt.plot(t_vals, psi2_vals, label="psi2(t)")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$\bar{p}_2(t)$")

    # plt.figure()
    # plt.plot(t_vals, sw_vals,'.', label="Switching function")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r'$\phi(t)$')
