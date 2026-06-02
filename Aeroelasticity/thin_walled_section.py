"Cell airfoil & shear stress shear flow distribution"
import numpy as np
import matplotlib.pyplot as plt

phi_c=10*np.pi/180; tc=0.10; c=10
x=np.concatenate([np.arange(0,0.05+0.0025,0.0025),np.arange(0.05,0.1+0.005,0.005),np.arange(0.1,1.0+0.01,0.01)])*c
z=np.sqrt(c**2/(4*np.sin(phi_c/2)**2)-(x-c/2)**2)-c/(2*np.tan(phi_c/2))
dzdx=-(x-c/2)/(z+c/(2*np.tan(phi_c/2)))
theta=np.arctan(dzdx)
zp=5*tc*c*(0.2969*np.sqrt(x/c)-0.126*x/c-0.3516*(x/c)**2+0.2843*(x/c)**3-0.1015*(x/c)**4)
xu=x-zp*np.sin(theta); zu=z+zp*np.cos(theta)
xl=x+zp*np.sin(theta); zl=z-zp*np.cos(theta)

# ── Problem 1 ─────────────────────────────────────────────────────────────────
n=len(x)
dU=np.sqrt((xu[1:]-xu[:-1])**2+(zu[1:]-zu[:-1])**2)+np.sqrt((xl[1:]-xl[:-1])**2+(zl[1:]-zl[:-1])**2)
U=np.sum(dU)
dA=((zu[1:]+zu[:-1])/2-(zl[1:]+zl[:-1])/2)*(x[1:]-x[:-1])
A_sec=np.sum(dA)
# Open section J (Bredt-Batho)
J_open=np.trapz((1/3)*(2*zp)**3*np.sqrt(1+dzdx**2),x)
T=100000; tau_open=(T*2*zp/J_open)/144
idx_max=np.argmax(tau_open)
print("P1: Open section J = %.4e ft^4, J_open/144 in^4 = %.4e" % (J_open, J_open/144))
print("    Max tau = %.2f psi at x/c = %.4f" % (tau_open[idx_max], x[idx_max]/c))
# Closed single-cell
t_wall=0.12/12
J_closed=4*A_sec**2*t_wall/U
tau_closed=(T/(2*A_sec*t_wall))/144
print("P1: Closed cell J = %.4e ft^4,  tau = %.2f psi" % (J_closed, tau_closed))

# ── Problem 2 ─────────────────────────────────────────────────────────────────
# Cell boundaries at indices 37, 77 (0-based)
# Using same cross-section geometry
ts=0.12/12; tw=0.48/12; G=4.06e6*144
def cell_props(m,n,U_arr,A_arr):
    return np.sum(U_arr[m:n]), np.sum(A_arr[m:n])

dU_arr=np.sqrt((xu[1:]-xu[:-1])**2+(zu[1:]-zu[:-1])**2)+np.sqrt((xl[1:]-xl[:-1])**2+(zl[1:]-zl[:-1])**2)
dA_arr=((zu[1:]+zu[:-1])/2-(zl[1:]+zl[:-1])/2)*(x[1:]-x[:-1])
# Spar walls (vertical connections at indices 37,77)
U12_spar=np.sqrt((xu[37]-xl[37])**2+(zu[37]-zl[37])**2)
U23_spar=np.sqrt((xu[77]-xl[77])**2+(zu[77]-zl[77])**2)
U1=np.sum(dU_arr[:37])+U12_spar; A1=np.sum(dA_arr[:37])
U2=np.sum(dU_arr[37:77])+U23_spar; A2=np.sum(dA_arr[37:77])
U3=np.sum(dU_arr[77:])+U23_spar; A3=np.sum(dA_arr[77:])
D=np.array([
    [2*A1, 2*A2, 2*A3, 0],
    [(U1-U12_spar)/(G*ts)+U12_spar/(G*tw), -U12_spar/(G*tw), 0, -2*A1],
    [-U12_spar/(G*tw), (U2-U12_spar-U23_spar)/(G*(ts+tw))+(U12_spar+U23_spar)/(G*tw), -U23_spar/(G*tw), -2*A2],
    [0, -U23_spar/(G*tw), (U3-U23_spar)/(G*ts)+U23_spar/(G*tw), -2*A3]
])
B_d=np.array([1,0,0,0])
C_sol=np.linalg.solve(D,B_d)
J_3cell=1/(G*C_sol[3])
q1=C_sol[0]*T; q2=C_sol[1]*T; q3=C_sol[2]*T
tau1=(q1/ts)/144; tau2=(q2/(ts+tw))/144; tau3=(q3/ts)/144
tau12=((q1-q2)/tw)/144; tau23=((q2-q3)/tw)/144
print("P2: J_3cell = %.4e ft^4" % J_3cell)
print("    tau1=%.2f, tau2=%.2f, tau3=%.2f psi (cells)" % (tau1,tau2,tau3))
print("    tau_spar12=%.2f, tau_spar23=%.2f psi" % (tau12,tau23))

fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(xu,zu,'k',xl,zl,'k'); axes[0].set_aspect('equal'); axes[0].set_xlabel('x (ft)'); axes[0].set_ylabel('z (ft)'); axes[0].set_title('HW5 – Airfoil Cross-section'); axes[0].grid(True,linestyle='--',alpha=0.5)
axes[1].plot(x,tau_open); axes[1].set_xlabel('x (ft)'); axes[1].set_ylabel('tau_xy (psi)'); axes[1].set_title('P1 – Open-cell Shear Stress'); axes[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw05_plots.png',dpi=150); print("Saved hw05_plots.png")
