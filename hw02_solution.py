"""ME 184 – Homework 2 Problem 4
Part (a): Geometric twist distribution
Part (b): Lifting-line – cl and cl_alpha spanwise distributions
"""
import numpy as np
import matplotlib.pyplot as plt

def atmosphere(h_ft):
    psl=2116.2; Tsl=518.69; a1=-3.56616e-3/Tsl; a2=0.54864e-3/Tsl
    h1=36089.0; h2=65617.0; g=32.174; R=1716.56
    b1=-g/(R*Tsl*a1); b2=-g/(R*Tsl*a2)
    hi=float(h_ft); theta=1+a1*hi if hi<=h1 else (1+a1*h1 if hi<=h2 else 1+a1*h1+a2*(hi-h2))
    delta=theta**b1 if hi<=h1 else (theta**b1*np.exp(g*(h1-hi)/(R*Tsl*(1+a1*h1))) if hi<=h2 else (1+a1*h1)**(b1-b2)*np.exp(g*(h1-h2)/(R*Tsl*(1+a1*h1)))*theta**b2)
    p=psl*delta; T=Tsl*theta; rho=p/(1716.0*T); a_s=np.sqrt(1.4*1716.0*T)
    return p,T,rho,a_s

# ── Parameters ────────────────────────────────────────────────────────────────
W=20000; h_alt=40000; Md=0.5
p,_,_,_=atmosphere(h_alt)
gamma=1.4; q=gamma/2*p*Md**2
c=10; b=80; S=c*b; alpha0=-2*np.pi/180; alphad=1*np.pi/180
clalphad=2*np.pi/np.sqrt(1-Md**2); AR=b**2/S; CLd=W/(q*S)

# ── Part (a) twist ────────────────────────────────────────────────────────────
y=np.linspace(-b/2,b/2,101)
xi=alpha0-alphad+CLd/(np.pi*AR)+4*CLd/(np.pi*AR*c*clalphad)*np.sqrt(b**2-4*y**2)

# ── Part (b) Lifting-line ────────────────────────────────────────────────────
alpha=2*np.pi/180; Minf=0.4
clalphas=2*np.pi/np.sqrt(1-Minf**2); N=9
phi=np.linspace(0.01,np.pi-0.01,101)
xi_phi=alpha0-alphad+CLd/(np.pi*AR)+4*CLd/(np.pi*AR*c*clalphad)*b*np.sin(phi)
cls=clalphas*(alpha+xi_phi-alpha0)
C=np.zeros((N,N)); B=np.zeros(N)
for m in range(N):
    for n in range(N):
        f=0.5*c*clalphas*(n+1)*np.sin((n+1)*phi)*np.sin((m+1)*phi)/np.sin(phi)
        anm=np.trapz(f[1:-1],phi[1:-1])
        C[m,n]=np.pi*b+anm if n==m else anm
    B[m]=np.trapz(0.5*c*cls*np.sin((m+1)*phi),phi)
A_coeff=np.linalg.solve(C,B)

phi_plot=np.linspace(0,np.pi,101); y_plot=-b/2*np.cos(phi_plot)
cl=np.zeros(len(phi_plot)); kappa=np.ones(len(phi_plot))
for n in range(N):
    cl+=4*AR*A_coeff[n]*np.sin((n+1)*phi_plot)
    if n>0: kappa+=A_coeff[n]*(n+1)*np.sin((n+1)*phi_plot)/(A_coeff[0]*np.sin(phi_plot))
kappa[0]=sum(1+(A_coeff[n]/A_coeff[0])*(n+1)**2 for n in range(1,N))
kappa[-1]=kappa[0]
kappabar=np.trapz(kappa*np.sin(phi_plot),phi_plot)/2
CLalpha=clalphas/(1+clalphas*kappabar/(np.pi*AR))
clalpha=clalphas*(1-CLalpha*kappa/(np.pi*AR))

fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].plot(2*y/b,xi*180/np.pi); axes[0].set_xlabel('2y/b'); axes[0].set_ylabel('xi (deg)'); axes[0].set_title('(a) Twist'); axes[0].grid(True,linestyle='--',alpha=0.5)
axes[1].plot(2*y_plot/b,cl); axes[1].set_xlabel('2y/b'); axes[1].set_ylabel('c_l'); axes[1].set_title('(b) Section lift coeff'); axes[1].grid(True,linestyle='--',alpha=0.5)
axes[2].plot(2*y_plot/b,clalpha); axes[2].set_xlabel('2y/b'); axes[2].set_ylabel('c_la'); axes[2].set_title('(b) Section lift curve slope'); axes[2].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw02_plots.png',dpi=150); print("Saved hw02_plots.png")
