# Lateral Force/Moment Balance, Aileron Deflection for Bending/Torsuin Moments
import numpy as np
import matplotlib.pyplot as plt

def atmosphere(h_ft):
    psl=2116.2; Tsl=518.69; a1=-3.56616e-3/Tsl; a2=0.54864e-3/Tsl
    h1=36089.0; h2=65617.0; g=32.174; R=1716.56; b1=-g/(R*Tsl*a1); b2=-g/(R*Tsl*a2)
    hi=float(h_ft)
    if hi<=h1: theta=1+a1*hi; delta=theta**b1
    elif hi<=h2: theta=1+a1*h1; delta=theta**b1*np.exp(g*(h1-hi)/(R*Tsl*(1+a1*h1)))
    else: theta=1+a1*h1+a2*(hi-h2); delta=(1+a1*h1)**(b1-b2)*np.exp(g*(h1-h2)/(R*Tsl*(1+a1*h1)))*theta**b2
    p=psl*delta; T=Tsl*theta; rho=p/(1716.0*T); a_s=np.sqrt(1.4*1716.0*T)
    return p,T,rho,a_s

b=124.2572; M=0.5; h_alt=10000
p,T,rho,a_s=atmosphere(h_alt); V=M*a_s
Ybeta=-687097; Ydr=688338; nbeta=41765779; ndr=-44295916; Tbar=13271; ye=21.6
A_mat=np.array([[Ybeta,Ydr],[nbeta,ndr]]); B_vec=np.array([0,-Tbar*ye])
x_sol=np.linalg.solve(A_mat,B_vec); beta=x_sol[0]; dr=x_sol[1]
print("(a) beta = %.6f rad (%.4f deg),  delta_r = %.6f rad (%.4f deg)" % (beta,np.degrees(beta),dr,np.degrees(dr)))

cr=146.6092-123.3152; ct=155.2385-147.4552
cfr=146.6092-139.1505; cft=155.2385-152.5688
b2=2*(32.9545-10.6970); S=(cr+ct)/2*b2; AR=b2**2/S
Lambdaa=np.arctan(((155.2385+147.4552)/2-(146.6092+123.3152)/2)/(b2/2))
Lambdaf=np.arctan((152.5688-139.1505)/(b2/2))
phi=np.linspace(0.01,np.pi-0.01,101)
c=(ct-cr)*np.abs(np.cos(phi))+cr
cf=(cft-cfr)*np.abs(np.cos(phi))+cfr
cs=1-2*cf/c
clalphas=2*np.pi*np.cos(Lambdaa)/np.sqrt(1-M**2*np.cos(Lambdaa)**2)
cldeltas=2*np.pi*np.cos(Lambdaf)/np.sqrt(1-M**2*np.cos(Lambdaf)**2)/np.pi*(np.arccos(cs)+np.sqrt(1-cs**2))
cls=clalphas*beta-cldeltas*dr
N=5; C_mat=np.zeros((N,N)); B_vp=np.zeros(N)
for m in range(N):
    for n in range(N):
        f=0.5*c*clalphas*(n+1)*np.sin((n+1)*phi)*np.sin((m+1)*phi)/np.sin(phi)
        anm=np.trapz(f[1:-1],phi[1:-1])
        C_mat[m,n]=np.pi*b2+anm if n==m else anm
    B_vp[m]=np.trapz(0.5*c*cls*np.sin((m+1)*phi),phi)
A_c=np.linalg.solve(C_mat,B_vp)
phi2=np.linspace(np.pi/2,np.pi,101); y2=-b2/2*np.cos(phi2)
l=np.zeros(len(phi2))
for n in range(N): l+=rho*V**2*2*b2*A_c[n]*np.sin((n+1)*phi2)
Lambda=np.arctan((147.4552+0.4*ct-123.3152-0.4*cr)/(b2/2))
c2=(ct-cr)*np.abs(np.cos(phi2))+cr; e2=0.4*c2-0.25*c2
fz=l*np.cos(Lambda); mx=l*e2*np.cos(Lambda)**2; my=-l*e2*np.cos(Lambda)*np.sin(Lambda)
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(2*y2/b2,fz); axes[0].set_xlabel('2y/b'); axes[0].set_ylabel('fz (lb/ft)'); axes[0].set_title('HW4 – Spanwise Normal Load'); axes[0].grid(True,linestyle='--',alpha=0.5)
axes[1].plot(2*y2/b2,mx,2*y2/b2,my); axes[1].set_xlabel('2y/b'); axes[1].set_ylabel('m (ft-lb/ft)'); axes[1].legend(['mx','my']); axes[1].set_title('Distributed Moments'); axes[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw04_plots.png',dpi=150); print("Saved hw04_plots.png")
