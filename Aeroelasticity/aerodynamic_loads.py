"Swept Tapered Wing for Lifting-line Loads and Distributed Force/Moment(s)" 
import numpy as np
import matplotlib.pyplot as plt

def atmosphere(h_ft):
    psl=2116.2; Tsl=518.69; a1=-3.56616e-3/Tsl; a2=0.54864e-3/Tsl
    h1=36089.0; h2=65617.0; g=32.174; R=1716.56
    b1=-g/(R*Tsl*a1); b2=-g/(R*Tsl*a2)
    hi=float(h_ft)
    if hi<=h1: theta=1+a1*hi; delta=theta**b1
    elif hi<=h2: theta=1+a1*h1; delta=theta**b1*np.exp(g*(h1-hi)/(R*Tsl*(1+a1*h1)))
    else: theta=1+a1*h1+a2*(hi-h2); delta=(1+a1*h1)**(b1-b2)*np.exp(g*(h1-h2)/(R*Tsl*(1+a1*h1)))*theta**b2
    p=psl*delta; T=Tsl*theta; rho=p/(1716.0*T); a_s=np.sqrt(1.4*1716.0*T)
    return p,T,rho,a_s

W=150000; b=120; cr=28.6; ct=6.6; M=0.3; h_alt=2000
p,T,rho,a_s=atmosphere(h_alt)
V=M*a_s; q=0.5*rho*V**2; S=(cr+ct)/2*b; AR=b**2/S
phi=np.linspace(0.01,np.pi-0.01,101)
c=(ct-cr)*np.abs(np.cos(phi))+cr
g_grav=32.174; km=0.4/g_grav; mw=km*c**2
Lambda=25*np.pi/180; cL=np.cos(Lambda); sL=np.sin(Lambda)
coeffs=np.polyfit([0,b/4,b/2],[3,1,-4]*np.array([np.pi/180]*3),2)
y_fit=-b/2*np.cos(phi)
xi=coeffs[0]*y_fit**2+coeffs[1]*np.abs(y_fit)+coeffs[2]
alpha0=-4*np.pi/180; cmac=-0.1; xcg=0.45*c; xe=0.4*c; xac=0.25*c
e=xe-xac; ecg=xcg-xe; We=8000; ep=12; epsilonp=5
Lambdaa=np.arctan((60*np.tan(Lambda)+0.1*ct-0.1*cr)/60)
clalphas=2*np.pi*np.cos(Lambdaa)/np.sqrt(1-M**2*np.cos(Lambdaa)**2)
alpha=12.4980*np.pi/180
cls=clalphas*(alpha+xi-alpha0)
N=9; C_mat=np.zeros((N,N)); B_vec=np.zeros(N)
for m in range(N):
    for n in range(N):
        f=0.5*c*clalphas*(n+1)*np.sin((n+1)*phi)*np.sin((m+1)*phi)/np.sin(phi)
        anm=np.trapz(f[1:-1],phi[1:-1])
        C_mat[m,n]=np.pi*b+anm if n==m else anm
    B_vec[m]=np.trapz(0.5*c*cls*np.sin((m+1)*phi),phi)
A_c=np.linalg.solve(C_mat,B_vec)
phi2=np.linspace(0,np.pi,101); y2=-b/2*np.cos(phi2)
l=np.zeros(len(phi2)); kappa=np.ones(len(phi2)); alpha2=np.zeros(len(phi2))
kappatemp=1; einv=1
for n in range(N):
    l+=rho*V**2*2*b*A_c[n]*np.sin((n+1)*phi2)
    alpha2+=A_c[n]*(n+1)*np.sin((n+1)*phi2)/np.sin(np.clip(phi2,1e-10,np.pi-1e-10))
    if n>0:
        kappa+=A_c[n]*(n+1)*np.sin((n+1)*phi2)/(A_c[0]*np.sin(np.clip(phi2,1e-10,np.pi-1e-10)))
        kappatemp+=A_c[n]*(n+1)**2/A_c[0]; einv+=((n+1)*(A_c[n]/A_c[0])**2)
CL=np.pi*AR*A_c[0]; L_tot=q*S*CL; n_load=L_tot/W
kappa[0]=kappatemp; kappa[-1]=kappatemp
alphai=CL*kappa/(np.pi*AR); d=2*l*alphai
espan=1/einv; CD=2*CL**2/(np.pi*AR*espan); Te=q*S*CD/2
fxa=d*np.cos(alpha)-l*np.sin(alpha); fza=l*np.cos(alpha)+d*np.sin(alpha)
mac=q*(c**2)*cmac
c2=np.abs((ct-cr)*np.cos(phi2)+cr)
e2=0.4*c2-0.25*c2; ecg2=0.45*c2-0.4*c2; mw2=km*c2**2
fx=fxa*sL*cL; fy=-fxa*cL**2; fz=fza-n_load*mw2*g_grav
mx=(mac+fza*e2-n_load*mw2*g_grav*ecg2)*cL**2
my=(mac+fza*e2-n_load*mw2*g_grav*ecg2)*sL*cL
mz=np.zeros(len(y2))
print("CL = %.4f,  Load factor n = %.4f" % (CL, n_load))
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].plot(2*y2/b,l,2*y2/b,d); axes[0].set_xlabel('2y/b'); axes[0].set_ylabel('l, d (lb/ft)'); axes[0].legend(['l','d']); axes[0].set_title('HW3 – Lift and Drag'); axes[0].grid(True,linestyle='--',alpha=0.5)
axes[1].plot(2*y2/b,fx,2*y2/b,fy,2*y2/b,fz); axes[1].set_xlabel('2y/b'); axes[1].set_ylabel('f (lb/ft)'); axes[1].legend(['fx','fy','fz']); axes[1].set_title('Distributed Forces'); axes[1].grid(True,linestyle='--',alpha=0.5)
axes[2].plot(2*y2/b,mx,2*y2/b,my,2*y2/b,mz); axes[2].set_xlabel('2y/b'); axes[2].set_ylabel('m (ft-lb/ft)'); axes[2].legend(['mx','my','mz']); axes[2].set_title('Distributed Moments'); axes[2].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw03_plots.png',dpi=150); print("Saved hw03_plots.png")
