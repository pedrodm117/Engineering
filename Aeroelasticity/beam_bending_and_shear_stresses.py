"Cell airfoil, Bending & Shear Stress Distribution"
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
ts=0.12/12; tw=0.48/12; n=len(x)
t=np.full(n,ts); t[37:78]=ts+tw
# Part (a)
dAu=0.5*(t[1:]+t[:-1])*np.sqrt((xu[1:]-xu[:-1])**2+(zu[1:]-zu[:-1])**2)
dAl=0.5*(t[1:]+t[:-1])*np.sqrt((xl[1:]-xl[:-1])**2+(zl[1:]-zl[:-1])**2)
A_tot=(np.sum(dAu)+np.sum(dAl)+tw*(zu[37]-zl[37])+tw*(zu[77]-zl[77]))
zA=(np.sum(0.5*(zu[1:]+zu[:-1])*dAu)+np.sum(0.5*(zl[1:]+zl[:-1])*dAl)+
    0.5*tw*(zu[37]**2-zl[37]**2)+0.5*tw*(zu[77]**2-zl[77]**2))
zbar=zA/A_tot
Iyy0=(np.sum((0.5*(zu[1:]+zu[:-1]))**2*dAu)+np.sum((0.5*(zl[1:]+zl[:-1]))**2*dAl)+
      (1/3)*tw*(zu[37]**3-zl[37]**3)+(1/3)*tw*(zu[77]**3-zl[77]**3))
Iyy=Iyy0-A_tot*zbar**2
print("(a) zbar = %.5f ft,  Iyy = %.5e ft^4" % (zbar, Iyy))
# Part (b)
M_bending=200000
sigmaxxu=-M_bending*(zu-zbar)/(Iyy*144)
sigmaxxl=-M_bending*(zl-zbar)/(Iyy*144)
# Part (c)
V_shear=-30000
idx_s=slice(38,78); idx_b=slice(37,77)
dA_web=0.5*(t[38:78]+t[37:77])*np.sqrt((xu[38:78]-xu[37:77])**2+(zu[38:78]-zu[37:77])**2)
zdA_web=(0.5*(zu[38:78]+zu[37:77])-zbar)*dA_web
Q_web=np.sum(zdA_web)+0.5*tw*(zu[37]-zbar)**2+0.5*tw*(zu[77]-zbar)**2
tauxz=V_shear*Q_web/(Iyy*2*tw*144)
print("(c) Q = %.5e ft^3,  tau_xz = %.2f psi" % (Q_web, tauxz))
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(xu,zu,'k',xl,zl,'k',[0,c],[zbar,zbar],'--k'); axes[0].set_aspect('equal'); axes[0].set_xlabel('x (ft)'); axes[0].set_ylabel('z (ft)'); axes[0].set_title('HW7 (a) – Section with centroid'); axes[0].grid(True,linestyle='--',alpha=0.5)
axes[1].plot(x,sigmaxxu,x,sigmaxxl); axes[1].legend(['upper','lower']); axes[1].set_xlabel('x (ft)'); axes[1].set_ylabel('sigma_xx (psi)'); axes[1].set_title('HW7 (b) – Bending Stress'); axes[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw07_plots.png',dpi=150); print("Saved hw07_plots.png")
