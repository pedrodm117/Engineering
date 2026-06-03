#Variable Stiffness Wing Static deflection (Galerkin), Wing & Lumped Mass Natural Frequencies
import numpy as np
import matplotlib.pyplot as plt

def cantilever_modes(nmode, L, x):
    mu_v=[]; gam_v=[]
    for n in range(nmode):
        x0=(n+0.5)*np.pi; a_b,b_b=x0-np.pi/2+1e-6,x0+np.pi/2-1e-6
        for _ in range(100):
            m_b=(a_b+b_b)/2; fm=np.cos(m_b)+1/np.cosh(m_b)
            if abs(fm)<1e-12 or b_b-a_b<1e-12: break
            if (np.cos(a_b)+1/np.cosh(a_b))*fm<0: b_b=m_b
            else: a_b=m_b
        mu_v.append(m_b); gn=(np.cos(m_b)+np.cosh(m_b))/(np.sin(m_b)+np.sinh(m_b)); gam_v.append(gn)
    Phi=np.zeros((len(x),nmode)); Phi2p=np.zeros((len(x),nmode))
    for n in range(nmode):
        xi=mu_v[n]*x/L
        Phi[:,n]=np.cosh(xi)-np.cos(xi)-gam_v[n]*(np.sinh(xi)-np.sin(xi))
        Phi2p[:,n]=mu_v[n]**2/L**2*(np.cosh(xi)+np.cos(xi)-gam_v[n]*(np.sinh(xi)+np.sin(xi)))
    return Phi,Phi2p,np.array(mu_v)

# 2
L2=20; x2=np.linspace(0,L2,1001)
EIyy2=2.5459e6*(1-0.8*x2**2/L2**2); fz2=250*(np.cos(np.pi*x2/L2)+1)
nmode2=5; Phi2,Phi2p2,mu2=cantilever_modes(nmode2,L2,x2)
K2=np.zeros((nmode2,nmode2)); Q2=np.zeros(nmode2)
for i in range(nmode2):
    for j in range(nmode2): K2[i,j]=np.trapz(Phi2p2[:,i]*EIyy2*Phi2p2[:,j],x2)
    Q2[i]=np.trapz(Phi2[:,i]*fz2,x2)
q2=np.linalg.solve(K2,Q2); w2=Phi2@q2
print("P2: Max deflection = %.5f ft at x = %.2f ft" % (np.max(np.abs(w2)), x2[np.argmax(np.abs(w2))]))

# 3
L3=62; x3=np.linspace(0,L3,1001); EIyy3=2.5234e8; rhoA3=5.2960; m_lump=357.43; xc=21.5
nmode3=2; Phi3,Phi2p3,mu3=cantilever_modes(nmode3,L3,x3)
# Interpolate Phi at xc
Phic=np.array([np.interp(xc,x3,Phi3[:,n]) for n in range(nmode3)])
M3=np.zeros((nmode3,nmode3)); K3=np.zeros((nmode3,nmode3))
for i in range(nmode3):
    for j in range(nmode3):
        M3[i,j]=np.trapz(Phi3[:,i]*rhoA3*Phi3[:,j],x3)+Phic[i]*m_lump*Phic[j]
        K3[i,j]=np.trapz(Phi2p3[:,i]*EIyy3*Phi2p3[:,j],x3)
eig3=np.linalg.eigvals(np.linalg.solve(M3,K3)); omega3=np.sort(np.sqrt(np.abs(eig3)))
print("P3: Natural frequencies (with lumped mass) = %s rad/s" % str(np.round(omega3,4)))
# Without lumped mass
M3b=np.zeros((nmode3,nmode3))
for i in range(nmode3):
    for j in range(nmode3): M3b[i,j]=np.trapz(Phi3[:,i]*rhoA3*Phi3[:,j],x3)
eig3b=np.linalg.eigvals(np.linalg.solve(M3b,K3)); omega3b=np.sort(np.sqrt(np.abs(eig3b)))
print("P3: Natural frequencies (no lumped mass)   = %s rad/s" % str(np.round(omega3b,4)))

# 4
L4=66
Iyy_pts=np.array([1.88329,0.99233,0.43889,0.14262,0.02314])
A_pts=np.array([1.56047,1.26038,0.96029,0.66020,0.36011])
xn=np.array([0,0.25,0.5,0.75,1.0]); E4=1.5264e9; rho4=5.4245
EIcoeff=np.polyfit(xn,E4*Iyy_pts,4); rhoAcoeff=np.polyfit(xn,rho4*A_pts,1)
x4=np.linspace(0,L4,1001)
EIyy4=np.polyval(EIcoeff,x4/L4); rhoA4=np.polyval(rhoAcoeff,x4/L4)
fz4=2000*np.sqrt(1-x4**2/L4**2)
nmode4=2; Phi4,Phi2p4,mu4=cantilever_modes(nmode4,L4,x4)
M4=np.zeros((nmode4,nmode4)); K4=np.zeros((nmode4,nmode4)); Q4=np.zeros(nmode4)
Phi2mat=np.zeros((nmode4,nmode4)); qdot04=np.zeros(nmode4)
for i in range(nmode4):
    for j in range(nmode4):
        M4[i,j]=np.trapz(Phi4[:,i]*rhoA4*Phi4[:,j],x4)
        K4[i,j]=np.trapz(Phi2p4[:,i]*EIyy4*Phi2p4[:,j],x4)
        Phi2mat[i,j]=np.trapz(Phi4[:,i]*Phi4[:,j],x4)
    Q4[i]=np.trapz(Phi4[:,i]*fz4,x4)
    qdot04[i]=np.trapz(Phi4[:,i]*10*x4/L4,x4)
qdot04=np.linalg.solve(Phi2mat,qdot04)
eigv4=np.linalg.eigvals(np.linalg.solve(M4,K4)); omega4=np.sort(np.sqrt(np.abs(eigv4)))
print("P4: Natural frequencies = %s rad/s" % str(np.round(omega4,4)))
n2=nmode4; A4=np.block([[np.zeros((n2,n2)),np.eye(n2)],[-np.linalg.solve(M4,K4),np.zeros((n2,n2))]])
v4=np.concatenate([np.zeros(n2),np.linalg.solve(M4,Q4)])
dt4=0.001; t4=np.arange(0,1+dt4,dt4); nt4=len(t4)
y4=np.zeros((2*n2,nt4)); y4[:,0]=np.concatenate([np.zeros(n2),qdot04])
I2=np.eye(2*n2); Ainv4=np.linalg.solve(I2-dt4*A4/2,I2); Aip4=I2+dt4*A4/2
for i in range(nt4-1): y4[:,i+1]=Ainv4@(Aip4@y4[:,i]+dt4*v4)
q_dyn=y4[:n2,:]; w4=Phi4@q_dyn
print("P4: Max dynamic deflection = %.5f ft" % np.max(np.abs(w4)))

fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(x2,w2); axes[0].set_xlabel('x (ft)'); axes[0].set_ylabel('w (ft)'); axes[0].set_title('HW9 P2 – Beam Deflection'); axes[0].grid(True,linestyle='--',alpha=0.5)
im=axes[1].imshow(w4,aspect='auto',origin='lower',extent=[0,1,0,L4],cmap='jet')
axes[1].set_xlabel('t (sec)'); axes[1].set_ylabel('x (ft)'); axes[1].set_title('HW9 P4 – Beam Response w(x,t)')
plt.colorbar(im,ax=axes[1],label='w (ft)')
plt.tight_layout(); plt.savefig('hw09_plots.png',dpi=150); print("Saved hw09_plots.png")
