# Wing Forced Response via Mode Superposition
import numpy as np
import matplotlib.pyplot as plt

def cantilever_modes(nmode, L, EIyy, rhoA, x):
    mu_roots=[]; gamma_v=[]
    for n in range(nmode):
        x0=(n+0.5)*np.pi
        # bisect for cos(mu)+1/cosh(mu)=0
        a_b,b_b=x0-np.pi/2+1e-6,x0+np.pi/2-1e-6
        for _ in range(100):
            m_b=(a_b+b_b)/2; fm=np.cos(m_b)+1/np.cosh(m_b)
            if abs(fm)<1e-12 or b_b-a_b<1e-12: break
            if (np.cos(a_b)+1/np.cosh(a_b))*fm<0: b_b=m_b
            else: a_b=m_b
        mu_n=m_b; mu_roots.append(mu_n)
        gn=(np.cos(mu_n)+np.cosh(mu_n))/(np.sin(mu_n)+np.sinh(mu_n)); gamma_v.append(gn)
    Phi=np.zeros((nmode,len(x))); Phi2p=np.zeros((nmode,len(x)))
    for n in range(nmode):
        mu_n=mu_roots[n]; gn=gamma_v[n]; xi=mu_n*x/L
        Phi[n]=np.cosh(xi)-np.cos(xi)-gn*(np.sinh(xi)-np.sin(xi))
        Phi2p[n]=(mu_n/L)**2*(np.cosh(xi)+np.cos(xi)-gn*(np.sinh(xi)+np.sin(xi)))
    omegas=np.array([mu_roots[n]**2/L**2*np.sqrt(EIyy/rhoA) for n in range(nmode)])
    return Phi, Phi2p, omegas, mu_roots

rhoA=9.3243; EIyy=274.751e6; L=50; nx=1001
x=np.linspace(0,L,nx); nmode=5
Phi,Phi2p,omega_n,mu_v=cantilever_modes(nmode,L,EIyy,rhoA,x)
f0=1000; fz=f0*np.cos(np.pi*x/(2*L)); Fz=5000; dt=0.01
t=np.arange(0,4+dt,dt); nt=len(t); omega0=10
w=np.zeros((nx,nt))
for n in range(nmode):
    mn=np.trapz(rhoA*Phi[n]**2,x); kn=np.trapz(EIyy*Phi2p[n]**2,x)
    om=omega_n[n]; P=np.trapz(Phi[n]*fz,x); R=Phi[n,-1]*Fz
    # Steady-state + transient for harmonic load + step load
    q_t=(P/(mn*(om**2-omega0**2))*(np.sin(omega0*t)-omega0/om*np.sin(om*t))
         +R/(mn*om**2)*(1-np.cos(om*t)))
    w+=np.outer(Phi[n],q_t)
print("HW8: Beam response computed. Max deflection = %.4f ft" % np.max(np.abs(w)))
fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111,projection='3d')
xs=x[::10]; ts=t[::5]; WW=w[::10,::5]
X,T_g=np.meshgrid(xs,ts)
ax.plot_surface(X,T_g,WW.T,cmap='hsv',alpha=0.8)
ax.set_xlabel('x (ft)'); ax.set_ylabel('t (sec)'); ax.set_zlabel('w (ft)')
ax.set_title('HW8 – Beam Dynamic Response'); ax.view_init(30,60)
plt.tight_layout(); plt.savefig('hw08_plots.png',dpi=150); print("Saved hw08_plots.png")
