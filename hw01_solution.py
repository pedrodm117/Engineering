"""ME 184 - Homework 1 Python Solution
Problem 2: Dynamic pressure and Mach number along a constant-slope flight path.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def atmosphere(h_ft):
    psl=2116.2; Tsl=518.69; a1=-3.56616e-3/Tsl; a2=0.54864e-3/Tsl
    h1=36089.0; h2=65617.0; g=32.174; R=1716.56; b1=-g/(R*Tsl*a1); b2=-g/(R*Tsl*a2)
    h_arr=np.atleast_1d(np.asarray(h_ft,dtype=float)); theta=np.zeros_like(h_arr); delta=np.zeros_like(h_arr)
    for i,hi in enumerate(h_arr):
        if hi<=h1: theta[i]=1+a1*hi; delta[i]=theta[i]**b1
        elif hi<=h2: theta[i]=1+a1*h1; delta[i]=theta[i]**b1*np.exp(g*(h1-hi)/(R*Tsl*(1+a1*h1)))
        else: theta[i]=1+a1*h1+a2*(hi-h2); delta[i]=(1+a1*h1)**(b1-b2)*np.exp(g*(h1-h2)/(R*Tsl*(1+a1*h1)))*theta[i]**b2
    p=psl*delta; T=Tsl*theta; rho=p/(1716.0*T); a_s=np.sqrt(1.4*1716.0*T)
    return p,T,rho,a_s

Vf = 1624 * 1.466666667   # mph -> ft/s
hf = 66000.0
m  = Vf / hf               # slope ft/s per ft
h  = np.arange(0, hf+1, dtype=float)
V  = m * h
p, T, rho, a = atmosphere(h)
M  = V / a
gamma = 1.4
q  = gamma/2 * p * M**2
idx   = np.argmax(q)
qmax  = q[idx]; hqmax = h[idx]; Mqmax = M[idx]

print("Max dynamic pressure qmax = %.2f psf" % qmax)
print("  at altitude h = %.0f ft, Mach = %.4f" % (hqmax, Mqmax))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
ax1.plot(M,h,'b',Mqmax,hqmax,'or',markerfacecolor='r',markersize=8,label='M @ q_max')
ax1.set_xlabel('M'); ax1.set_ylabel('h (ft)'); ax1.legend(); ax1.grid(True,linestyle='--',alpha=0.5)
ax1.set_title('HW1 P2 – Mach vs Altitude')
ax2.plot(q,h,'b',qmax,hqmax,'or',markerfacecolor='r',markersize=8,label='q_max')
ax2.set_xlabel('q (psf)'); ax2.set_ylabel('h (ft)'); ax2.legend(); ax2.grid(True,linestyle='--',alpha=0.5)
ax2.set_title('HW1 P2 – Dynamic Pressure vs Altitude')
plt.tight_layout()
plt.savefig('hw01_plots.png',dpi=150)
print("Saved hw01_plots.png")
