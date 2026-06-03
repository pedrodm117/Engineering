# Aerodynamic Fluttering & Gust Response
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
import mpmath

#Atmosphere
def atmosphere(h_ft):
    psl=2116.2; Tsl=518.69; a1=-3.56616e-3/Tsl; a2=0.54864e-3/Tsl
    h1=36089.0; h2=65617.0; g=32.174; R=1716.56; b1=-g/(R*Tsl*a1); b2=-g/(R*Tsl*a2)
    hi=float(h_ft)
    if hi<=h1: theta=1+a1*hi; delta=theta**b1
    elif hi<=h2: theta=1+a1*h1; delta=theta**b1*np.exp(g*(h1-hi)/(R*Tsl*(1+a1*h1)))
    else: theta=1+a1*h1+a2*(hi-h2); delta=(1+a1*h1)**(b1-b2)*np.exp(g*(h1-h2)/(R*Tsl*(1+a1*h1)))*theta**b2
    p=psl*delta; T=Tsl*theta; rho=p/(1716.0*T); a_s=float(np.sqrt(1.4*1716.0*T))
    return float(p),float(T),float(rho),a_s

def theodorsen(k_arr):
    k_arr=np.atleast_1d(k_arr).astype(float)
    C=np.zeros(len(k_arr),dtype=complex)
    for i,kv in enumerate(k_arr):
        if kv<=0: C[i]=1+0j
        else:
            H1=complex(mpmath.hankel2(1,kv)); H0=complex(mpmath.hankel2(0,kv))
            C[i]=H1/(H1+1j*H0)
    return C,C.real,C.imag

def cantilever_modes(nmode,L,x):
    mu_v=[]; gam_v=[]
    for n in range(nmode):
        x0=(n+0.5)*np.pi; a_b,b_b=x0-np.pi/2+1e-6,x0+np.pi/2-1e-6
        for _ in range(100):
            m_b=(a_b+b_b)/2; fm=np.cos(m_b)+1/np.cosh(m_b)
            if abs(fm)<1e-12 or b_b-a_b<1e-12: break
            if (np.cos(a_b)+1/np.cosh(a_b))*fm<0: b_b=m_b
            else: a_b=m_b
        mu_v.append(m_b); gn=(np.cos(m_b)+np.cosh(m_b))/(np.sin(m_b)+np.sinh(m_b)); gam_v.append(gn)
    Phi=np.zeros((len(x),nmode)); Phi1p=np.zeros((len(x),nmode)); Phi2p=np.zeros((len(x),nmode))
    for n in range(nmode):
        xi=mu_v[n]*x/L
        Phi[:,n]=np.cosh(xi)-np.cos(xi)-gam_v[n]*(np.sinh(xi)-np.sin(xi))
        Phi1p[:,n]=mu_v[n]/L*(np.sinh(xi)+np.sin(xi)-gam_v[n]*(np.cosh(xi)-np.cos(xi)))
        Phi2p[:,n]=mu_v[n]**2/L**2*(np.cosh(xi)+np.cos(xi)-gam_v[n]*(np.sinh(xi)+np.sin(xi)))
    return Phi,Phi1p,Phi2p,np.array(mu_v),np.array(gam_v)

def interp1(xp,fp,xi): return np.interp(xi,xp,fp)

print("="*60); print("PROBLEM 1 – 2-DOF flutter (plunge + pitch), C=1")
# **************************
_,_,rhoinf,ainf=atmosphere(0)
c1=10; ma=rhoinf*np.pi*c1**2/4; m1=10*ma; I1=m1*c1**2/12
ecg=0.1*c1; e1=0.15*c1; gh=0.01; gth=0.01
Kh=500*(1+1j*gh); Kth=10000*(1+1j*gth)
Mm=np.array([[m1,m1*ecg],[m1*ecg,I1]]); Km=np.array([[Kh,0],[0,Kth]])
A0=np.block([[np.zeros((2,2)),np.eye(2)],[-nla.solve(Mm,Km),np.zeros((2,2))]])
lam0=np.linalg.eigvals(A0); pos=lam0[np.imag(lam0)>0]; omega0=np.sort(np.abs(np.imag(pos)))
Vinf=np.arange(0.001,400,0.5); qinf=0.5*rhoinf*Vinf**2
clalpha_v=2*np.pi/np.sqrt(1-np.clip(Vinf/ainf,0,0.99)**2)
n_v=len(Vinf); omega_p=np.zeros((2,n_v)); g_p=np.zeros((2,n_v))
omega_p[:,0]=omega0
for i in range(n_v):
    Ka=np.array([[0,qinf[i]*c1*clalpha_v[i]],[0,-qinf[i]*c1*e1*clalpha_v[i]]])
    A=np.block([[np.zeros((2,2)),np.eye(2)],[-nla.solve(Mm,Km+Ka),np.zeros((2,2))]])
    lam=np.linalg.eigvals(A); lam=lam[np.argsort(np.imag(lam))]
    pos=lam[np.imag(lam)>0]
    if len(pos)>=2:
        omega_p[:,i]=np.sort(np.abs(np.imag(pos)))[:2]
        g_p[:,i]=2*np.real(pos[np.argsort(np.abs(np.imag(pos)))][:2])/omega_p[:,i]

VF_list=[]; omF_list=[]
for j in range(2):
    g_row=g_p[j]; om_row=omega_p[j]
    sign_changes=np.where(np.diff(np.sign(g_row)))[0]
    if len(sign_changes)>0:
        i0=sign_changes[0]
        VF_j=np.interp(0,g_row[i0:i0+2],Vinf[i0:i0+2])
        omF_j=np.interp(0,g_row[i0:i0+2],om_row[i0:i0+2])
        VF_list.append(VF_j); omF_list.append(omF_j)
if VF_list:
    VF=min(VF_list); MF=VF/ainf
    print("Flutter speed VF = %.2f ft/s,  MF = %.4f" % (VF, MF))

fig1,axes=plt.subplots(1,2,figsize=(11,4))
for j in range(2): axes[0].plot(Vinf,omega_p[j])
if VF_list:
    for k,v in enumerate(VF_list): axes[0].plot(v,omF_list[k],'ro',markersize=8)
axes[0].set_xlabel('V (ft/s)'); axes[0].set_ylabel('omega (rad/s)'); axes[0].set_title('P1 V-omega'); axes[0].legend(['Plunge','Pitch','Flutter']); axes[0].grid(True,linestyle='--',alpha=0.5)
for j in range(2): axes[1].plot(Vinf,g_p[j])
axes[1].axhline(0,color='k',linewidth=0.5); axes[1].set_ylim([-1,1])
if VF_list:
    axes[1].plot(min(VF_list),0,'ro',markersize=8)
axes[1].set_xlabel('V (ft/s)'); axes[1].set_ylabel('g'); axes[1].set_title('P1 V-g'); axes[1].legend(['Plunge','Pitch','Flutter']); axes[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw12_p1.png',dpi=150); print("Saved hw12_p1.png")

# -------------------------------
print("\n"+"="*60); print("PROBLEM 2 – 1-DOF torsional flutter with Theodorsen")
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
nu=0.001; c2=1; e2_p=-0.25*c2; em2=(0.25-0)*(c2); ec2=(0.5-0)*c2
AR2=8; Vinf2=np.arange(0.1,300.1,0.1)
_,_,rho2,ainf2=atmosphere(0)
omega2=np.zeros(len(Vinf2)); omega2[0]=10; g2=np.zeros(len(Vinf2))
for i,V in enumerate(Vinf2):
    om=omega2[i] if i>0 else 10
    for _ in range(4):
        k=om*c2/(2*V); C_k=theodorsen(np.array([k]))[0][0]
        Minf2=V/ainf2; cla2=2*np.pi/(np.sqrt(1-Minf2**2)+2/AR2)
        coeff=[1+nu*(em2**2+1/32), nu*V/c2*ec2/c2*(1-2/np.pi*0.0*cla2),
               10**2-2*nu/np.pi*V**2/c2**2*0.0*cla2]
        # simplified: use strip theory for torsion only
        coeff_c=[1+nu*(em2**2+1/32),
                 nu*V/c2*ec2/c2*(1-2/np.pi*(0.0)*C_k*cla2),
                 10**2-2*nu/np.pi*V**2/c2**2*(0.0)*C_k*cla2]
        p_roots=np.roots(coeff_c)
        for p_r in p_roots:
            if np.imag(p_r)>0: om=np.imag(p_r); g2[i]=2*np.real(p_r)/om
    omega2[i]=om
sc=np.where(np.diff(np.sign(g2)))[0]
if len(sc)>0:
    VF2=np.interp(0,g2[sc[0]:sc[0]+2],Vinf2[sc[0]:sc[0]+2])
    omF2=np.interp(0,g2[sc[0]:sc[0]+2],omega2[sc[0]:sc[0]+2])
    print("P2: Flutter speed = %.2f ft/s" % VF2)
else:
    VF2=None; print("P2: No flutter detected in range")
fig2,axes2=plt.subplots(1,2,figsize=(11,4))
axes2[0].plot(Vinf2,omega2); axes2[0].set_xlabel('V (ft/s)'); axes2[0].set_ylabel('omega'); axes2[0].set_title('P2 V-omega (torsion)'); axes2[0].grid(True,linestyle='--',alpha=0.5)
axes2[1].plot(Vinf2,g2); axes2[1].axhline(0,color='k',linewidth=0.5); axes2[1].set_xlabel('V (ft/s)'); axes2[1].set_ylabel('g'); axes2[1].set_title('P2 V-g (torsion)'); axes2[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw12_p2.png',dpi=150); print("Saved hw12_p2.png")

# ****************************************************
print("\n"+"="*60); print("PROBLEM 3 – 3D wing flutter (unswept, C=1)")
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def flutter_3d(EIyy,GJ,rhoA,rhoIxx,c3,e3,em3,ec3,ecg3,Lambda_deg,L3,AR3,
               altitude,nmode3,Vinf_range,use_theodorsen=False,zeta=0.005):
    _,_,rhoinf3,ainf3=atmosphere(altitude)
    ma3=rhoinf3*np.pi*c3**2/4
    cL=np.cos(np.radians(Lambda_deg)); sL=np.sin(np.radians(Lambda_deg)); tL=np.tan(np.radians(Lambda_deg))
    x3=np.linspace(0,L3,501)
    Phi3,Phi1p3,Phi2p3,mu3v,_=cantilever_modes(nmode3,L3,x3)
    Psi3=np.zeros((len(x3),nmode3)); Psi1p3=np.zeros((len(x3),nmode3))
    for n in range(nmode3):
        Psi3[:,n]=np.sin((n+1)*np.pi*x3/L3)
        Psi1p3[:,n]=(n+1)*np.pi/L3*np.cos((n+1)*np.pi*x3/L3)
    M11=np.zeros((nmode3,nmode3)); M12=np.zeros((nmode3,nmode3))
    M21=np.zeros((nmode3,nmode3)); M22=np.zeros((nmode3,nmode3))
    K11=np.zeros((nmode3,nmode3)); K22=np.zeros((nmode3,nmode3))
    for m in range(nmode3):
        for n in range(nmode3):
            M11[m,n]=np.trapz(Phi3[:,m]*rhoA*Phi3[:,n],x3)
            M12[m,n]=np.trapz(Phi3[:,m]*rhoA*ecg3*Psi3[:,n],x3)
            M21[m,n]=M12[n,m]
            M22[m,n]=np.trapz(Psi3[:,m]*rhoIxx*Psi3[:,n],x3)
            K11[m,n]=np.trapz(Phi2p3[:,m]*EIyy*Phi2p3[:,n],x3)
            K22[m,n]=np.trapz(Psi1p3[:,m]*GJ*Psi1p3[:,n],x3)
    Mm3=np.block([[M11,M12],[M21,M22]]); Km3=np.block([[K11,np.zeros((nmode3,nmode3))],[np.zeros((nmode3,nmode3)),K22]])
    eigV,eigvec=np.linalg.eig(nla.solve(Mm3,Km3))
    eigV=np.sort(np.abs(eigV.real)); omega0v=np.sqrt(eigV)
    Cs=2*Mm3@eigvec@np.diag(zeta*np.sqrt(eigV))@nla.inv(eigvec)
    N2=2*nmode3; omega_v=np.zeros((N2,len(Vinf_range))); g_v=np.zeros((N2,len(Vinf_range)))
    omega_v[:,0]=omega0v
    for i,V in enumerate(Vinf_range):
        Minf3=V/ainf3; cla3=2*np.pi*cL/(np.sqrt(1-Minf3**2*cL**2)+2*cL/AR3)
        C_t=complex(1,0)  # C=1 for P3
        lth=2*ma3*V**2*C_t*cla3*cL**2/(np.pi*c3)
        ltht=ma3*V*cL**2*(2*ec3/c3*C_t*cla3/np.pi+1)
        lthtt=ma3*em3*cL**2
        lwx=-2*ma3*V**2*C_t*cla3*sL*cL/(np.pi*c3)
        lwt=-2*ma3*V*C_t*cla3*cL/(np.pi*c3)
        lwtt=-ma3*cL
        mth=2*ma3*V**2*e3/c3*C_t*cla3*cL**3/np.pi
        mtht=ma3*V*ec3*cL**3*(2*e3/c3*C_t*cla3/np.pi-1)
        mthtt=-ma3*c3**2*cL**3*(em3**2/c3**2+1/32)
        mwx=-2*ma3*V**2*e3/c3*C_t*cla3*sL*cL**2/np.pi
        mwt=-2*ma3*V*e3/c3*C_t*cla3*cL**2/np.pi
        mwtt=ma3*em3*cL**2
        Ma11=np.zeros((nmode3,nmode3)); Ma12=np.zeros((nmode3,nmode3))
        Ma21=np.zeros((nmode3,nmode3)); Ma22=np.zeros((nmode3,nmode3))
        Ca11=np.zeros((nmode3,nmode3)); Ca12=np.zeros((nmode3,nmode3))
        Ca21=np.zeros((nmode3,nmode3)); Ca22=np.zeros((nmode3,nmode3))
        Ka11=np.zeros((nmode3,nmode3)); Ka12=np.zeros((nmode3,nmode3))
        Ka21=np.zeros((nmode3,nmode3)); Ka22=np.zeros((nmode3,nmode3))
        for m in range(nmode3):
            for n in range(nmode3):
                Ma11[m,n]=np.trapz((-Phi3[:,m]*lwtt+Phi1p3[:,m]*mwtt*tL)*Phi3[:,n],x3)
                Ma12[m,n]=np.trapz((-Phi3[:,m]*lthtt+Phi1p3[:,m]*mthtt*tL)*Psi3[:,n],x3)
                Ma21[m,n]=np.trapz(-Psi3[:,m]*mwtt*Phi3[:,n],x3)
                Ma22[m,n]=np.trapz(-Psi3[:,m]*mthtt*Psi3[:,n],x3)
                Ca11[m,n]=np.trapz((-Phi3[:,m]*lwt+Phi1p3[:,m]*mwt*tL)*Phi3[:,n],x3)
                Ca12[m,n]=np.trapz((-Phi3[:,m]*ltht+Phi1p3[:,m]*mtht*tL)*Psi3[:,n],x3)
                Ca21[m,n]=np.trapz(-Psi3[:,m]*mwt*Phi3[:,n],x3)
                Ca22[m,n]=np.trapz(-Psi3[:,m]*mtht*Psi3[:,n],x3)
                Ka11[m,n]=np.trapz((-Phi3[:,m]*lwx+Phi1p3[:,m]*mwx*tL)*Phi1p3[:,n],x3)
                Ka12[m,n]=np.trapz((-Phi3[:,m]*lth+Phi1p3[:,m]*mth*tL)*Psi3[:,n],x3)
                Ka21[m,n]=np.trapz(-Psi3[:,m]*mwx*Phi1p3[:,n],x3)
                Ka22[m,n]=np.trapz(-Psi3[:,m]*mth*Psi3[:,n],x3)
        Ma=np.block([[Ma11,Ma12],[Ma21,Ma22]])
        Ca=np.block([[Ca11,Ca12],[Ca21,Ca22]])
        Ka=np.block([[Ka11,Ka12],[Ka21,Ka22]])
        MtKt=Mm3+Ma; MtCt=Cs+Ca; MtKa=Km3+Ka
        A3d=np.block([[np.zeros((N2,N2)),np.eye(N2)],[-nla.solve(MtKt,MtKa),-nla.solve(MtKt,MtCt)]])
        lam3=np.linalg.eigvals(A3d); lam3=lam3[np.argsort(np.imag(lam3))]
        pos3=lam3[np.imag(lam3)>0]
        for j in range(min(N2,len(pos3))):
            omega_v[j,i]=np.abs(np.imag(pos3[j])); g_v[j,i]=2*np.real(pos3[j])/omega_v[j,i]
    # Find flutter
    VF_v=[]; omF_v=[]
    for j in range(N2):
        sc=np.where(np.diff(np.sign(g_v[j])))[0]
        if len(sc)>0:
            i0=sc[0]; VFj=np.interp(0,g_v[j,i0:i0+2],Vinf_range[i0:i0+2]); omFj=np.interp(0,g_v[j,i0:i0+2],omega_v[j,i0:i0+2])
            VF_v.append(VFj); omF_v.append(omFj)
        else: VF_v.append(1e9); omF_v.append(0)
    idx_f=int(np.argmin(VF_v))
    VF_out=VF_v[idx_f] if VF_v[idx_f]<1e8 else None
    return omega_v,g_v,VF_out,(VF_v,omF_v,idx_f),Vinf_range

# P3 parameters
E3=30e6*144; G3=11.4e6*144; L3p=3000; c3p=30; AR3p=L3p/c3p; rho3=20
h3=3; a3=c3p/2; b3=h3/2; A3=c3p*h3; Iyy3=c3p*h3**3/12
J3=a3*b3**3*(16/3-3.36*b3/a3*(1-(b3/a3)**4/16))
EIyy3=E3*Iyy3; GJ3=G3*J3; rhoA3=rho3*A3; rhoIxx3=rho3*(Iyy3+c3p**3*h3/12)
e3p=c3p/4; em3p=0; ec3p=c3p/4; ecg3p=0
print("P3: EIyy=%.3e, GJ=%.3e, rhoA=%.3f, rhoIxx=%.3f" % (EIyy3,GJ3,rhoA3,rhoIxx3))
Vinf3=np.arange(1,301,1)
omega3,g3,VF3,flutter3,_=flutter_3d(EIyy3,GJ3,rhoA3,rhoIxx3,c3p,e3p,em3p,ec3p,ecg3p,0,L3p,AR3p,0,2,Vinf3)
if VF3: print("P3 Flutter speed = %.2f ft/s,  Mach = %.4f" % (VF3, VF3/atmosphere(0)[3]))
else: print("P3: No flutter in range")
fig3,axes3=plt.subplots(1,2,figsize=(11,4))
for j in range(4): axes3[0].plot(Vinf3,omega3[j]); axes3[1].plot(Vinf3,g3[j])
axes3[0].set_xlabel('V (ft/s)'); axes3[0].set_ylabel('omega'); axes3[0].set_title('P3 V-omega'); axes3[0].grid(True,linestyle='--',alpha=0.5)
axes3[1].axhline(0,color='k',linewidth=0.5); axes3[1].set_xlabel('V (ft/s)'); axes3[1].set_ylabel('g'); axes3[1].set_title('P3 V-g'); axes3[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw12_p3.png',dpi=150); print("Saved hw12_p3.png")

# ----------------------------------------------------
print("\n"+"="*60); print("PROBLEM 4 – Swept wing flutter with Theodorsen")
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
EIyy4p=2e9; GJ4p=1e9; rhoA4p=5; rhoIxx4p=1500
c4=14; e4=0.15*c4; em4=0.25*c4-e4; ec4=0.5*c4-e4; ecg4=0; Lambda4=30
b4=140; L4p=b4/(2*np.cos(np.radians(Lambda4))); AR4=b4/c4
Vinf4=np.arange(1,801,2)
print("P4: L = %.2f ft, AR = %.2f" % (L4p, AR4))
omega4,g4,VF4,flutter4,_=flutter_3d(EIyy4p,GJ4p,rhoA4p,rhoIxx4p,c4,e4,em4,ec4,ecg4,Lambda4,L4p,AR4,0,4,Vinf4)
if VF4: print("P4 Flutter speed = %.2f ft/s,  Mach = %.4f" % (VF4, VF4/atmosphere(0)[3]))
else: print("P4: No flutter in range")
fig4,axes4=plt.subplots(1,2,figsize=(11,4))
for j in range(8): axes4[0].plot(Vinf4,omega4[j]); axes4[1].plot(Vinf4,g4[j])
axes4[0].set_xlabel('V (ft/s)'); axes4[0].set_ylabel('omega'); axes4[0].set_title('P4 V-omega (swept)'); axes4[0].grid(True,linestyle='--',alpha=0.5)
axes4[1].axhline(0,color='k',linewidth=0.5); axes4[1].set_xlabel('V (ft/s)'); axes4[1].set_ylabel('g'); axes4[1].set_title('P4 V-g (swept)'); axes4[1].grid(True,linestyle='--',alpha=0.5)
plt.tight_layout(); plt.savefig('hw12_p4.png',dpi=150); print("Saved hw12_p4.png")

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
print("\n"+"="*60); print("PROBLEM 5 – Gust response at altitude 35000 ft")
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
_,_,rhoinf5,ainf5=atmosphere(35000)
Vinf5=580; Lambda5=0; L5=b4/(2*np.cos(np.radians(Lambda5))); AR5=b4/c4
ma5=rhoinf5*np.pi*c4**2/4
Minf5=Vinf5/ainf5; cla5=2*np.pi/(np.sqrt(1-Minf5**2)+2/AR5)
print("P5: Minf = %.4f, cla = %.4f" % (Minf5, cla5))
# Run flutter analysis at V=580 ft/s to get A matrix
x5=np.linspace(0,L5,501); nmode5=4
Phi5,Phi1p5,Phi2p5,mu5v,_=cantilever_modes(nmode5,L5,x5)
Psi5=np.zeros((len(x5),nmode5)); Psi1p5=np.zeros((len(x5),nmode5))
for n in range(nmode5):
    Psi5[:,n]=np.sin((n+0.5)*np.pi*x5/L5)
    Psi1p5[:,n]=(n+0.5)*np.pi/L5*np.cos((n+0.5)*np.pi*x5/L5)
M11p=np.zeros((nmode5,nmode5)); M22p=np.zeros((nmode5,nmode5)); K11p=np.zeros((nmode5,nmode5)); K22p=np.zeros((nmode5,nmode5))
for m in range(nmode5):
    for n in range(nmode5):
        M11p[m,n]=np.trapz(Phi5[:,m]*rhoA4p*Phi5[:,n],x5)
        M22p[m,n]=np.trapz(Psi5[:,m]*rhoIxx4p*Psi5[:,n],x5)
        K11p[m,n]=np.trapz(Phi2p5[:,m]*EIyy4p*Phi2p5[:,n],x5)
        K22p[m,n]=np.trapz(Psi1p5[:,m]*GJ4p*Psi1p5[:,n],x5)
Mm5=np.block([[M11p,np.zeros((nmode5,nmode5))],[np.zeros((nmode5,nmode5)),M22p]])
Km5=np.block([[K11p,np.zeros((nmode5,nmode5))],[np.zeros((nmode5,nmode5)),K22p]])
eigV5,eigvec5=np.linalg.eig(nla.solve(Mm5,Km5)); eigV5=np.sort(np.abs(eigV5.real))
omega0_5=np.sqrt(eigV5); Cs5=2*Mm5@eigvec5@np.diag(0.005*np.sqrt(eigV5))@nla.inv(eigvec5)
# Aero at V=580
lth5=2*ma5*Vinf5**2*cla5/(np.pi*c4); ltht5=ma5*Vinf5*(2*ec4/c4*cla5/np.pi+1)
lthtt5=ma5*em4; mth5=2*ma5*Vinf5**2*e4/c4*cla5/np.pi; mtht5=ma5*Vinf5*ec4*(2*e4/c4*cla5/np.pi-1)
mthtt5=-ma5*c4**2*(em4**2/c4**2+1/32); mwt5=-2*ma5*Vinf5*cla5/(np.pi*c4); mwtt5=ma5
Ma11p=np.zeros((nmode5,nmode5)); Ma22p=np.zeros((nmode5,nmode5))
Ca11p=np.zeros((nmode5,nmode5)); Ca12p=np.zeros((nmode5,nmode5))
Ca21p=np.zeros((nmode5,nmode5)); Ca22p=np.zeros((nmode5,nmode5))
Ka12p=np.zeros((nmode5,nmode5)); Ka21p=np.zeros((nmode5,nmode5)); Ka22p=np.zeros((nmode5,nmode5))
for m in range(nmode5):
    for n in range(nmode5):
        Ma11p[m,n]=np.trapz(-Phi5[:,m]*(-ma5)*Phi5[:,n],x5)  # lwtt=-ma for Lambda=0
        Ma22p[m,n]=np.trapz(-Psi5[:,m]*mthtt5*Psi5[:,n],x5)
        Ca11p[m,n]=np.trapz(-Phi5[:,m]*(-2*ma5*Vinf5*cla5/(np.pi*c4))*Phi5[:,n],x5)
        Ca12p[m,n]=np.trapz(-Phi5[:,m]*ltht5*Psi5[:,n],x5)
        Ca21p[m,n]=np.trapz(-Psi5[:,m]*mwt5*Phi5[:,n],x5)
        Ca22p[m,n]=np.trapz(-Psi5[:,m]*mtht5*Psi5[:,n],x5)
        Ka12p[m,n]=np.trapz(-Phi5[:,m]*lth5*Psi5[:,n],x5)
        Ka21p[m,n]=np.trapz(-Psi5[:,m]*0*Phi5[:,n],x5)  # mwx=0 for Lambda=0
        Ka22p[m,n]=np.trapz(-Psi5[:,m]*mth5*Psi5[:,n],x5)
Ma5=np.block([[Ma11p,np.zeros((nmode5,nmode5))],[np.zeros((nmode5,nmode5)),Ma22p]])
Ca5=np.block([[Ca11p,Ca12p],[Ca21p,Ca22p]])
Ka5=np.block([[np.zeros((nmode5,nmode5)),Ka12p],[Ka21p,Ka22p]])
MtK5=Mm5+Ma5; N5=2*nmode5
A5=np.block([[np.zeros((N5,N5)),np.eye(N5)],[-nla.solve(MtK5,Km5+Ka5),-nla.solve(MtK5,Cs5+Ca5)]])
# Gust profile
dt5=0.001; t5=np.arange(0,10+dt5,dt5); nt5=len(t5)
w0=33.7156; H=290; xg=580; tg=2*H/Vinf5; qinf5=0.5*rhoinf5*Vinf5**2
wg=np.zeros(nt5); tstart=2; tend=tstart+tg
i0s=round(tstart/dt5); i0e=round(tend/dt5)
for i in range(i0s,i0e): wg[i]=w0/2*(1-np.cos(2*np.pi/tg*(t5[i]-xg/Vinf5)))
lg5=qinf5*c4*cla5*wg/Vinf5; mg5=lg5*e4
y5=np.zeros((4*nmode5,nt5))
I4n=np.eye(4*nmode5); Ainv5=nla.solve(I4n-dt5*A5/2,I4n); Aip5=I4n+dt5*A5/2
for i in range(nt5-1):
    Q1=np.array([np.trapz(Phi5[:,n]*lg5[i],x5) for n in range(nmode5)])
    Q2=np.array([np.trapz(Psi5[:,n]*mg5[i],x5) for n in range(nmode5)])
    Q5=np.concatenate([Q1,Q2])
    f5=np.concatenate([np.zeros(N5),nla.solve(MtK5,Q5)])
    y5[:,i+1]=Ainv5@(Aip5@y5[:,i]+dt5*f5)
W5=y5[:nmode5,:]; Theta5=y5[nmode5:N5,:]
w_xt=Phi5@W5; theta_xt=Psi5@Theta5
print("P5: Max bending = %.5f ft, Max twist = %.5f deg" % (np.max(np.abs(w_xt)), np.max(np.abs(theta_xt))*180/np.pi))
fig5,axes5=plt.subplots(1,2,figsize=(11,4))
im5a=axes5[0].imshow(w_xt[::5,::20],aspect='auto',origin='lower',extent=[0,10,0,1],cmap='jet')
axes5[0].set_xlabel('t (sec)'); axes5[0].set_ylabel('x/L'); axes5[0].set_title('P5 w(x,t) ft'); plt.colorbar(im5a,ax=axes5[0])
im5b=axes5[1].imshow(theta_xt[::5,::20]*180/np.pi,aspect='auto',origin='lower',extent=[0,10,0,1],cmap='jet')
axes5[1].set_xlabel('t (sec)'); axes5[1].set_ylabel('x/L'); axes5[1].set_title('P5 theta(x,t) deg'); plt.colorbar(im5b,ax=axes5[1])
plt.tight_layout(); plt.savefig('hw12_p5.png',dpi=150); print("Saved hw12_p5.png")
plt.show()
print("Done – all HW12 problems complete.")
