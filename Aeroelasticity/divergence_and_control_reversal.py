# Divergence & Control Reversal
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt
def _brentq(f, a, b, **kw):
    fa, fb = f(a), f(b)
    for _ in range(100):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < 1e-12 or (b - a) < 1e-12:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2

def atmosphere(h_ft):
    psl = 2116.2; Tsl = 518.69
    a1 = -3.56616e-3 / Tsl
    a2 =  0.54864e-3 / Tsl
    h1 = 36089.0; h2 = 65617.0
    g = 32.174; R = 1716.56
    b1 = -g / (R * Tsl * a1)
    b2 = -g / (R * Tsl * a2)
    h = np.atleast_1d(float(h_ft))
    theta = np.zeros_like(h); delta = np.zeros_like(h)
    for i, hi in enumerate(h):
        if hi <= h1:
            theta[i] = 1 + a1 * hi
            delta[i] = theta[i] ** b1
        elif hi <= h2:
            theta[i] = 1 + a1 * h1
            delta[i] = theta[i]**b1 * np.exp(g*(h1-hi)/(R*Tsl*(1+a1*h1)))
        else:
            theta[i] = 1 + a1*h1 + a2*(hi-h2)
            delta[i] = (1+a1*h1)**(b1-b2)*np.exp(g*(h1-h2)/(R*Tsl*(1+a1*h1)))*theta[i]**b2
    p   = psl * delta
    T   = Tsl * theta
    rho = p / (1716.0 * T)
    a_s = np.sqrt(1.4 * 1716.0 * T)
    return float(p), float(T), float(rho), float(a_s)

def solve_divergence_mach(lmin, AR, pinf, gamma=1.4):
    """Solve the quartic (in M^2) for divergence Mach number.
    Returns MD and qD."""
    A_c =  np.pi**2 * gamma**2 * pinf**2 / lmin**2
    B_c =  1 - 4*np.pi*gamma*pinf / (lmin * AR)
    C_c = -1 + 4 / AR**2
    coeffs = [A_c, B_c, C_c]          # A*M^4 + B*M^2 + C = 0
    roots  = np.roots(coeffs)          # roots in M^2
    real_pos = [r.real for r in roots if r.real > 0 and abs(r.imag) < 1e-8]
    if not real_pos:
        return None, None
    MD = np.sqrt(max(real_pos))
    qD = gamma / 2 * pinf * MD**2
    return MD, qD

# 1 
# Tapered unswept wing, energy / Galerkin method
print("=" * 60)
print("PROBLEM 1")
print("=" * 60)

b_span = 100.0          # wingspan (ft)
L      = b_span / 2     # semi-span (ft)
cr, ct = 20.0, 10.0
h_alt  = 20000.0

x = np.linspace(0, 1, 1001)   # normalised coordinate x/L
GJ = 0.5e8 * (-1.3083*x**3 + 5.0010*x**2 - 6.4473*x + 2.7855)
c  = (ct - cr)*x + cr          # chord (ft)
e  = 0.15 * c                  # elastic axis offset (ft)

# Trial functions satisfying θ(0)=0 and θ'(L)=0
# Ψ1 = x^2/L^2 - 2x/L,  Ψ2 = x^3/L^3 - 3x/L
Psi   = np.column_stack([x**2 - 2*x,          x**3 - 3*x])
Psi1p = np.column_stack([(2*x - 2)/L,         (3*x**2 - 3)/L])

# Divergence
pinf, *_ = atmosphere(h_alt)[0], None, None, None
pinf = atmosphere(h_alt)[0]
gamma = 1.4

K             = np.zeros((2, 2))
dKa_dqcla     = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        K[i, j]         = np.trapz(Psi1p[:, i] * GJ * Psi1p[:, j], x*L)
        dKa_dqcla[i, j] = -np.trapz(Psi[:, i] * c * e * Psi[:, j], x*L)

print(f"\nGeneralised stiffness K (×10^6):\n{K/1e6}")
print(f"\nKa/(q_inf*cla) (×10^3):\n{dKa_dqcla/1e3}")

# Eigenvalue problem: K*q = lambda * Ka/(q*cla) * q
# K - lambda * (Ka/q_inf/cla) = 0  → lambda = q_D * cla
eigvals = nla.eigvals(nla.solve(dKa_dqcla, K))   # equivalent to eig(K, dKa)
print(f"\nEigenvalues λ = q_D*c_la: {eigvals}")
lmin = np.min(np.abs(eigvals))
print(f"q_D * c_la = {lmin:.4f}  (solution: 2102.2)")

AR = 2 * b_span / (cr + ct)   # aspect ratio = b / c_bar
print(f"Aspect ratio AR = {AR:.4f}")

MD, qD = solve_divergence_mach(lmin, AR, pinf, gamma)
print(f"Divergence Mach number MD = {MD:.4f}  (solution: 0.7045)")
print(f"Divergence dynamic pressure qD = {qD:.4f} psf  (solution: 337.84 psf)")

# Torsional rotation and shear stress at M=0.65
Minf = 0.65
qinf = gamma / 2 * pinf * Minf**2
clalpha = 2*np.pi / (np.sqrt(1 - Minf**2) + 2/AR)
Ka  = qinf * clalpha * dKa_dqcla

cmac = -0.1
W    = 100000.0

Q = np.zeros(2)
for i in range(2):
    Q[i] = np.trapz(
        Psi[:, i] * (qinf * c**2 * cmac + 2*W*e/(np.pi*L) * np.sqrt(1 - x**2)),
        x * L
    )

# With aeroelasticity
q_ae  = nla.solve(K + Ka, Q)
# Without aeroelasticity (Ka = 0)
q_nae = nla.solve(K, Q)

theta_ae  = Psi @ q_ae
theta_nae = Psi @ q_nae

# Max shear stress: τ_xy = G * Ψ' * q * t_max
# tmax = 0.1*c (airfoil thickness = 10% chord)
# G_al = 4.06e6 psi; convert: G in lb/ft^2 = G_psi * 144
G_al  = 4.06e6 * 144    # lb/ft^2
tmax  = 0.1 * c          # ft (max thickness)
tau_ae  = G_al * (Psi1p @ q_ae)  * tmax / 144   # back to psi
tau_nae = G_al * (Psi1p @ q_nae) * tmax / 144

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 5))

ax1a.plot(x, theta_ae  * 180/np.pi, label='w/ Aeroelasticity')
ax1a.plot(x, theta_nae * 180/np.pi, label='w/o Aeroelasticity')
ax1a.set_xlabel('x/L');  ax1a.set_ylabel('θ (deg)')
ax1a.set_title('Problem 1b – Torsional Rotation')
ax1a.legend(loc='upper left');  ax1a.grid(True, linestyle='--', alpha=0.5)

ax1b.plot(x, tau_ae,  label='w/ Aeroelasticity')
ax1b.plot(x, tau_nae, label='w/o Aeroelasticity')
ax1b.set_xlabel('x/L');  ax1b.set_ylabel('τ_xy (psi)')
ax1b.set_title('Problem 1b – Max Shear Stress')
ax1b.legend(loc='lower right');  ax1b.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('problem1_torsion_stress.png', dpi=150)
print("Saved problem1_torsion_stress.png")

# 2 Swept wing, Galerkin method (bending + torsion coupled)
print("\n" + "=" * 60)
print("PROBLEM 2")
print("=" * 60)

L2   = 60.0;  c2 = 20.0;  e2 = 0.15 * c2
EI   = 1e8;   GJ2 = 2e8
b2   = 2 * L2        # full wingspan for AR

# Find first root μ₁ of  cos(μ) + 1/cosh(μ) = 0
mu1 = _brentq(lambda mu: np.cos(mu) + 1/np.cosh(mu), 1.5, 2.0)
gam = (np.cos(mu1) + np.cosh(mu1)) / (np.sin(mu1) + np.sinh(mu1))
print(f"μ₁ = {mu1:.6f},  γ₁ = {gam:.6f}")

x2 = np.linspace(0, 1, 1001)  # normalised x/L

Phi   = (np.cosh(mu1*x2) - np.cos(mu1*x2)
         - gam*(np.sinh(mu1*x2) - np.sin(mu1*x2)))
Phi1p = mu1/L2 * (np.sinh(mu1*x2) + np.sin(mu1*x2)
                  - gam*(np.cosh(mu1*x2) - np.cos(mu1*x2)))
Phi2p = mu1**2/L2**2 * (np.cosh(mu1*x2) + np.cos(mu1*x2)
                         - gam*(np.sinh(mu1*x2) + np.sin(mu1*x2)))
Psi2  = np.sin(0.5*np.pi*x2)
Psi1p2 = 0.5*np.pi/L2 * np.cos(0.5*np.pi*x2)

# Structural stiffness matrix (diagonal, no bending-torsion coupling for straight refs)
K2 = np.zeros((2, 2))
K2[0, 0] = np.trapz(Phi2p * EI  * Phi2p, x2*L2)
K2[1, 1] = np.trapz(Psi1p2 * GJ2 * Psi1p2, x2*L2)
print(f"\nK (structural):\n{K2}")

def swept_Ka_over_qcla(Lambda_deg):
    # Aerodynamic stiffness matrix Ka / (q*cla) for sweep angle Lambda (deg)
    cL = np.cos(np.deg2rad(Lambda_deg))
    sL = np.sin(np.deg2rad(Lambda_deg))
    tL = np.tan(np.deg2rad(Lambda_deg))
    Ka = np.zeros((2, 2))
    Ka[0, 0] = c2*cL**2 * np.trapz((Phi - e2*sL*Phi1p)*tL*Phi1p,  x2*L2)
    Ka[0, 1] = c2*cL**2 * np.trapz(-(Phi - e2*sL*Phi1p)*Psi2,     x2*L2)
    Ka[1, 0] = c2*cL**2 * np.trapz(Psi2*e2*sL*Phi1p,              x2*L2)
    Ka[1, 1] = c2*cL**2 * np.trapz(-Psi2*e2*cL*Psi2,              x2*L2)
    return Ka

AR2 = 100 / c2   # = 5  
pinf2 = atmosphere(h_alt)[0]

for case_name, Lambda_deg in [("(a) Forward sweep Λ=−30°", -30),
                               ("(b) Backward sweep Λ=+30°", 30),
                               ("(c) Unswept Λ=0°",          0)]:
    dKa = swept_Ka_over_qcla(Lambda_deg)
    print(f"\n{case_name}")
    print(f"  Ka/(q*cla):\n  {dKa}")

    # Eigenvalue problem: K q = λ * (Ka/q_inf/cla) q
    # Solve as generalised eigenproblem only if dKa is non-singular
    try:
        eigv = nla.eigvals(nla.solve(dKa, K2))
        neg  = [v.real for v in eigv if v.real < 0 and abs(v.imag) < 1e-6]
        if neg:
            lmin2 = min(abs(v) for v in neg)
            print(f"  q_D*c_la = {lmin2:.4f}")
            MD2, qD2 = solve_divergence_mach(lmin2, AR2, pinf2, gamma)
            print(f"  MD = {MD2:.4f},  qD = {qD2:.4f} psf")
        else:
            print("  All eigenvalues positive → NO divergence")
    except nla.LinAlgError:
        # dKa singular (Lambda=0 case): treat as 1-DOF torsion only
        lmin2 = abs(K2[1, 1] / dKa[1, 1])
        print(f"  q_D*c_la (1-DOF) = {lmin2:.4f}")
        MD2, qD2 = solve_divergence_mach(lmin2, AR2, pinf2, gamma)
        print(f"  MD = {MD2:.4f},  qD = {qD2:.4f} psf")

# 3 
# Control reversal, iterative solution

print("\n" + "=" * 60)
print("PROBLEM 3")
print("=" * 60)

k2_const = np.pi**2 / 8
k9       = 0.2822
k8       = 1.0          # ≈1 since lf << L
GJ3      = 5e6          # ft-lb^2
L3       = 20.0         # ft
lf       = 0.2 * L3
c3       = 5.0
e3       = 0.15 * c3
cf       = 0.25 * c3
cs       = 1 - 2*cf/c3  # = 0.5

T4  = -np.arccos(cs) + cs*np.sqrt(1 - cs**2)
T10 =  np.arccos(cs) + np.sqrt(1 - cs**2)
print(f"cs={cs:.4f},  T4={T4:.4f},  T10={T10:.4f}")

cmac_delta = -(T4 + 10) / 2
pinf3 = atmosphere(h_alt)[0]

print(f"\n{'Iter':>4}  {'MR':>8}  {'c*_la':>10}  {'qR (psf)':>12}")
MR = 0.0
for i in range(10):
    clalpha3 = 2*np.pi / np.sqrt(1 - MR**2)
    cldelta   = clalpha3 / np.pi * T10
    qR = (2*k2_const * GJ3 /
          (L3**2 * c3 * e3 * clalpha3 *
           (1 - 4*k8**2*lf/L3 * (cmac_delta/cldelta * c3/e3 + k9))))
    MR_new = np.sqrt(2*qR / (gamma * pinf3))
    print("  iter %d   MR=%.4f   cla=%.4f   qR=%.4f psf" % (i, MR, clalpha3, qR))
    MR = MR_new

print("Converged: MR = %.4f  (solution: 0.5084)" % MR)
print("           qR = %.4f psf  (solution: 175.9380 psf)" % qR)

plt.show()
print("Done.")
