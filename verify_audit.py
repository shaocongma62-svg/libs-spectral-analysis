# -*- coding: utf-8 -*-
"""INDEPENDENT verification of the external audit's key claims, using the model
from gap.py.  We do NOT trust the audit or my earlier scripts; we recompute."""
import numpy as np
g, T, s2 = 1.2, 3, 1.0/np.sqrt(2.0)
def get_step_matrix(m, k, phi):
    if m % 3 == 0:   gu, gv = np.sqrt(g), 1.0/np.sqrt(g)
    elif m % 3 == 1: gu, gv = 1.0, 1.0
    else:            gu, gv = 1.0/np.sqrt(g), np.sqrt(g)
    phases = [-phi if (n+3) % 4 < 2 else phi for n in range(4)]
    M = np.zeros((8, 8), dtype=complex)
    for n in range(4):
        iu, iv = 2*n, 2*n+1
        nxt, prv = n+1, n-1
        if nxt == 4:
            M[iu, 0] = gu*s2*np.exp(1j*k); M[iu, 1] = gu*s2*1j*np.exp(1j*k)
        else:
            M[iu, 2*nxt] = gu*s2; M[iu, 2*nxt+1] = gu*s2*1j
        pf = np.exp(1j*phases[n])
        if prv == -1:
            M[iv, 6] = gv*s2*1j*pf*np.exp(-1j*k); M[iv, 7] = gv*s2*pf*np.exp(-1j*k)
        else:
            M[iv, 2*prv] = gv*s2*1j*pf; M[iv, 2*prv+1] = gv*s2*pf
    return M
def Uf(k, phi):
    return get_step_matrix(2,k,phi) @ get_step_matrix(1,k,phi) @ get_step_matrix(0,k,phi)

rng = np.random.default_rng(0)
print("=== claim 1: det U_F = 1 ===")
ds = []
for _ in range(200):
    k, p = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
    ds.append(np.linalg.det(Uf(k, p)))
ds = np.array(ds)
print("  det range: [%.16f, %.16f]  max|det-1| = %.3e" % (ds.min(), ds.max(), np.abs(ds-1).max()))

print("=== claim 2: reciprocal-conjugate pairing lambda <-> 1/lambda* ===")
err = 0.0; nmax = 0
for _ in range(50):
    k, p = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
    lam = np.linalg.eigvals(Uf(k, p))
    inv = 1.0/np.conj(lam)
    # distance between the two multisets
    d = np.zeros((8,8))
    for a in range(8):
        for b in range(8):
            d[a,b] = abs(lam[a]-inv[b])
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(d)
    err = max(err, d[r,c].max())
print("  max multiset distance |{lambda} - {1/lambda*}| = %.3e" % err)

print("=== claim 3: phi period is 2pi (not pi) ===")
for (ka, kb, nama, nb) in [(0.7, 0.7, 'phi=0', 'phi=2pi'), (0.7, 0.7, 'phi=0', 'phi=pi')]:
    d2 = np.linalg.norm(Uf(ka, 0) - Uf(kb, (0 if '2pi' not in nb else 0)))
    if '2pi' in nb:
        d2 = np.linalg.norm(Uf(ka, 0.0) - Uf(kb, 2*np.pi))
        print("  U(k,0) vs U(k,2pi): rel err = %.3e" % (d2/np.linalg.norm(Uf(ka,0))))
    else:
        dpi = np.linalg.norm(Uf(ka, 0.0) - Uf(kb, np.pi))
        print("  U(k,0) vs U(k,pi) : rel err = %.3e" % (dpi/np.linalg.norm(Uf(ka,0))))

print("=== claim 4: spectrum at phi=0.80 rad is real? ===")
for pp in [0.80, 0.0, 0.2*np.pi]:
    mxim = 0.0
    for k in np.linspace(-np.pi, np.pi, 401):
        lam = np.linalg.eigvals(Uf(k, pp))
        E = (1j/T)*np.log(lam)
        mxim = max(mxim, np.abs(E.imag).max())
    print("  phi=%.4f rad (phi/pi=%.4f): max|ImE| = %.3e" % (pp, pp/np.pi, mxim))

print("=== claim 5: candidate EP (0.635,0.449) isolated point or on a contour? ===")
# scan a small box; compute min over band pairs of |lambda_a - lambda_b| (discriminant ~0)
def mingap(k, p):
    lam = np.linalg.eigvals(Uf(k, p))
    d = np.abs(lam[:,None]-lam[None,:]) + np.eye(8)*1e18
    return d.min()
k0, p0 = 0.635*np.pi, 0.449*np.pi
ks = np.linspace(k0-0.05, k0+0.05, 41); ps = np.linspace(p0-0.05, p0+0.05, 41)
gapmap = np.array([[mingap(k,p) for p in ps] for k in ks])
print("  min gap in box = %.3e ; min gap AT candidate = %.3e" % (gapmap.min(), gapmap[20,20]))
# is the zero set a contour? count near-zero points along phi=0.449 line and along k=0.635 line
row = gapmap[20,:]   # along phi at k0
col = gapmap[:,20]   # along k at p0
print("  along k=k0 (phi varies): #near-zero(1e-4) = %d / %d" % (np.sum(row<1e-4), len(row)))
print("  along phi=p0 (k varies): #near-zero(1e-4) = %d / %d" % (np.sum(col<1e-4), len(col)))

print("=== claim 6: half-integer winding convergence at (0.635,0.449) ===")
def winding(k0, p0, lam0, r, npts=1441):
    th = np.linspace(0, 2*np.pi, npts)
    prev=None; diff=np.zeros(npts, complex)
    for ti,t in enumerate(th):
        k=k0+r*np.cos(t); p=p0+r*np.sin(t)
        lam=np.linalg.eigvals(Uf(k,p))
        o=np.argsort(np.abs(lam-lam0)); a,b=lam[o[0]],lam[o[1]]
        if prev is not None:
            pa,pb=prev
            if abs(a-pa)+abs(b-pb) > abs(a-pb)+abs(b-pa): a,b=b,a
        prev=(a,b); diff[ti]=a-b
    ang=np.unwrap(np.angle(diff)); return (ang[-1]-ang[0])/(2*np.pi)
lam0 = -1j
for r in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    print("  r=%.1e -> W = %+.4f" % (r, winding(k0, p0, lam0, r)))
