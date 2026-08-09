"""high level evaluation API: one object binds a case, a W store and
the data, and exposes the marginal likelihood at any intrinsic point.
Everything here is a thin composition of the documented pieces
(profiles.profile, likelihood.zeta, kernel.Kernel, wtables.rebin);
nothing is computed that the notebooks could hide."""
import numpy as np

from .profiles import profile
from .tables import rebin
from .kernel import Kernel, load_sumstore, sum_lnlam_pair


def zeta(case, dtilde, oversample=8):
    '''
    the entry point of evaluation: turn a segment (real or
    synthetic) into the matched filter series — for every assumed
    arrival time, how well the data matches the template.

      z_I(t) = sum_nu conj(d)_I,nu / sig2_I,nu a_nu e^{i Psi0}
               e^{-2 pi i nu t / T}

    Outputs
    * t (n_time,), z {'H','L'} (n_time,) complex
    '''
    n_time = oversample * int(case.nu.max())
    t = np.arange(n_time) * (case.T / n_time)
    z = {}
    for I in 'HL':
        w = (np.conj(dtilde[I]) / case.sig2[I] * case.a
             * np.exp(1j * case.psi0))
        pad = np.zeros(n_time, dtype=complex)
        np.add.at(pad, case.nu, w)
        z[I] = np.fft.fft(pad)
    return t, z


def load_store(store):
    """a W store is (tau_f, {key: W row}) in Taylor units; accept
    that or an npz path.  npz stores must carry the norm flag —
    legacy absolute-unit stores are refused."""
    if isinstance(store, tuple):
        return store
    z = np.load(store)
    if 'norm' not in z.files or str(z['norm']) != 'taylor_off1':
        raise ValueError('store units are not the offset Taylor '
                         'convention; rebuild with '
                         'scripts/build_store.py')
    W = {tuple(int(v) for v in k): row.astype(float)
         for k, row in zip(z['keys'], z['W'])}
    for (n, a, b) in list(W):
        W[(n, b, a)] = W[(n, a, b)]
    return z['tau'], W


class Evaluator:
    """kernel path ln Lambda for one case, W store(s), one time window.

    ev = Evaluator(case, dtilde, 'wtaylor.npz', tc=3.7, dt=0.07)
    ev = Evaluator(case, dtilde, [(u0, store0), (u1, store1)], ...)
    ev.lnlam(m1, m2, chi)            # the bound data
    ev.lnlam(m1, m2, chi, other_d)   # same tables, other data

    Evaluating a template with weight u on a store built at u_node
    through T^(n) -> (u_node/u)^n T^(n) is exact for the distance
    prior with cutoff ymin_eff = ymin sqrt(u_node/u), NOT for the
    fixed-ymin prior of the paper (the two coincide only at
    u = u_node).  With a single store the evaluation therefore
    carries a loudness correlated tilt away from u_node.  With two
    or more stores at fixed ymin and different u nodes, ln Lambda is
    evaluated at the two nodes bracketing u and log-linearly
    interpolated in ln u; the nodes' effective cutoffs bracket the
    true ymin and the first order error cancels (validated at the
    0.05 level)."""

    def __init__(self, case, dtilde, store, tc, dt, qpad=1.15,
                 u_lo_frac=0.45):
        self.case, self.d, self.tc, self.dt = case, dtilde, tc, dt
        if isinstance(store, list):
            nodes = sorted(store, key=lambda p: p[0])
        else:
            nodes = [(case.u['H'], store)]
        self.u_nodes = np.array([float(un) for un, _ in nodes])
        t, z = zeta(case, dtilde)
        self.n_time = t.size
        tau_all = np.fft.fftfreq(self.n_time)*case.T
        qmax = max(np.abs(z['H']).max(), np.abs(z['L']).max())
        self.Wds, self.kernels = [], []
        for u_node, st in nodes:
            tau_f, W = load_store(st)
            keep = np.abs(tau_all) <= 1.05*np.abs(tau_f).max()
            tau_idx = np.rint(tau_all[keep]/(t[1]-t[0])).astype(int)
            Wd = rebin(tau_f, W, tau_all[keep])
            keys = np.array(sorted(k for k in W if k[2] <= k[1]))
            WM = np.array([Wd[tuple(k)] for k in keys])
            # the node grid top covers sqrt(u_node/u)|z| down to
            # u = u_lo_frac * min(u_nodes); single node keeps the
            # original top (bit compatible with the references)
            if len(nodes) == 1:
                top = qpad*qmax
            else:
                top = qpad*qmax*np.sqrt(
                    u_node/(u_lo_frac*self.u_nodes[0]))
            self.Wds.append(Wd)
            self.kernels.append(Kernel(u_node, tau_idx, self.n_time,
                                       case.T, keys, WM, top))
        # stores are in Taylor units: the exact-series weight is one
        # at every order; a shorter ones-vector truncates at n_max
        self.cn_nodes = [np.ones(k.N + 1) for k in self.kernels]
        self.u_store = self.u_nodes[0] if len(nodes) == 1 else None
        self.Wd = self.Wds[0]
        self.kernel = self.kernels[0]

    def _bracket(self, u):
        """indices and ln-u weight of the nodes bracketing u"""
        un = self.u_nodes
        if un.size == 1 or u <= un[0]:
            return 0, 0, 0.0
        if u >= un[-1]:
            return un.size-1, un.size-1, 0.0
        j = int(np.searchsorted(un, u))
        x = np.log(u/un[j-1])/np.log(un[j]/un[j-1])
        return j-1, j, x

    def template(self, m1, m2, chi, dtilde=None):
        """(z, u, profile object) for one aligned spin template.

        Heavy templates cut off inside the band (IMRPhenomXAS
        returns exact zeros above ~0.3/M_sec) and profile() drops
        those bins; the template is embedded back onto the case's
        fixed frequency grid (zero amplitude bins contribute nothing
        to z or u) so zeta always returns the fiducial n_time."""
        c = self.case
        nu, f, a1, p0 = profile(m1, m2, chi, chi, c.T, c.fmin, c.fmax,
                                1.0)
        a = np.zeros(c.nu.size)
        psi0 = np.zeros(c.nu.size)
        keep = (nu >= c.nu[0]) & (nu <= c.nu[-1])
        j = np.searchsorted(c.nu, nu[keep])
        a[j] = a1[keep]/c.D0
        psi0[j] = p0[keep]
        # the case carries the noise curve of its data; recomputing
        # from the module here could silently disagree with it
        sig2 = c.sig2
        u = (a**2/sig2['H']).sum()
        obj = type('P', (), dict(T=c.T, nu=c.nu, f=c.f, a=a,
                                 psi0=psi0, sig2=sig2))()
        _, z = zeta(obj, self.d if dtilde is None else dtilde)
        return z, u, obj

    def lnlam(self, m1, m2, chi, dtilde=None, cn=None):
        """ln Lambda (node bracketed, see class doc).  cn: optional
        weight vector relative to the exact Taylor series (ones);
        a shorter cn truncates at n_max = len(cn) - 1"""
        z, u, _ = self.template(m1, m2, chi, dtilde)
        j0, j1, x = self._bracket(u)
        cn0 = self.cn_nodes[j0] if cn is None else cn
        ln0 = self.kernels[j0].lnlam(z, u, self.tc, self.dt, cn0)
        if j1 == j0:
            return ln0
        cn1 = self.cn_nodes[j1] if cn is None else cn
        ln1 = self.kernels[j1].lnlam(z, u, self.tc, self.dt, cn1)
        return (1-x)*ln0 + x*ln1

    def lnsvec(self, m1, m2, chi, dtilde=None):
        """the per order log data sums at the bracketing nodes.

        Outputs (lns0, lns1, x); ln Lambda for ANY weight vector cn
        is (1-x) LSE(ln cn + lns0) + x LSE(ln cn + lns1) with LSE the
        logsumexp, so one call prices every truncation n_max <= N
        for free (the n_max scan)"""
        z, u, _ = self.template(m1, m2, chi, dtilde)
        j0, j1, x = self._bracket(u)
        l0 = self.kernels[j0].lns(z, u, self.tc, self.dt)
        if j1 == j0:
            return l0, l0, 0.0
        l1 = self.kernels[j1].lns(z, u, self.tc, self.dt)
        return l0, l1, x


class SumEvaluator:
    '''PE runtime on precompiled summed tables (kernel.SumKernel):
    loads a ladder of sumstore stems (mmap, opens in milliseconds),
    brackets the template u between neighbouring nodes and evaluates
    both in ONE fused sweep (kernel.sum_lnlam_pair).  the truncation
    n_max is frozen inside the tables; pick the stem set at load
    time.  a template u OUTSIDE the ladder raises — the per-order
    path's silent clamp analyzed with the wrong node and an O(1)
    lnLambda error, that failure class is forbidden here.'''

    def __init__(self, case, dtilde, stems, tc, dt, parallel=False):
        ks = sorted((load_sumstore(s) for s in stems),
                    key=lambda k: k.u_node)
        self.kernels = ks
        self.u_nodes = np.array([k.u_node for k in ks])
        self.case, self.d = case, dtilde
        self.tc, self.dt, self.parallel = tc, dt, parallel

    template = Evaluator.template

    def lnlam(self, m1, m2, chi, dtilde=None):
        z, u, _ = self.template(m1, m2, chi, dtilde)
        un = self.u_nodes
        if u < un[0] or u > un[-1]:
            raise ValueError(
                'template u=%.2f is outside the loaded ladder '
                '[%g, %g]; extend the stem set (silent clamping is '
                'forbidden)' % (u, un[0], un[-1]))
        j = int(np.searchsorted(un, u))
        j0, j1 = max(j - 1, 0), min(j, un.size - 1)
        return sum_lnlam_pair(self.kernels[j0], self.kernels[j1],
                              z, u, self.tc, self.dt,
                              parallel=self.parallel)


def mc_q_to_m1m2(Mc, q):
    m1 = Mc*(1+q)**0.2/q**0.6
    return m1, q*m1


_EV = None          # module level for the fork based pool


