"""Per-generation inbreeding coefficients for the simulate loop (ADR 0033).

:class:`Inbreeding` is stepped once per mating iteration with that iteration's
``parents`` (rows of the previous iteration's offspring) and MZ ``twins``, and
returns the offspring's inbreeding coefficients over the whole simulated
ancestry, founders at F = 0. The engine is fixed by an
:class:`~simace.simulation.inbreeding_plan.InbreedingPlan`:

- ``densep``: the dense kinship recurrence ``K_t = P K_{t-1} P'`` in float32,
  two N x N arrays (``notes/issue-76-recommendation/densep.py``).
- ``xexact`` (approx mode): each family's ancestry traced exactly for
  ``depth`` generations, the boundary generation closed with its mean
  kinship (``notes/issue-76-recommendation/fusedx_pre.py``).
- ``walkd2``: per distinct mating, breadth-first gene-contribution walks of
  mother and father over the whole history, ``F = 0.5 sum_k c_m c_f D_k``
  (``notes/issue-76-recommendation/walkd2.py`` at ``L = G_sim``, so no
  boundary closure). History is int32 parents and float64 D and F.

Every kernel is bitwise thread-invariant: parallel loops write disjoint
outputs and no reduction depends on the thread count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from simace.simulation.inbreeding_plan import InbreedingPlan

    prange = range
else:
    from numba import prange

__all__ = ["Inbreeding"]


# ---------------------------------------------------------------- densep


@numba.njit(parallel=True, cache=True)
def _rows(K, m, f, Kp):
    N = K.shape[0]
    for i in prange(N):
        a, b = m[i], f[i]
        for j in range(N):
            Kp[i, j] = K[a, j] + K[b, j]


@numba.njit(parallel=True, cache=True)
def _cols(Kp, m, f, K):
    N = K.shape[0]
    for i in prange(N):
        for j in range(N):
            K[i, j] = (Kp[i, m[j]] + Kp[i, f[j]]) * 0.25


@numba.njit(parallel=True, cache=True)
def _diag_F(K, m, f, F):
    for i in prange(K.shape[0]):
        F[i] = K[m[i], f[i]]


class _Densep:
    """Dense float32 kinship of the current generation.

    Each sum of two values is rounded once to float32 and ``* 0.25`` is exact,
    so F matches the float64 recurrence to float32 rounding. F of an offspring
    is its parents' kinship; MZ co-twins' kinship is copied from the diagonal.
    """

    def __init__(self, N: int) -> None:
        self.K = np.zeros((N, N), np.float32)
        self.K[np.diag_indices(N)] = 0.5
        self.Kp = np.empty_like(self.K)

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        K = self.K
        m = np.ascontiguousarray(parents[:, 0], np.int64)
        f = np.ascontiguousarray(parents[:, 1], np.int64)
        Fk = np.empty(len(m), K.dtype)
        _diag_F(K, m, f, Fk)
        F = Fk.astype(np.float64)
        _rows(K, m, f, self.Kp)
        _cols(self.Kp, m, f, K)
        K[np.diag_indices(len(m))] = 0.5 * (1 + F)
        if len(twins):
            a, b = twins[:, 0], twins[:, 1]
            K[a, b] = K[b, a] = K[a, a]
        return F


# ---------------------------------------------------------------- walkd2


@numba.njit(cache=True)
def _bfs(root, par, stopkey, sc, q):
    """Breadth-first gene-contribution coefficients of root into sc, distinct ancestors in q.

    Generations are discrete, so the queue is ordered by generation and an
    entry is final before it is expanded. Founders (``key < stopkey``) are not
    expanded.
    """
    q[0] = root
    sc[root] = 1.0
    head = 0
    tail = 1
    while head < tail:
        a = q[head]
        head += 1
        if a >= stopkey:
            h = 0.5 * sc[a]
            for s in range(2):
                pk = par[a, s]
                if sc[pk] == 0.0:
                    q[tail] = pk
                    tail += 1
                sc[pk] += h
    return tail


@numba.njit(cache=True)
def _father_dot(root, par, stopkey, D, scm, scf, q):
    """The father's breadth-first walk fused with the dot product.

    An entry's coefficient is final when dequeued, so it is used, expanded and
    zeroed in one visit (later pushes only reach older generations).
    """
    q[0] = root
    scf[root] = 1.0
    head = 0
    tail = 1
    near = 0.0
    while head < tail:
        a = q[head]
        head += 1
        cf = scf[a]
        scf[a] = 0.0
        c = scm[a]
        if c != 0.0:
            near += np.float64(c) * np.float64(cf) * D[a]
        if a >= stopkey:
            h = 0.5 * cf
            for s in range(2):
                pk = par[a, s]
                if scf[pk] == 0.0:
                    q[tail] = pk
                    tail += 1
                scf[pk] += h
    return near


@numba.njit(cache=True)
def _walk_chunk(pm, pf, s, e, par, D, stopkey, scm, scf, qm, qf, out):
    p = s
    nm = 0
    prev = -1
    while p < e:
        m = pm[p]
        if prev >= 0 and par[m, 0] == par[prev, 0] and par[m, 1] == par[prev, 1]:
            scm[prev] = 0.0  # full sister of the previous mother: same ancestors, swap the self entry
            scm[m] = 1.0
            qm[0] = m
        else:
            for i in range(nm):
                scm[qm[i]] = 0.0
            nm = _bfs(m, par, stopkey, scm, qm)
        q = p
        while q < e and pm[q] == m:
            out[q] = 0.5 * _father_dot(pf[q], par, stopkey, D, scm, scf, qf)
            q += 1
        prev = m
        p = q
    for i in range(nm):
        scm[qm[i]] = 0.0


@numba.njit(parallel=True, cache=True)
def _walk(pm, pf, bounds, par, D, stopkey, scm, scf, qm, qf, out):
    for t in prange(bounds.shape[0] - 1):
        _walk_chunk(pm, pf, bounds[t], bounds[t + 1], par, D, stopkey, scm[t], scf[t], qm[t], qf[t], out)


def _chunks(pm: np.ndarray, T: int) -> np.ndarray:
    """T contiguous ranges of the mother-sorted pairs, cut at mother-group starts."""
    n = len(pm)
    cuts = [0]
    for t in range(1, T):
        i = int(np.searchsorted(pm, pm[min(n - 1, t * n // T)], "left"))
        cuts.append(max(i, cuts[-1]))
    cuts.append(n)
    return np.asarray(cuts, np.int64)


def _twin_alias(N: int, twins: np.ndarray) -> np.ndarray:
    """Each row's canonical genome row: a second co-twin maps to the first."""
    alias = np.arange(N, dtype=np.int64)
    if len(twins):
        alias[twins[:, 1]] = twins[:, 0]
    return alias


class _Walkd2:
    """Exact F by deduplicating path walks over the whole history (walkd2 at L = G_sim).

    Keys are ``generation * N + row``, founders generation 0. Each thread owns
    two scratch coefficient vectors over the whole history, float32 while the
    dyadic coefficients (``G_sim + 1`` bits) are exact in it, else float64, and
    two int32 queues. Work is split at mother-group starts and each pair's F
    is computed by one thread, so F does not depend on the thread count.
    """

    def __init__(self, N: int, G_sim: int) -> None:
        self.N = N
        self.g = 0
        tot = N * (G_sim + 1)
        self.par = np.zeros((tot, 2), np.int32)
        self.D = np.ones(tot)
        self.Fg = np.zeros(tot)
        self.last_alias = np.arange(N)
        self.T = numba.get_num_threads()
        win = tot
        sdt = np.float32 if G_sim <= 23 else np.float64
        self.scm = np.zeros((self.T, win), sdt)
        self.scf = np.zeros((self.T, win), sdt)
        qmax = sum(min(1 << k, N) for k in range(G_sim + 1))
        self.qm = np.empty((self.T, qmax), np.int32)
        self.qf = np.empty((self.T, qmax), np.int32)

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        N, g, last_alias = self.N, self.g, self.last_alias
        alias = _twin_alias(N, twins)
        rep = alias == np.arange(N)
        cr = np.flatnonzero(rep)
        mloc = last_alias[parents[cr, 0]]
        floc = last_alias[parents[cr, 1]]
        keys = mloc * N + floc
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        first = np.empty(len(ks), bool)
        first[0] = True
        np.not_equal(ks[1:], ks[:-1], out=first[1:])
        uniq = ks[first]
        inv = np.empty(len(ks), np.int64)
        inv[order] = np.cumsum(first) - 1
        pm = uniq // N + g * N
        pf = uniq % N + g * N
        out = np.empty(len(uniq))
        _walk(pm, pf, _chunks(pm, self.T), self.par, self.D, N, self.scm, self.scf, self.qm, self.qf, out)
        F = np.empty(N)
        F[cr] = out[inv]
        F[~rep] = F[alias[~rep]]
        lo = (g + 1) * N
        pmg = last_alias[parents[:, 0]] + g * N
        pfg = last_alias[parents[:, 1]] + g * N
        self.par[lo : lo + N, 0] = pmg
        self.par[lo : lo + N, 1] = pfg
        self.D[lo : lo + N] = 0.5 - (self.Fg[pmg] + self.Fg[pfg]) / 4
        self.Fg[lo : lo + N] = F
        self.last_alias = alias
        self.g += 1
        return F


# ---------------------------------------------------------------- xexact
#
# notes/issue-76-recommendation/fusedx_pre.py with ABAR=down DESC=par FAM=par (DESC=ser FAM=count for
# Wright-Fisher, whose offspring are not mother-sorted) PACK=0|1 RECYCLE=0 FASTDRV=1. Each family (distinct
# mating) carries a tail: its ancestors within `depth` generations with their gene-contribution coefficients,
# sorted by id. A child's tail merges its parents' tails one generation down and adds the two parents. F is
# exact within the tail; ancestors at the boundary generation b = g - depth close with that generation's mean
# kinship abar, from the exact descending pass 1'A_b 1 = sum_j D_j y_j^2, y = T'1_b.
#
# Packed tails (ADR 0033 §5): one uint32 per entry, word = gr << SH | loc << MB | (mult - 1), SH = IB + MB.
# gr = gen - (h - depth) for a tail of an individual of generation h, so gr = 0 is the boundary generation and
# the child's offset is gr - 1. The coefficient at depth d = depth - gr from the parent is mult * 2^-d
# (dyadic, exact in float64). mult <= 2^(d-1) <= 2^(depth-1), because one individual fills at most the
# same-sex half of the 2^d slots at depth d, so MB = depth - 1 bits hold mult - 1. The word is an in-memory
# exception to the int32 floor for ids and counts and is never stored. Unpacked tails are an int32 global id
# and a float32 coefficient; F is bitwise the same either way.

NCH = 64  # fixed chunk count: sums reduce in the same order on any thread count


@numba.njit(cache=True)
def _lower(ids, s, e, key):
    while s < e:
        h = (s + e) >> 1
        if ids[h] < key:
            s = h + 1
        else:
            e = h
    return s


@numba.njit(parallel=True, cache=True)
def _caps(q0, q1, tst, tln, ids, pfam, fpar, bkey, has_b, bi, bj, cap):
    for q in prange(q0, q1):
        tm, tf = pfam[fpar[q, 0]], pfam[fpar[q, 1]]
        bi[q] = _lower(ids, tst[tm], tst[tm] + tln[tm], bkey) if has_b else tst[tm]
        bj[q] = _lower(ids, tst[tf], tst[tf] + tln[tf], bkey) if has_b else tst[tf]
        cap[q] = (tst[tm] + tln[tm] - bi[q]) + (tst[tf] + tln[tf] - bj[q]) + 2


@numba.njit(parallel=True, cache=True)
def _merge(q0, q1, tst, tln, ids, co, pfam, fpar, self_base, D, Fg, abar, has_b, bi, bj, nst, nln, nids, nco, F_out):
    for q in prange(q0, q1):
        tm, tf = pfam[fpar[q, 0]], pfam[fpar[q, 1]]
        diag = 0.0
        if has_b:
            ii, ie = tst[tm], bi[q]
            jj, je = tst[tf], bj[q]
            while ii < ie and jj < je:
                if ids[ii] == ids[jj]:
                    diag += np.float64(co[ii]) * np.float64(co[jj]) * (1.0 + Fg[ids[ii]] - abar)
                    ii += 1
                    jj += 1
                elif ids[ii] < ids[jj]:
                    ii += 1
                else:
                    jj += 1
        i, ie = bi[q], tst[tm] + tln[tm]
        j, je = bj[q], tst[tf] + tln[tf]
        o = nst[q]
        near = 0.0
        while i < ie or j < je:
            if j >= je or (i < ie and ids[i] < ids[j]):
                a = ids[i]
                v = 0.5 * co[i]
                i += 1
            elif i >= ie or ids[j] < ids[i]:
                a = ids[j]
                v = 0.5 * co[j]
                j += 1
            else:
                a = ids[i]
                near += np.float64(co[i]) * np.float64(co[j]) * D[a]
                v = 0.5 * (co[i] + co[j])
                i += 1
                j += 1
            nids[o] = a
            nco[o] = v
            o += 1
        sm, sf = self_base + fpar[q, 0], self_base + fpar[q, 1]
        nids[o] = min(sm, sf)
        nids[o + 1] = max(sm, sf)
        nco[o] = 0.5
        nco[o + 1] = 0.5
        o += 2
        nln[q] = o - nst[q]
        F_out[q] = 0.5 * (near + abar + diag) if has_b else 0.5 * near


@numba.njit(cache=True)
def _lower_pk(w, s, e, key):
    while s < e:
        h = (s + e) >> 1
        if np.int64(w[h]) < key:
            s = h + 1
        else:
            e = h
    return s


@numba.njit(parallel=True, cache=True)
def _caps_pk(q0, q1, tst, tln, w, pfam, fpar, bkey, has_b, bi, bj, cap):
    for q in prange(q0, q1):
        tm, tf = pfam[fpar[q, 0]], pfam[fpar[q, 1]]
        bi[q] = _lower_pk(w, tst[tm], tst[tm] + tln[tm], bkey) if has_b else tst[tm]
        bj[q] = _lower_pk(w, tst[tf], tst[tf] + tln[tf], bkey) if has_b else tst[tf]
        cap[q] = (tst[tm] + tln[tm] - bi[q]) + (tst[tf] + tln[tf] - bj[q]) + 2


@numba.njit(parallel=True, cache=True)
def _merge_pk(
    q0, q1, tst, tln, w, pfam, fpar, L, MB, SH, N, base_gen, D, Fg, abar, has_b, bi, bj, nst, nln, nw, F_out, scale, bad
):
    mmask = (np.int64(1) << MB) - 1
    one = np.int64(1) << SH
    for q in prange(q0, q1):
        tm, tf = pfam[fpar[q, 0]], pfam[fpar[q, 1]]
        diag = 0.0
        if has_b:
            ii, ie = tst[tm], bi[q]
            jj, je = tst[tf], bj[q]
            while ii < ie and jj < je:
                ki = np.int64(w[ii]) >> MB
                kj = np.int64(w[jj]) >> MB
                if ki == kj:
                    ci = np.float64((np.int64(w[ii]) & mmask) + 1) * scale[L]
                    cj = np.float64((np.int64(w[jj]) & mmask) + 1) * scale[L]
                    loc = ki & ((np.int64(1) << (SH - MB)) - 1)
                    diag += ci * cj * (1.0 + Fg[base_gen * N + loc] - abar)
                    ii += 1
                    jj += 1
                elif ki < kj:
                    ii += 1
                else:
                    jj += 1
        i, ie = bi[q], tst[tm] + tln[tm]
        j, je = bj[q], tst[tf] + tln[tf]
        o = nst[q]
        near = 0.0
        while i < ie or j < je:
            wi = np.int64(w[i]) if i < ie else np.int64(0)
            wj = np.int64(w[j]) if j < je else np.int64(0)
            if j >= je or (i < ie and (wi >> MB) < (wj >> MB)):
                x = wi - one
                i += 1
            elif i >= ie or (wj >> MB) < (wi >> MB):
                x = wj - one
                j += 1
            else:
                gr = wi >> SH
                loc = (wi >> MB) & ((np.int64(1) << (SH - MB)) - 1)
                mi = (wi & mmask) + 1
                mj = (wj & mmask) + 1
                d = L - gr
                near += (np.float64(mi) * scale[d]) * (np.float64(mj) * scale[d]) * D[(base_gen + gr) * N + loc]
                if mi + mj - 1 > mmask:
                    bad[0] = 1
                x = ((wi >> MB) << MB) - one + (mi + mj - 1)
                i += 1
                j += 1
            nw[o] = np.uint32(x)
            o += 1
        sm, sf = fpar[q, 0], fpar[q, 1]
        top = np.int64(L - 1) << SH
        nw[o] = np.uint32(top | (min(sm, sf) << MB))
        nw[o + 1] = np.uint32(top | (max(sm, sf) << MB))
        o += 2
        nln[q] = o - nst[q]
        F_out[q] = 0.5 * (near + abar + diag) if has_b else 0.5 * near


@numba.njit(cache=True)
def _families(mloc, floc, N):
    """Families (distinct matings) by an O(N) counting sort on mother, in the stable argsort order of mloc * N + floc."""
    n = mloc.shape[0]
    cnt = np.zeros(N + 1, np.int64)
    for i in range(n):
        cnt[mloc[i] + 1] += 1
    for k in range(N):
        cnt[k + 1] += cnt[k]
    pos = cnt[:N].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        m = mloc[i]
        order[pos[m]] = i
        pos[m] += 1
    for m in range(N):  # stable sort by father inside each mother's run: order = argsort(keys, stable)
        s, e = cnt[m], cnt[m + 1]
        if e - s > 32:
            sub = order[s:e].copy()
            order[s:e] = sub[np.argsort(floc[sub], kind="mergesort")]
        else:
            for i in range(s + 1, e):
                x = order[i]
                fx = floc[x]
                j = i - 1
                while j >= s and floc[order[j]] > fx:
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = x
    fam = np.empty(n, np.int64)
    fpar = np.empty((n, 2), np.int64)
    nf = 0
    for i in range(n):
        x = order[i]
        if i == 0 or mloc[x] != mloc[order[i - 1]] or floc[x] != floc[order[i - 1]]:
            fpar[nf, 0] = mloc[x]
            fpar[nf, 1] = floc[x]
            nf += 1
        fam[x] = nf - 1
    return fam, fpar[:nf].copy()


@numba.njit(cache=True)
def _descend(par, D, b, rep_b, N):
    """1'A_b 1 over generation b's representatives, exact down to the founders (A = I)."""
    cur = np.zeros(N)
    for i in range(N):
        if rep_b[i]:
            cur[i] = 1.0
    nxt = np.zeros(N)
    Q = 0.0
    for k in range(b, 0, -1):
        base = k * N
        pbase = (k - 1) * N
        for i in range(N):
            y = cur[i]
            if y != 0.0:
                a = base + i
                Q += D[a] * y * y
                h = 0.5 * y
                nxt[par[a, 0] - pbase] += h
                nxt[par[a, 1] - pbase] += h
        cur, nxt = nxt, cur
        nxt[:] = 0.0
    for i in range(N):
        y = cur[i]
        if y != 0.0:
            Q += y * y  # founders: D = 1, A = I
    return Q


@numba.njit(cache=True)
def _cuts(key, s0, n, perm):
    """NCH + 1 chunk edges over key[s0 + perm[i]] (perm < 0: identity), each edge moved forward to a run start."""
    e = np.empty(NCH + 1, np.int64)
    e[0] = 0
    for t in range(1, NCH):
        i = max(e[t - 1], t * n // NCH)
        while 0 < i < n:
            a = key[s0 + (perm[i] if perm[0] >= 0 else i)]
            p = key[s0 + (perm[i - 1] if perm[0] >= 0 else i - 1)]
            if a != p:
                break
            i += 1
        e[t] = i
    e[NCH] = n
    return e


@numba.njit(parallel=True, cache=True)
def _descend_p(par, D, fperm, b, rep_b, N):
    """_descend with both scatters parallel.

    Children of a generation are sorted by mother, so chunks cut at mother
    runs own their mothers; fathers go through fperm (children of each
    generation sorted by father).
    """
    cur = np.zeros(N)
    nxt = np.zeros(N)
    for i in prange(N):
        if rep_b[i]:
            cur[i] = 1.0
    part = np.zeros(NCH)
    ident = np.full(1, -1, np.int64)
    Q = 0.0
    pm = par[:, 0]
    pf = par[:, 1]
    for k in range(b, 0, -1):
        base = k * N
        pbase = (k - 1) * N
        em = _cuts(pm, base, N, ident)
        fp = fperm[base : base + N]
        ef = _cuts(pf, base, N, fp)
        for t in prange(NCH):
            q = 0.0
            for i in range(em[t], em[t + 1]):
                y = cur[i]
                if y != 0.0:
                    a = base + i
                    q += D[a] * y * y
                    nxt[pm[a] - pbase] += 0.5 * y
            part[t] = q
        for t in prange(NCH):
            for x in range(ef[t], ef[t + 1]):
                i = fp[x]
                y = cur[i]
                if y != 0.0:
                    nxt[pf[base + i] - pbase] += 0.5 * y
        for t in range(NCH):
            Q += part[t]
        cur, nxt = nxt, cur
        for i in prange(N):
            nxt[i] = 0.0
    for t in prange(NCH):
        q = 0.0
        for i in range(t * N // NCH, (t + 1) * N // NCH):
            y = cur[i]
            if y != 0.0:
                q += y * y
        part[t] = q
    for t in range(NCH):
        Q += part[t]
    return Q


@numba.njit(parallel=True, cache=True)
def _count_perm_par(key, N, out, T):
    """Stable counting sort of local ids 0..N-1 by key into out (int32), in parallel.

    Ids scatter into key-range buckets (stable per chunk), then a counting
    sort runs inside each bucket. The output is the unique stable order.
    """
    n = key.shape[0]
    C = 4 * T
    B = 8 * T
    bw = (N + B - 1) // B
    hist = np.zeros((C, B), np.int64)
    for c in prange(C):
        for i in range(c * n // C, (c + 1) * n // C):
            hist[c, key[i] // bw] += 1
    off = np.empty((C, B), np.int64)
    bst = np.empty(B + 1, np.int64)
    s = 0
    for b in range(B):
        bst[b] = s
        for c in range(C):
            off[c, b] = s
            s += hist[c, b]
    bst[B] = s
    tmp = np.empty(n, np.int32)
    for c in prange(C):
        o = off[c].copy()
        for i in range(c * n // C, (c + 1) * n // C):
            b = key[i] // bw
            tmp[o[b]] = i
            o[b] += 1
    for b in prange(B):
        k0 = b * bw
        cnt = np.zeros(bw + 1, np.int64)
        for x in range(bst[b], bst[b + 1]):
            cnt[key[tmp[x]] - k0 + 1] += 1
        for k in range(bw):
            cnt[k + 1] += cnt[k]
        for x in range(bst[b], bst[b + 1]):
            i = tmp[x]
            r = key[i] - k0
            out[bst[b] + cnt[r]] = i
            cnt[r] += 1


@numba.njit(parallel=True, cache=True)
def _hist_par(mloc, floc, fam, Ff, g, N, par, D, Fg, F):
    """History write for generation g + 1 (parents, D, F) in one parallel pass."""
    lo = (g + 1) * N
    for i in prange(N):
        pm = mloc[i] + g * N
        pf = floc[i] + g * N
        par[lo + i, 0] = pm
        par[lo + i, 1] = pf
        D[lo + i] = 0.5 - (Fg[pm] + Fg[pf]) / 4
        f = Ff[fam[i]]
        Fg[lo + i] = f
        F[i] = f


@numba.njit(parallel=True, cache=True)
def _families_par(mloc, floc):
    """_families for mother-sorted input, parallel over NCH chunks cut at mother runs.

    Same order as the stable argsort of mloc * N + floc.
    """
    n = mloc.shape[0]
    ident = np.full(1, -1, np.int64)
    e = _cuts(mloc, 0, n, ident)
    order = np.arange(n)
    nf = np.zeros(NCH + 1, np.int64)
    for t in prange(NCH):
        s = e[t]
        c = 0
        while s < e[t + 1]:
            r = s + 1
            while r < e[t + 1] and mloc[r] == mloc[s]:
                r += 1
            for i in range(s + 1, r):  # stable insertion sort by father inside the mother run
                x = order[i]
                fx = floc[x]
                j = i - 1
                while j >= s and floc[order[j]] > fx:
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = x
            for i in range(s, r):
                if i == s or floc[order[i]] != floc[order[i - 1]]:
                    c += 1
            s = r
        nf[t + 1] = c
    for t in range(NCH):
        nf[t + 1] += nf[t]
    fam = np.empty(n, np.int64)
    fpar = np.empty((nf[NCH], 2), np.int64)
    for t in prange(NCH):
        f = nf[t] - 1
        for i in range(e[t], e[t + 1]):
            x = order[i]
            if i == e[t] or mloc[x] != mloc[order[i - 1]] or floc[x] != floc[order[i - 1]]:
                f += 1
                fpar[f, 0] = mloc[x]
                fpar[f, 1] = floc[x]
            fam[x] = f
    return fam, fpar


class _Xexact:
    """Depth-truncated F (approx mode): exact within ``depth`` generations, the rest closed with mean kinship."""

    def __init__(self, N: int, G_sim: int, depth: int, packed: bool, mother_sorted: bool) -> None:
        self.N, self.L, self.pack, self.sorted = N, depth, packed, mother_sorted
        self.g = 0
        tot = N * (G_sim + 1)
        self.par = np.zeros((tot, 2), np.int32)
        self.D = np.zeros(tot)
        self.D[:N] = 1.0
        self.Fg = np.zeros(tot)
        if mother_sorted:
            self.fperm = np.zeros(tot, np.int32)
            self.fperm[:N] = np.arange(N, dtype=np.int32)
        if packed:
            self.IB = max(1, (N - 1).bit_length())
            self.MB = depth - 1
            self.SH = self.IB + self.MB
            if max(1, (depth - 1).bit_length()) + self.SH > 32:
                raise ValueError(f"a packed tail word needs {max(1, (depth - 1).bit_length()) + self.SH} bits")
            self.scale = np.array([2.0**-k for k in range(depth + 1)])
            self.bad = np.zeros(1, np.int64)
        self.tst = np.zeros(1, np.int64)
        self.tln = np.zeros(1, np.int64)
        self.ids = np.empty(0, np.uint32 if packed else np.int32)
        self.co = np.empty(0, np.float32)
        self.pfam = np.zeros(N, np.int64)
        self.reps = [np.ones(N, bool)]
        self.last_alias = np.arange(N)

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        N, g, L = self.N, self.g, self.L
        par, D, Fg = self.par, self.D, self.Fg
        alias = _twin_alias(N, twins)
        b = g - L
        has_b = b >= 0
        abar = 0.0
        if has_b:
            rep_b = self.reps[b]
            if self.sorted:
                Q = _descend_p(par, D, self.fperm, b, rep_b, N)
            else:
                Q = _descend(par, D, b, rep_b, N)
            n = int(rep_b.sum())
            abar = (Q - (1.0 + Fg[b * N : (b + 1) * N])[rep_b].sum()) / (n * (n - 1))
        mloc = self.last_alias[parents[:, 0]]
        floc = self.last_alias[parents[:, 1]]
        if self.sorted:
            if np.any(mloc[1:] < mloc[:-1]):
                raise ValueError("xexact's parallel passes need offspring sorted by mother")
            fam, fpar = _families_par(mloc, floc)
        else:
            fam, fpar = _families(mloc, floc, N)
        nfam = len(fpar)
        bi = np.empty(nfam, np.int64)
        bj = np.empty(nfam, np.int64)
        cap = np.empty(nfam, np.int64)
        nln = np.empty(nfam, np.int64)
        Ff = np.empty(nfam)
        nst = np.zeros(nfam, np.int64)
        tst, tln, ids, pfam = self.tst, self.tln, self.ids, self.pfam
        if self.pack:
            bkey = 1 << self.SH
            _caps_pk(0, nfam, tst, tln, ids, pfam, fpar, bkey, has_b, bi, bj, cap)
        else:
            bkey = (b + 1) * N
            _caps(0, nfam, tst, tln, ids, pfam, fpar, bkey, has_b, bi, bj, cap)
        np.cumsum(cap[:-1], out=nst[1:])
        total = int(nst[-1] + cap[-1])
        if self.pack:
            nids = np.empty(total, np.uint32)
            _merge_pk(
                0,
                nfam,
                tst,
                tln,
                ids,
                pfam,
                fpar,
                L,
                self.MB,
                self.SH,
                N,
                g - L,
                D,
                Fg,
                abar,
                has_b,
                bi,
                bj,
                nst,
                nln,
                nids,
                Ff,
                self.scale,
                self.bad,
            )
            if self.bad[0]:
                raise RuntimeError("a packed tail multiplicity overflowed its bits, which the depth bound excludes")
        else:
            nids = np.empty(total, np.int32)
            nco = np.empty(total, np.float32)
            _merge(
                0, nfam, tst, tln, ids, self.co, pfam, fpar, g * N, D, Fg, abar, has_b, bi, bj, nst, nln, nids, nco, Ff
            )
            self.co = nco
        self.ids = nids
        self.tst, self.tln, self.pfam = nst, nln, fam
        lo = (g + 1) * N
        F = np.empty(N)
        _hist_par(mloc, floc, fam, Ff, g, N, par, D, Fg, F)
        if self.sorted:
            _count_perm_par(floc, N, self.fperm[lo : lo + N], numba.get_num_threads())
        self.reps.append(alias == np.arange(N))
        self.last_alias = alias
        self.g += 1
        return F


# ---------------------------------------------------------------- stepper


class Inbreeding:
    """The offspring inbreeding coefficients of each mating iteration, under one plan.

    Args:
        plan: the resolved plan; its ``engine`` picks the computation.
        N: individuals per generation.
        G_sim: mating iterations the simulation runs.
        mother_sorted: whether every iteration's offspring rows are sorted by
            mother (the standard mating model), which xexact's parallel
            passes need.
    """

    def __init__(self, plan: InbreedingPlan, N: int, G_sim: int, mother_sorted: bool) -> None:
        self.plan = plan
        if plan.engine == "densep":
            self._engine: _Densep | _Walkd2 | _Xexact = _Densep(N)
        elif plan.engine == "walkd2":
            self._engine = _Walkd2(N, G_sim)
        elif plan.engine == "xexact":
            assert plan.depth is not None
            self._engine = _Xexact(N, G_sim, plan.depth, plan.packed, mother_sorted)
        else:
            raise ValueError(f"no inbreeding engine {plan.engine!r}")

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        """Return the float64 inbreeding coefficients of the offspring of ``parents``, shape ``(N,)``.

        ``twins`` holds the ``(first, second)`` offspring rows of each MZ pair.
        """
        return self._engine.step(parents, np.asarray(twins, np.int64).reshape(-1, 2))
