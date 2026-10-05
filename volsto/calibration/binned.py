"""Binned ("hybrid") regression stage of the particle calibration (SPEC §4.1).

The estimator of :mod:`volsto.calibration.particle` — local-linear regression of ``V`` on
``k = ln(S/F)`` at the 201 nodes spanning the cloud's trusted quantiles, Gaussian kernel truncated
at four bandwidths, the k-NN floor of the tails — without sorting the particles.  What the sorted
path reads off the sorted array is obtained exactly, by counting and selection:

1. **Quantile nodes.**  The two order statistics that bound the trusted range, by an exact
   histogram selection (one counting pass finds the two cells that hold the ranks, a partition
   inside those cells gives the values).  The nodes are the sorted path's, bit for bit.
2. **Floor decision and window ranks.**  The number of particles below ``g − 4h``, ``g`` and
   ``g + 4h`` for every node ``g`` (the sorted path's three ``searchsorted``), counted in one
   pass: integers, so the floored set and the rank windows are the sorted path's.
3. **Floor bandwidths.**  The order statistics at the window-edge ranks, from a partition of the
   two tail subsets that hold them (a few percent of the particles): the bandwidth
   ``max(h, span / 4)`` of every floored node is the sorted path's, bit for bit.
4. **Kernel sums.**  Unfloored nodes: five arrays linearly binned at ``h / BIN_FRACTION`` (the
   counts, the first two moments of ``k`` about the bin centre, ``V`` and its first moment), the
   kernel evaluated at the bin centres within ``4h`` of the node.  Floored nodes: the kernel summed
   over the particles of the rank window themselves (selected by value between the two exact
   window-edge order statistics), with the sorted path's formulas — the window is a hard cut in
   particle space where the kernel weight is still large, which bins cannot place.

Floored nodes therefore agree with the sorted path to the rounding of a sum taken in another
order; unfloored nodes carry the binning error, a relative 1e-6 (first steps) to 3e-4 (3y) on
``E[V|k]`` at ``h/4`` and 8·10⁵ particles (measured on real clouds, K.1 / K.5 of the calibration
speed study).  Linear binning also smooths every particle over one bin, which acts as a kernel of
variance ``h² + d²/6`` (``d`` the bin width): a uniform −0.0003 vol-point shift of the repriced
surface.  ``deflate=True`` evaluates the unfloored kernel sums with the bandwidth
``sqrt(h² − d²/6)``, which removes it (off by default).

Everything after the regression (bad-node interpolation, bias correction, interpolation onto
the leverage grid, tails) is shared with the sorted path
(:func:`volsto.calibration.particle._finish_estimate`).

Only the default kernel and regression are implemented (Gaussian, local-linear):
:class:`~volsto.config.ParticleConfig` refuses the binned estimator with any other.  The kernels
are serial ``numba`` loops: the result does not depend on the thread count.  A later option, not
implemented: the threshold counts and the binning pass visit the particles in the same order and
could share one loop (bit-identical, about 1 ms per step at 8·10⁵ particles).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit
from volsto.config import ParticleConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Bins per base bandwidth ``h`` (bin width ``h / 4``): the binned kernel sums differ from the
#: sorted ones by 1/100 of the estimate's own sampling noise at this width; ``h / 2`` would also
#: do, ``h / 8`` buys nothing (K.1).
BIN_FRACTION = 4
#: Half-width of the kernel window in bandwidths (the sorted path's Gaussian truncation).
KERNEL_SPAN = 4.0
#: Cells of the counting histogram that locates the two quantile ranks.
QUANTILE_CELLS = 8192
#: Cells of the lookup table that starts each particle's threshold search.
THRESHOLD_CELLS = 4096


@njit(cache=True)
def _minmax(k: FloatArray) -> tuple[float, float]:  # pragma: no cover - numba
    lo = k[0]
    hi = k[0]
    for i in range(k.shape[0]):
        x = k[i]
        if x < lo:
            lo = x
        elif x > hi:
            hi = x
    return lo, hi


@njit(cache=True)
def _hist_cells(k: FloatArray, lo: float, inv_cell: float, nc: int) -> IntArray:  # pragma: no cover
    """Particles per cell of a uniform grid of ``nc`` cells starting at ``lo``."""
    h = np.zeros(nc, dtype=np.int64)
    for i in range(k.shape[0]):
        c = int((k[i] - lo) * inv_cell)
        if c >= nc:
            c = nc - 1
        h[c] += 1
    return h


@njit(cache=True)
def _collect_cells(
    k: FloatArray, lo: float, inv_cell: float, nc: int, c0: int, c1: int, n0: int, n1: int
) -> tuple[FloatArray, FloatArray]:  # pragma: no cover - numba
    """The particles of cells ``c0`` and ``c1`` (``n0`` and ``n1`` of them, by the histogram)."""
    a0 = np.empty(n0)
    a1 = np.empty(n1)
    i0 = 0
    i1 = 0
    for i in range(k.shape[0]):
        c = int((k[i] - lo) * inv_cell)
        if c >= nc:
            c = nc - 1
        if c == c0:
            a0[i0] = k[i]
            i0 += 1
        if c == c1:
            a1[i1] = k[i]
            i1 += 1
    return a0, a1


def quantile_pair(k: FloatArray, i_lo: int, i_hi: int, lo: float, hi: float) -> tuple[float, float]:
    """The order statistics of ranks ``i_lo`` and ``i_hi`` of ``k`` (0-based), exactly.

    The cell index is a non-decreasing function of the value, so the cells hold contiguous rank
    ranges: the cell of a rank is found from the cumulative counts and the order statistic is
    the one of rank ``i − (particles in the cells below)`` among the particles of that cell.
    ``lo``, ``hi``: the extremes of ``k`` (``hi > lo``)."""
    nc = QUANTILE_CELLS
    inv_cell = nc / (hi - lo)
    cum = np.cumsum(_hist_cells(k, lo, inv_cell, nc))
    c0 = int(np.searchsorted(cum, i_lo, side="right"))
    c1 = int(np.searchsorted(cum, i_hi, side="right"))
    below0 = int(cum[c0 - 1]) if c0 > 0 else 0
    below1 = int(cum[c1 - 1]) if c1 > 0 else 0
    a0, a1 = _collect_cells(
        k, lo, inv_cell, nc, c0, c1, int(cum[c0]) - below0, int(cum[c1]) - below1
    )
    r0 = i_lo - below0
    r1 = i_hi - below1
    return float(np.partition(a0, r0)[r0]), float(np.partition(a1, r1)[r1])


@njit(cache=True)
def _count_less(
    k: FloatArray, thresholds: FloatArray, lo: float, inv_cell: float, lut: IntArray
) -> IntArray:  # pragma: no cover - numba
    """Number of particles strictly below each of the sorted ``thresholds`` (exact: what
    ``searchsorted`` of the threshold in the sorted particles returns).  ``lut[c]``: where the
    search of a particle of cell ``c`` starts; the two loops make the result independent of it."""
    nt = thresholds.shape[0]
    hist = np.zeros(nt + 1, dtype=np.int64)
    nc = lut.shape[0]
    for i in range(k.shape[0]):
        x = k[i]
        c = int((x - lo) * inv_cell)
        if c < 0:
            c = 0
        elif c >= nc:
            c = nc - 1
        b = lut[c]
        while b > 0 and thresholds[b - 1] > x:
            b -= 1
        while b < nt and thresholds[b] <= x:
            b += 1  # b = number of thresholds <= x
        hist[b] += 1
    out = np.empty(nt, dtype=np.int64)
    s = 0
    for j in range(nt):
        s += hist[j]
        out[j] = s  # particles with at most j thresholds at or below them: x < thresholds[j]
    return out


@njit(cache=True)
def _select_tails(
    k: FloatArray, v: FloatArray, cut_lo: float, cut_hi: float, n_lo: int, n_hi: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:  # pragma: no cover - numba
    """The particles with ``k < cut_lo`` (exactly ``n_lo``) and with ``k >= cut_hi`` (exactly
    ``n_hi``), each in particle order, with their ``v``."""
    kl = np.empty(n_lo)
    vl = np.empty(n_lo)
    kh = np.empty(n_hi)
    vh = np.empty(n_hi)
    a = 0
    b = 0
    for i in range(k.shape[0]):
        x = k[i]
        if x < cut_lo:
            kl[a] = x
            vl[a] = v[i]
            a += 1
        elif x >= cut_hi:
            kh[b] = x
            vh[b] = v[i]
            b += 1
    return kl, vl, kh, vh


@njit(cache=True)
def _bin5(
    k: FloatArray,
    v: FloatArray,
    a: float,
    inv_d: float,
    w0: FloatArray,
    w1: FloatArray,
    w2: FloatArray,
    y0: FloatArray,
    y1: FloatArray,
) -> None:  # pragma: no cover - numba
    """Linear binning onto the centres ``a + j d``: counts ``w0``, ``Σ e`` and ``Σ e²`` with
    ``e = k − centre`` (``w1``, ``w2``), ``Σ v`` and ``Σ v e`` (``y0``, ``y1``)."""
    for i in range(k.shape[0]):
        x = (k[i] - a) * inv_d
        j = int(x)
        f = x - j
        g = 1.0 - f
        e0 = f / inv_d
        e1 = -g / inv_d
        vi = v[i]
        w0[j] += g
        w0[j + 1] += f
        w1[j] += g * e0
        w1[j + 1] += f * e1
        w2[j] += g * e0 * e0
        w2[j + 1] += f * e1 * e1
        y0[j] += g * vi
        y0[j + 1] += f * vi
        y1[j] += g * e0 * vi
        y1[j + 1] += f * e1 * vi


@njit(cache=True)
def _node_sums(
    w0: FloatArray,
    w1: FloatArray,
    w2: FloatArray,
    y0: FloatArray,
    y1: FloatArray,
    a: float,
    d: float,
    grid: FloatArray,
    hg: FloatArray,
    x_lo: FloatArray,
    x_hi: FloatArray,
) -> tuple[FloatArray, FloatArray]:  # pragma: no cover - numba
    """Local-linear estimate and slope at every node from the bins whose centres lie in the
    node's window ``[x_lo, x_hi]``, with the node's bandwidth ``hg`` (``nan`` on an empty
    window; the Nadaraya–Watson value where the design is degenerate, as the sorted kernel)."""
    ng = grid.shape[0]
    m_bins = w0.shape[0]
    out = np.empty(ng)
    slope = np.zeros(ng)
    for gi in range(ng):
        g = grid[gi]
        h = hg[gi]
        jl = max(0, int(np.ceil((x_lo[gi] - a) / d)))
        jh = min(m_bins - 1, int(np.floor((x_hi[gi] - a) / d)))
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        t0 = 0.0
        t1 = 0.0
        for j in range(jl, jh + 1):
            x = a + j * d - g
            u = x / h
            kq = np.exp(-0.5 * u * u)
            s0 += kq * w0[j]
            s1 += kq * (w0[j] * x + w1[j])
            s2 += kq * (w0[j] * x * x + 2.0 * x * w1[j] + w2[j])
            t0 += kq * y0[j]
            t1 += kq * (y0[j] * x + y1[j])
        if s0 <= 0.0:
            out[gi] = np.nan
        else:
            det = s0 * s2 - s1 * s1
            if det > 1e-12 * s0 * s2:
                out[gi] = (s2 * t0 - s1 * t1) / det
                slope[gi] = (s0 * t1 - s1 * t0) / det
            else:
                out[gi] = t0 / s0
    return out, slope


@njit(cache=True)
def _exact_nodes(
    kt: FloatArray,
    vt: FloatArray,
    grid: FloatArray,
    hg: FloatArray,
    x_lo: FloatArray,
    x_hi: FloatArray,
    idx: IntArray,
    out: FloatArray,
    slope: FloatArray,
) -> None:  # pragma: no cover - numba
    """The floored nodes ``idx`` summed over the particles ``kt``, ``vt`` of one tail, in particle
    order, with the formulas of :func:`volsto.calibration.particle.kernel_regression`: the rank
    window of a node is exactly the particles with ``x_lo <= k <= x_hi``."""
    for q in range(idx.shape[0]):
        gi = idx[q]
        g = grid[gi]
        h = hg[gi]
        a = x_lo[gi]
        b = x_hi[gi]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        t0 = 0.0
        t1 = 0.0
        for i in range(kt.shape[0]):
            x = kt[i]
            if x >= a and x <= b:
                u = (x - g) / h
                wt = np.exp(-0.5 * u * u)
                s0 += wt
                s1 += wt * u
                s2 += wt * u * u
                t0 += wt * vt[i]
                t1 += wt * u * vt[i]
        if s0 <= 0.0:
            out[gi] = np.nan
        else:
            det = s0 * s2 - s1 * s1
            if det > 1e-12 * s0 * s2:
                out[gi] = (s2 * t0 - s1 * t1) / det
                slope[gi] = (s0 * t1 - s1 * t0) / det / h
            else:
                out[gi] = t0 / s0


def binned_regression(
    k: FloatArray, v: FloatArray, h: float, cfg: ParticleConfig, *, deflate: bool = False
) -> tuple[FloatArray, float, float, FloatArray, FloatArray] | None:
    """The regression stage on unsorted particles: ``(kreg, q_lo, q_hi, m, slope)`` — the
    nodes, the trusted quantile range, ``E[v | k]`` at the nodes (``nan`` where a window is empty)
    and its slope — or ``None`` for a degenerate cloud (the caller then uses the plain mean).

    ``deflate``: unfloored kernel sums with the bandwidth ``sqrt(h² − d²/6)`` (module docstring).
    """
    n = k.size
    i_lo = min(int(cfg.quantile_clip * n), n - 1)
    i_hi = max(int((1.0 - cfg.quantile_clip) * n) - 1, 0)
    lo, hi = _minmax(k)
    if not hi > lo:
        return None
    q_lo, q_hi = quantile_pair(k, i_lo, i_hi, lo, hi)
    if not q_hi > q_lo:
        return None
    kreg = np.linspace(q_lo, q_hi, cfg.n_regression_points)
    ng = kreg.size
    window = max(cfg.min_window, int(cfg.min_window_fraction * n))
    w = KERNEL_SPAN

    # the sorted path's three searchsorted per node, by counting
    unsorted = np.concatenate([kreg - w * h, kreg + w * h, kreg])
    order = np.argsort(unsorted, kind="stable")
    thresholds = unsorted[order]
    inv_cell = THRESHOLD_CELLS / (hi - lo)
    lut = np.searchsorted(
        thresholds, lo + np.arange(THRESHOLD_CELLS) / inv_cell, side="right"
    ).astype(np.int64)
    counts = _count_less(k, thresholds, lo, inv_cell, lut)
    by_node = np.empty(3 * ng, dtype=np.int64)
    by_node[order] = counts
    c_lo, c_hi, c_mid = by_node[:ng], by_node[ng : 2 * ng], by_node[2 * ng :]
    floored = ((c_hi - c_lo) < window) & (n >= window)

    hg = np.full(ng, h)
    x_lo = kreg - w * h
    x_hi = kreg + w * h
    tails: list[tuple[FloatArray, FloatArray, IntArray]] = []
    if floored.any():
        idx = np.where(floored)[0]
        # rank window of each floored node (particle.py, kernel_regression)
        rl = np.maximum(0, c_mid[idx] - window // 2)
        rh = np.minimum(n, rl + window)
        rl = np.maximum(0, rh - window)
        put = idx < ng // 2
        need_lo = int(rh[put].max()) if put.any() else 0
        need_hi = int(rl[~put].min()) if (~put).any() else n
        # the smallest threshold with at least need_lo particles below it, the largest with at
        # most need_hi: the two tail subsets hold every window-edge rank
        j_lo = int(np.searchsorted(counts, need_lo, side="left"))
        j_hi = int(np.searchsorted(counts, need_hi, side="right")) - 1
        e_lo = np.empty(idx.size)
        e_hi = np.empty(idx.size)
        if j_lo < counts.size and j_hi >= 0 and counts[j_lo] <= counts[j_hi]:
            n_lo = int(counts[j_lo])
            off = int(counts[j_hi])
            kl, vl, kh, vh = _select_tails(k, v, thresholds[j_lo], thresholds[j_hi], n_lo, n - off)
            if put.any():
                ranks = np.unique(np.concatenate([rl[put], rh[put] - 1]))
                part = np.partition(kl, ranks)
                e_lo[put] = part[rl[put]]
                e_hi[put] = part[rh[put] - 1]
            if (~put).any():
                ranks = np.unique(np.concatenate([rl[~put], rh[~put] - 1])) - off
                part = np.partition(kh, ranks)
                e_lo[~put] = part[rl[~put] - off]
                e_hi[~put] = part[rh[~put] - 1 - off]
            tails = [(kl, vl, idx[put]), (kh, vh, idx[~put])]
        else:
            # the two tails overlap (a cloud so thin that the floor is active everywhere): the
            # order statistics from the whole array, the particles of each side by value
            ranks = np.unique(np.concatenate([rl, rh - 1]))
            part = np.partition(k, ranks)
            e_lo = part[rl]
            e_hi = part[rh - 1]
            for side in (put, ~put):
                if side.any():
                    sel = (k >= e_lo[side].min()) & (k <= e_hi[side].max())
                    tails.append(
                        (np.ascontiguousarray(k[sel]), np.ascontiguousarray(v[sel]), idx[side])
                    )
        g = kreg[idx]
        span = np.maximum(np.abs(e_lo - g), np.abs(e_hi - g))
        hg[idx] = np.maximum(h, span / w)
        x_lo[idx] = e_lo
        x_hi[idx] = e_hi

    d = h / BIN_FRACTION
    a = lo - d
    m_bins = int(np.ceil((hi - lo) / d)) + 4
    bins = np.zeros((5, m_bins))
    _bin5(k, v, a, 1.0 / d, bins[0], bins[1], bins[2], bins[3], bins[4])
    hb = np.sqrt(hg * hg - d * d / 6.0) if deflate else hg
    m, slope = _node_sums(bins[0], bins[1], bins[2], bins[3], bins[4], a, d, kreg, hb, x_lo, x_hi)
    for kt, vt, side_idx in tails:
        if side_idx.size:
            _exact_nodes(kt, vt, kreg, hg, x_lo, x_hi, side_idx, m, slope)
    return kreg, q_lo, q_hi, m, slope
