"""M10 Part 0: the eSSVI calendar repair (:func:`volsto.market.import_hdn.repair_calendar`).

The invariant the repair guarantees, by construction, when it returns without a fallback
(``calendar_fallback is None`` — all 127 days of the 2022 H2 sample): the exact derivative
``∂_T w(k, T) ≥ margin`` for every ``|k| ≤ 3`` and every ``T`` in ``[1/365, max_maturity]``,
both one-sided limits at every knot included.  The escalation steps return WEAKER invariants,
recorded on the result: ``margin0`` proves ``∂_T w ≥ 0`` only, ``k_abs_1`` proves it on ``|k| ≤ 1``
only, and ``ssvi`` returns a plain SSVI (single ``ρ``, calendar-free by construction for
non-decreasing ``θ_T``) that the certificate does not examine.  The repair returns an eSSVI surface
only with a proof (:func:`volsto.market.surface.certify_calendar`); these tests re-check the proof
independently on the constructed surface, on the exact derivative over the dense grid and at
every knot.

Synthetic surfaces only (no market data): seeded random eSSVI parameter sets whose
``θ_T(1+ρ_T)`` is non-monotone, a synthetic vol-space residual, and the properties the repair
promises — the invariant on 500 seeded draws and on every draw the previous verifier found
failing, a constructible surface, both butterfly conditions, bounded ``ρ``, exact identity on
proven input, a projection rather than a clip, the certificate's cuts, and the documented
escalation order.  (``hypothesis`` is not a dependency: seeded numpy draws.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from volsto.market import ESSVISurface, ForwardCurve
from volsto.market.import_hdn import (
    CALENDAR_FALLBACKS,
    DEFAULT_CALENDAR_REPAIR,
    CalendarRepair,
    CalendarRepairConfig,
    _eta_max,
    calendar_constraint_grid,
    repair_calendar,
)
from volsto.market.surface import calendar_knots

FloatArray = NDArray[np.float64]
PILLARS = {
    7: np.array([1 / 12, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]),
    6: np.array([1 / 12, 0.25, 0.5, 1.0, 1.5, 2.0]),
    5: np.array([0.25, 0.5, 1.0, 1.5, 2.0]),
}
GAMMA_BOUNDS = (0.05, 1.0)
Params = tuple[FloatArray, FloatArray, FloatArray, float, float]
Case = tuple[FloatArray, FloatArray, FloatArray, float, float, CalendarRepair]
SEED = 20260916
N_CASES = 30
"""Draws of the module fixture used by the structural tests."""
N_PROPERTY = 500
"""Draws of the invariant test.  If the repair failed on a fraction p of draws, 500 independent
draws would all pass with probability (1 - p)^500: 7.3e-12 for p = 5%, 6.6e-3 for p = 1%."""
PROPERTY_CHUNK = 25
"""Draws per parametrised chunk (so ``pytest -n`` spreads the 500 draws)."""
MAX_MAT = 3.0
GAMMA_DRAW = (0.2, 0.6)
"""gamma of the synthetic cases (the design's range).  Above 1/2, theta*phi^2 diverges as
theta -> 0, so the unconstrained draw's eta may already fail the constructor's butterfly check on
[theta(1/365), theta(max_maturity)]; the repair caps eta on that range too, so its output must
construct regardless."""
DENSE_TOL = 1e-12
"""Round-off allowance when the dense grid's exact minimum is compared with the proven margin."""


class _Unchecked(ESSVISurface):
    def _check_calendar_numeric(self) -> None:
        return None


def _theta_range(theta_p: FloatArray) -> tuple[float, float]:
    return float(theta_p.min()) * 0.5, float(theta_p.max()) * 1.5


def _construct_range(pil: FloatArray, theta_p: FloatArray) -> tuple[float, float]:
    """The fit's theta range widened to the constructor's butterfly range (so a draw builds)."""
    lo, hi = _theta_range(theta_p)
    s = _Unchecked(
        pil, theta_p, np.full(pil.size, -0.5), 0.0, 0.5, _FC, _FC.rate_curve, max_maturity=MAX_MAT
    )
    return min(lo, float(s.theta(s.min_maturity))), max(hi, float(s.theta(MAX_MAT)))


def _vols(
    pil: FloatArray,
    theta_p: FloatArray,
    rp: FloatArray,
    g: float,
    e: float,
    k: FloatArray,
    T: FloatArray,
) -> FloatArray:
    """eSSVI implied vols at points inside the pillar range (where θ is interpolated)."""
    th = np.interp(T, np.concatenate(([0.0], pil)), np.concatenate(([0.0], theta_p)))
    r = np.interp(T, pil, rp)
    phi = e / (th**g * (1.0 + th) ** (1.0 - g))
    pk = phi * k
    w = 0.5 * th * (1.0 + r * pk + np.sqrt((pk + r) ** 2 + 1.0 - r * r))
    return np.asarray(np.sqrt(w / T), dtype=np.float64)


def _points(pil: FloatArray) -> tuple[FloatArray, FloatArray]:
    ts = np.unique(np.concatenate((pil, 0.5 * (pil[1:] + pil[:-1]))))
    ks = np.linspace(-0.25, 0.25, 11)
    K, T = np.meshgrid(ks, ts)
    return K.ravel(), T.ravel()


def _unchecked(
    pil: FloatArray, theta_p: FloatArray, rp: FloatArray, g: float, e: float
) -> ESSVISurface:
    return _Unchecked(
        pil, theta_p, rp, e, g, _FC, _FC.rate_curve, max_maturity=max(MAX_MAT, pil[-1])
    )


def _surface(pil: FloatArray, theta_p: FloatArray, rep: CalendarRepair) -> ESSVISurface:
    """The repaired surface through the CHECKED constructor."""
    return ESSVISurface(
        pil, theta_p, rep.rhos, rep.eta, rep.gamma, _FC, _FC.rate_curve, max_maturity=MAX_MAT
    )


def _min_dw_dt(
    pil: FloatArray,
    theta_p: FloatArray,
    rp: FloatArray,
    g: float,
    e: float,
    k_abs: float = 3.0,
    cfg: CalendarRepairConfig = DEFAULT_CALENDAR_REPAIR,
) -> float:
    """Smallest exact ``∂_T w`` on the repair's initial grid, evaluated on the surface object
    (its own segments, both limits at every knot)."""
    s = _unchecked(pil, theta_p, rp, g, e)
    ks, seg, ts = calendar_constraint_grid(
        pil, s.max_maturity, cfg, k_max=k_abs, n_k=cfg.n_k if k_abs == 3.0 else 81
    )
    D = s.calendar_segments().dw_dt(ks[None, :], seg[:, None], ts[:, None], s.eta, s.gamma)
    return float(np.min(D))


_FC = ForwardCurve.flat(100.0, 0.02, 0.01)


def _case(rng: np.random.Generator) -> Params:
    """A random eSSVI parameter set violating the calendar condition on ``±3``."""
    while True:
        P = int(rng.choice([5, 6, 7]))
        pil = PILLARS[P]
        fwd = rng.uniform(0.02, 0.10, P)
        theta_p = np.cumsum(fwd * np.diff(np.concatenate(([0.0], pil))))
        g = float(rng.uniform(*GAMMA_DRAW))
        rp = rng.uniform(-0.8, -0.4, P)
        i = int(rng.integers(max(P - 4, 0), P - 1))  # a pair in the long half
        c = rng.uniform(0.6, 0.95)
        rp[i + 1] = max(-0.98, -1.0 + theta_p[i] * (1.0 + rp[i]) / theta_p[i + 1] * c)
        lo, hi = _construct_range(pil, theta_p)
        e = float(rng.uniform(0.5, 0.95) * 0.999 * _eta_max(float(np.max(np.abs(rp))), g, lo, hi))
        if _min_dw_dt(pil, theta_p, rp, g, e) < 0.0:
            return pil, theta_p, rp, g, e


def _draw(i: int) -> Params:
    """Draw ``i`` of the seeded family: its own generator, so any subset is reproducible."""
    return _case(np.random.default_rng([SEED, i]))


def _market_resid(
    pil: FloatArray, theta_p: FloatArray, rp0: FloatArray, g0: float, e0: float
) -> Callable[[FloatArray, float, float], FloatArray]:
    k, T = _points(pil)
    target = _vols(pil, theta_p, rp0, g0, e0, k, T)

    def resid(rp: FloatArray, g: float, e: float) -> FloatArray:
        return 100.0 * (_vols(pil, theta_p, rp, g, e, k, T) - target)

    return resid


def _repair(
    pil: FloatArray,
    theta_p: FloatArray,
    rp: FloatArray,
    g: float,
    e: float,
    cfg: CalendarRepairConfig = DEFAULT_CALENDAR_REPAIR,
) -> CalendarRepair:
    return repair_calendar(
        pil,
        theta_p,
        rp,
        g,
        e,
        _market_resid(pil, theta_p, rp, g, e),
        gamma_bounds=GAMMA_BOUNDS,
        cfg=cfg,
        max_maturity=MAX_MAT,
    )


def _butterfly_ok(theta_p: FloatArray, rp: FloatArray, g: float, e: float) -> bool:
    lo, hi = _theta_range(theta_p)
    th = np.linspace(lo, hi, 2001)
    phi = e / (th**g * (1.0 + th) ** (1.0 - g))
    a = 1.0 + float(np.max(np.abs(rp)))
    return bool(np.all(th * phi * a < 4.0) and np.all(th * phi * phi * a <= 4.0 + 1e-12))


def _assert_invariant(pil: FloatArray, theta_p: FloatArray, params: Params, tag: str) -> None:
    """Repair one violating draw and check the invariant three ways on the CONSTRUCTED surface:
    the repair's own proof, an independent certificate of the surface object, and the exact
    ``∂_T w`` on the dense grid (1201 ``k`` on ``±3`` × 3000 ``T``) plus both one-sided limits
    at every knot.  The unrepaired input fails the same dense check."""
    cfg = DEFAULT_CALENDAR_REPAIR
    _, _, rp, g, e = params
    rep = _repair(pil, theta_p, rp, g, e)
    assert rep.repaired and rep.feasible and rep.fallback is None, (tag, rep)
    assert rep.floor == cfg.margin and rep.k_abs == cfg.k_max, tag
    assert rep.certificate is not None and rep.certificate.certified, tag
    assert rep.lower_bound >= cfg.margin, tag
    s = _surface(pil, theta_p, rep)
    assert s.calendar_certificate(cfg.k_max, cfg.margin).certified, tag
    d = s.calendar_dense_check()
    assert (d.n_k, d.n_t, d.k_abs) == (1201, 3000, 3.0)
    assert d.ok, (tag, d)
    assert d.min_interior >= cfg.margin - DENSE_TOL, (tag, d)
    assert d.min_left >= cfg.margin - DENSE_TOL, (tag, d)
    assert d.min_right >= cfg.margin - DENSE_TOL, (tag, d)
    knots = calendar_knots(s.min_maturity, s.pillars, s.max_maturity)[1:-1]
    ks = np.linspace(-3.0, 3.0, 1201)
    for side in ("left", "right"):
        lim = s.dw_dT(ks[None, :], knots[:, None], side)
        assert float(np.min(lim)) >= cfg.margin - DENSE_TOL, (tag, side)
    u = _unchecked(pil, theta_p, rp, g, e).calendar_dense_check()
    assert not u.ok and u.min_dw_dt < 0.0, tag


_VERIFIER_TABLE = """
test:20260916:41 7 0.4381779618263946 1.24605626424703 0.007046683065825437,0.023117736376448778,0.03034468924827713,0.07185350691532494,0.0988926781175856,0.11652452130573089,0.15065186178090564 -0.5200605663034177,-0.7689535755015555,-0.699664484334485,-0.4350785091900904,-0.7468765003600607,-0.5571932007129703,-0.5863892168478948
test:20260916:43 5 0.45049648649424834 1.2619339020306033 0.007298536441860255,0.02616252785393258,0.06024030650751186,0.091429823526291,0.14070321586401463 -0.49250230271156575,-0.7872496919823139,-0.7025085983786785,-0.835898992594523,-0.4359465859113129
test:20260916:52 7 0.3658552389520598 1.8536217359826597 0.0047681181363946916,0.011252660913422148,0.03132732748191863,0.04151023943405998,0.0722819347193203,0.11847169801810475,0.20240413213134933 -0.7763842776626402,-0.4125182870726829,-0.7284644500468421,-0.6679785632155387,-0.8758434892689586,-0.5862777833868332,-0.6015307618906984
test:20260916:63 7 0.5098130309182349 1.1630172976572108 0.0018168142880060879,0.015755455366016555,0.026705710804653282,0.05434539310592089,0.06536464336995772,0.10136198350466855,0.15530973749089233 -0.5488142718412471,-0.7774764085624152,-0.520940155025368,-0.6134367149494852,-0.7789778088881137,-0.5392312881853106,-0.48246033302542024
test:20260916:73 6 0.27531091324263846 1.5476661095228166 0.003035261769522876,0.015367994297994236,0.0392872236511472,0.058693998796189,0.07284058207321867,0.11248023597300551 -0.7541346757890465,-0.4106599953623518,-0.5351788033210684,-0.7762156103370392,-0.8418002980131741,-0.771034461825189
test:20260916:78 7 0.5317873638769429 1.0059349697161062 0.0035491770000683626,0.01913103332264501,0.034365003168422795,0.05674494085546153,0.07230746936839241,0.10777470312883262,0.17458161684493897 -0.4963514227740381,-0.6521608653430505,-0.6346110961020948,-0.4744044950685908,-0.754895815535416,-0.5900430963703476,-0.8450030870608457
test:20260916:100 7 0.40543113682034987 1.3689650031598741 0.0022137215999459897,0.015377768252978312,0.022389644654320924,0.06514706099456066,0.09093664629970119,0.12316332687469908,0.18837841219585566 -0.585830603638151,-0.7622228402467177,-0.6923894010465819,-0.5668593039189975,-0.7641758432520391,-0.43725904630053947,-0.7389363225704011
test:20260916:126 7 0.4683619969842224 1.405118978035671 0.007926682830413555,0.02054740315357622,0.028033278240728352,0.07437587804151187,0.08740779249390991,0.11908118067443463,0.17508010475716512 -0.4001006667261235,-0.7065012136519424,-0.6461748360280823,-0.7195584343643184,-0.7884663849496095,-0.8781010564546691,-0.4273327263275286
test:20260916:141 6 0.3756498189963542 1.779424739169822 0.006090419432611299,0.02173142917862534,0.03898262927838517,0.06043951997799216,0.10894411844662323,0.13811953477661496 -0.42002374603248344,-0.7330847929440132,-0.4405697285695359,-0.6901236131084523,-0.6381086918717627,-0.8274931396246014
test:20260916:170 5 0.2963830494233503 2.173740989601532 0.022868684927026445,0.031107022824763568,0.07534060124162767,0.08881869457971606,0.0990995240413667 -0.4884378997787383,-0.4075470461352049,-0.6622129541564363,-0.7618561993505881,-0.7077896512139175
test:777:55 6 0.40414244030208596 1.821762553241412 0.005328739079457506,0.016927715971158032,0.03381359391822155,0.059596082062888765,0.07854876875611687,0.11305799665878 -0.4169573732627599,-0.6257611459758037,-0.602288574887599,-0.4302146501431179,-0.6540452659923622,-0.5344145214928236
test:777:124 6 0.3010281045943283 1.8794005331082138 0.001703424567009816,0.014907706731721856,0.023860359449792756,0.06646196517713332,0.0843142906416251,0.10518275677002376 -0.42338454704815226,-0.7830845796443433,-0.5388827500491851,-0.4357746845981943,-0.6302628557230326,-0.47734455011236226
test:777:138 5 0.3468564017701704 1.5196197780586942 0.006763066935740378,0.026391326676638436,0.043480445985732794,0.07442527482148631,0.10997536380543756 -0.4683333705401176,-0.6656409752251776,-0.53664647190307,-0.4139881451892005,-0.6874200985870447
test:777:152 6 0.5402092259523316 0.8072707902741839 0.0034704318197482224,0.01996348350227104,0.03497171291028853,0.051386869939700056,0.07214653032567091,0.1098555435283549 -0.49244466832476147,-0.7410962576528217,-0.4548788277817733,-0.6995260676491953,-0.5546965213141961,-0.7240574355784158
wide:1:0 6 0.4387391795507032 1.0753901810879127 0.02378223253467924,0.03170341349339456,0.10291631021976629,0.1514114493913129,0.2163521006147678,0.340938233203283 -0.2720160166552144,-0.9349999261813031,-0.013038351982995595,-0.6583999365369961,-0.5512407200461529,0.03130445335407353
wide:1:2 6 0.8600011965566239 0.09914161288239211 0.022966901909443022,0.025746885012099662,0.06598034317752625,0.1362323859031233,0.14792894882772997,0.24502485377578775 -0.21696490700755927,-0.6396762413737265,0.0966495317098892,-0.32294023046768305,-0.7493014795037389,-0.013651636218234054
wide:1:4 6 0.5081017687893697 1.232737591649975 0.0073461632560449935,0.018761056474891194,0.06716174575480205,0.18840733364369097,0.33304878738832416,0.35775119987544785 0.16628914498914038,-0.4331495281792946,-0.22133238115325887,-0.9388968395834287,-0.11470594331576753,-0.282556326824126
wide:1:5 6 0.4596654113385359 0.8109584043497033 0.010770891683128482,0.016812019154571936,0.06475756517476808,0.12337014498303159,0.23285100084927682,0.331796237481865 0.13149704216559302,-0.1671884007747878,-0.7865276024778902,-0.5359207005960565,-0.27954000212106134,-0.7207029559042631
wide:1:6 5 0.4074265763566895 1.0493718098528029 0.006647767838833741,0.026910747864843064,0.1419722064150909,0.2474115080908454,0.2688908069076279 -0.43542982883582204,-0.1254700071203098,-0.8671766934271904,-0.2251217249315751,0.09639944658335685
wide:1:7 5 0.34739546587894077 2.181360594031542 0.028367843264637985,0.03771077473513187,0.07018890248802287,0.1145503596072822,0.16338510920862176 -0.2375913603587509,0.2640462690370885,-0.5165048702344516,0.034740114179462704,-0.005729004314733424
wide:1:8 7 0.33025612066524346 1.7316684887238283 0.012717095588948077,0.017340382840224565,0.054613513337781944,0.08850608515048125,0.11057878903863237,0.1877233651117149,0.4243235264278178 0.006340135084641449,-0.30245050558393727,-0.780709010319195,0.2555090348542644,-0.45992199566158276,-0.8772392336384249,0.10568793070483666
wide:1:9 7 0.5454488660257967 0.5708269812384574 0.010065422826975985,0.022300500184697793,0.08559106464576133,0.14562706105689094,0.2918942509570286,0.38662031990825685,0.5962390561456168 -0.5776103871739993,-0.4676433452634155,-0.8969922834618858,-0.7144719350796761,0.28503800454966965,-0.006951555255150538,-0.5130706033543849
wide:1:10 6 0.47064944556697025 1.2379294984871256 0.012801822525615517,0.014457361256021557,0.05210826380184702,0.197919029785263,0.2425251510311358,0.3553872999949361 -0.7042130758754125,0.17935326493902454,-0.7101442075631914,-0.5845436632186214,0.2987628705514007,-0.6370735688387134
wide:1:12 5 0.12145055622368421 1.200149200276417 0.05526161175276333,0.1008960910249575,0.14581939234886687,0.2637765613679767,0.3033385291482113 0.25283801462289934,-0.28418576952748414,0.012845757888260412,-0.29788703430944263,-0.8025594431322292
wide:1:14 5 0.16219996579331575 0.9538877245708626 0.059995275647046736,0.11077486365312361,0.2379157696594547,0.3788816643602377,0.3847177793710706 -0.5124648664517732,-0.8511446966358983,-0.20860391922233795,-0.639337437617902,-0.8227785799885765
wide:1:16 5 0.3722351481411649 0.761508378766469 0.07228465160955545,0.12909535119652188,0.18142884610646198,0.20342522524529336,0.26296798478814293 0.14054032583194098,-0.7569840335663923,-0.86580284236016,0.20704260699540453,-0.1796620280816196
wide:1:17 7 0.6867913144015532 0.20603040523209576 0.0026805204065241056,0.0345750031898264,0.08128330778307602,0.08851530670425369,0.21011016690784462,0.3286623941464206,0.6036783333879365 -0.09006496493426643,-0.7620480988444768,-0.9396612062469091,-0.9700196120219472,0.2548659020036539,-0.14993386573926534,0.2325714664466274
wide:1:18 6 0.5969199023221436 0.29286068943305693 0.00450254365416054,0.018961599285366715,0.060797570348296415,0.14551538016170565,0.22161588187699033,0.28672426061022177 0.25766124214255465,-0.7825844205358667,0.09358870568397418,-0.8990426044095235,-0.4803041833178268,-0.25816951657415305
wide:1:19 6 0.18093163705620136 0.9533341256600989 0.023697234981419044,0.056432397053673214,0.10082365787407993,0.11295521767575646,0.12315121567812075,0.15683248673335026 0.27936241017707536,-0.9665111225446165,-0.98,-0.895803488977772,-0.15718659093655066,-0.9109028333649289
"""
"""Every draw on which the kink-aware secant repair (second pass) left ``∂_T w < 0`` on the dense
check, as the M10 Part 0 verifier found them: ``test:<seed>:<n>`` is draw ``n`` of the previous
test generator with that seed (the fixture's own seed 20260916 fails at draws 41, 43, 52, 63, 73,
78, 100, 126, 141, 170; seed 777 at 55, 124, 138, 152), ``wide:1:<n>`` draw ``n`` of the
verifier's wider family (any pillar pair, gamma 0.05-1, rho up to 0.3).  The parameters are the
unrepaired input, frozen here because the generator's rejection rule has changed."""

Draw = tuple[str, int, FloatArray, FloatArray, float, float]


def _parse_table(text: str) -> list[Draw]:
    """``tag P gamma eta theta_1,..,theta_P rho_1,..,rho_P`` per line (floats in ``repr``)."""
    out: list[Draw] = []
    for line in text.strip().splitlines():
        tag, P, g, e, th, rh = line.split()
        theta = np.array([float(x) for x in th.split(",")])
        rho = np.array([float(x) for x in rh.split(",")])
        assert theta.size == rho.size == int(P)
        out.append((tag, int(P), theta, rho, float(g), float(e)))
    return out


VERIFIER_FAILING_DRAWS = _parse_table(_VERIFIER_TABLE)


@pytest.fixture(scope="module")
def repaired_cases() -> list[Case]:
    out = []
    for i in range(N_CASES):
        pil, theta_p, rp, g, e = _draw(i)
        out.append((pil, theta_p, rp, g, e, _repair(pil, theta_p, rp, g, e)))
    return out


@pytest.mark.parametrize("chunk", range(N_PROPERTY // PROPERTY_CHUNK))
def test_invariant_holds_on_seeded_draws(chunk: int) -> None:
    """THE property: on 500 seeded violating draws (25 per chunk) the repaired surface
    constructs and ``∂_T w ≥ margin`` holds on the whole Dupire range — proven by the repair,
    re-proven on the surface object, and seen on the exact derivative over the dense grid and at
    both one-sided limits of every knot (see ``N_PROPERTY`` for the detection power)."""
    for i in range(chunk * PROPERTY_CHUNK, (chunk + 1) * PROPERTY_CHUNK):
        params = _draw(i)
        _assert_invariant(params[0], params[1], params, f"draw {i}")


@pytest.mark.parametrize("case", VERIFIER_FAILING_DRAWS, ids=[c[0] for c in VERIFIER_FAILING_DRAWS])
def test_invariant_holds_on_the_verifier_failing_draws(case: Draw) -> None:
    """Regression: the 29 draws the second pass got wrong (a secant over an edge step of 1e-3
    of the segment does not bound the one-sided limit where ``w`` is concave in ``T``) now carry
    the invariant; their unrepaired input fails the dense check."""
    tag, P, theta_p, rp, g, e = case
    pil = PILLARS[P]
    _assert_invariant(pil, theta_p, (pil, theta_p, rp, g, e), tag)


def test_repair_is_feasible_and_constructs(repaired_cases: list[Case]) -> None:
    """Every violating case is repaired with the constraint points at ``margin + headroom`` and
    the invariant proven at ``margin``; the surface builds with the constructor's own calendar
    and butterfly checks; the surface's exact derivative on the same grid, and its constructor
    quantity, agree with the repair's records."""
    cfg = DEFAULT_CALENDAR_REPAIR
    Ps = set()
    for pil, theta_p, _rp, _g, _e, rep in repaired_cases:
        Ps.add(pil.size)
        assert rep.repaired and rep.feasible and rep.fallback is None, rep
        assert rep.min_dw_dt_before < 0.0
        assert rep.min_dw_dt_after >= cfg.margin + cfg.headroom - cfg.tol
        assert rep.lower_bound >= cfg.margin and rep.floor == cfg.margin
        assert rep.cost_after > rep.cost_before  # the input was the unconstrained optimum
        s = _surface(pil, theta_p, rep)
        got = _min_dw_dt(pil, theta_p, rep.rhos, rep.gamma, rep.eta)
        assert got == pytest.approx(rep.min_dw_dt_after, rel=1e-9, abs=1e-12)
        assert s.calendar_min_dw_dt() == pytest.approx(rep.min_dw_dt1_after, rel=1e-9, abs=1e-12)
        assert np.all(np.abs(rep.rhos) <= cfg.rho_bound)
        assert GAMMA_BOUNDS[0] <= rep.gamma <= GAMMA_BOUNDS[1]
    assert Ps == {5, 6, 7}


def test_repair_keeps_both_butterfly_conditions(repaired_cases: list[Case]) -> None:
    """η, the only capped unknown (``ρ_T`` and ``γ`` are free inside their boxes), is capped at
    the repaired ``max|ρ|``: ``θφ(1+max|ρ|) < 4`` and
    ``θφ²(1+max|ρ|) ≤ 4`` on the fit's θ range."""
    for _pil, theta_p, _rp, _g, _e, rep in repaired_cases:
        assert _butterfly_ok(theta_p, rep.rhos, rep.gamma, rep.eta)


def test_feasible_input_is_returned_bit_identical() -> None:
    """A single ρ at every pillar is calendar-free with ``∂_T w ≥ θ'(1−ρ²)(1−χ) > margin``: the
    certificate proves it, no stage runs and the parameters come back as given."""
    rng = np.random.default_rng(7)
    for P in (5, 6, 7):
        pil = PILLARS[P]
        theta_p = np.cumsum(rng.uniform(0.02, 0.1, P) * np.diff(np.concatenate(([0.0], pil))))
        rp = np.full(P, -0.65)
        g = 0.45
        lo, hi = _theta_range(theta_p)
        e = 0.8 * 0.999 * _eta_max(0.65, g, lo, hi)
        rep = _repair(pil, theta_p, rp, g, e)
        assert not rep.repaired and rep.feasible and rep.stages == 0 and rep.fallback is None
        assert rep.cuts == 0 and rep.certificate is not None and rep.certificate.certified
        assert rep.rhos is rp and rep.gamma == g and rep.eta == e
        assert rep.cost_after == rep.cost_before == 0.0


def test_repair_is_a_projection_not_a_clip(repaired_cases: list[Case]) -> None:
    """With the residual ``100 (ρ − ρ_target)`` (γ and η pinned the same way), the repaired ρ is
    no farther from the infeasible target than the naive shrink toward the mean ρ that just
    reaches the repair's point constraint (``margin + headroom`` on the grid) and whose
    invariant is proven — a point of the repair's feasible set."""
    cfg = DEFAULT_CALENDAR_REPAIR
    for pil, theta_p, rt, g0, e0, _ in repaired_cases[:6]:

        def resid(
            rp: FloatArray,
            g: float,
            e: float,
            rt: FloatArray = rt,
            g0: float = g0,
            e0: float = e0,
        ) -> FloatArray:
            return 100.0 * np.concatenate([rp - rt, [g - g0, e - e0]])

        rep = repair_calendar(
            pil,
            theta_p,
            rt,
            g0,
            e0,
            resid,
            gamma_bounds=GAMMA_BOUNDS,
            cfg=cfg,
            max_maturity=MAX_MAT,
        )
        assert rep.repaired and rep.feasible and rep.cuts == 0
        mean = float(np.mean(rt))
        target = cfg.margin + cfg.headroom
        lo_l, hi_l = 0.0, 1.0  # largest shrink factor λ with a feasible ρ = mean + λ(ρ_t − mean)
        for _ in range(40):
            lam = 0.5 * (lo_l + hi_l)
            if _min_dw_dt(pil, theta_p, mean + lam * (rt - mean), g0, e0) >= target:
                lo_l = lam
            else:
                hi_l = lam
        shrink = mean + lo_l * (rt - mean)
        assert (
            _unchecked(pil, theta_p, shrink, g0, e0).calendar_certificate(3.0, cfg.margin).certified
        )
        assert np.linalg.norm(rep.rhos - rt) <= np.linalg.norm(shrink - rt) + 1e-6, (
            np.linalg.norm(rep.rhos - rt),
            np.linalg.norm(shrink - rt),
        )


def test_saturated_rho_is_bounded() -> None:
    """The tanh fit can return ``ρ = −1 + 1e-12`` (2022-12-22); the repair keeps
    ``|ρ| ≤ rho_bound`` and returns a constructible, proven surface."""
    pil = PILLARS[7]
    theta_p = np.array([0.30, 0.29, 0.28, 0.27, 0.265, 0.26, 0.255]) ** 2 * pil
    rp = np.array([-0.6, -0.7, -0.8, -0.85, -0.85, -0.83, -1.0 + 1e-12])
    g = 0.4
    lo, hi = _theta_range(theta_p)
    e = 0.9 * 0.999 * _eta_max(1.0 - 1e-12, g, lo, hi)
    assert _min_dw_dt(pil, theta_p, rp, g, e) < 0.0
    rep = _repair(pil, theta_p, rp, g, e)
    assert rep.repaired and rep.feasible and rep.lower_bound >= DEFAULT_CALENDAR_REPAIR.margin
    assert np.max(np.abs(rep.rhos)) <= DEFAULT_CALENDAR_REPAIR.rho_bound
    assert _surface(pil, theta_p, rep).calendar_dense_check().ok


CUT_CASE: Params = (
    np.array([0.5740518730413097, 1.045464665322683]),
    np.array([0.00114455140258254, 0.0043552268780453025]),
    np.array([0.9744880662401275, 0.6318330219289771]),
    0.5429860796813059,
    0.2074009430943149,
)
"""A surface whose ``∂_T w`` dips below the margin only in the interior of the first segment,
near ``k = 0.05`` (``rho = 0.97``, so ``S`` nearly vanishes at ``x = -rho``), found by a seeded
search over random eSSVI parameters."""


def test_certificate_cuts_close_an_interior_dip() -> None:
    """The exchange step: on a constraint grid of the knots and ``k ∈ {−3, 0, 3}`` only, a
    point constraint accepts ``CUT_CASE`` (every grid value above ``margin + headroom``), yet
    ``∂_T w < margin`` between the points.  The certificate finds the dip, its points join the
    constraint set, and the result is proven — also on the dense grid."""
    pil, theta_p, rp, g, e = CUT_CASE
    coarse = replace(DEFAULT_CALENDAR_REPAIR, n_seg=2, n_k=3)
    grid_min = _min_dw_dt(pil, theta_p, rp, g, e, cfg=coarse)
    assert grid_min >= coarse.margin + coarse.headroom
    u = _unchecked(pil, theta_p, rp, g, e)
    assert u.calendar_dense_check().min_dw_dt < coarse.margin
    assert u.calendar_certificate(3.0, coarse.margin).status == "violated"
    rep = _repair(pil, theta_p, rp, g, e, coarse)
    assert rep.repaired and rep.fallback is None and rep.cuts >= 1, rep
    assert rep.lower_bound >= coarse.margin
    d = _surface(pil, theta_p, rep).calendar_dense_check()
    assert d.ok and d.min_dw_dt >= coarse.margin - DENSE_TOL, d


def test_constraint_grid_is_kink_aware() -> None:
    """The initial constraint set lists ``n_seg`` maturities per knot segment, both ends exactly
    on the knots, so every interior knot appears twice — as the right end of one segment (left
    limit) and the left end of the next (right limit) — and ``max_maturity`` last."""
    cfg = DEFAULT_CALENDAR_REPAIR
    for P, pil in PILLARS.items():
        ks, seg, ts = calendar_constraint_grid(pil, 3.0, cfg)
        assert ks.size == cfg.n_k and ks[0] == -cfg.k_max and ks[-1] == cfg.k_max
        knots = calendar_knots(1.0 / 365.0, pil, 3.0)
        assert knots[0] == 1.0 / 365.0 and knots[-1] == 3.0
        assert np.all(np.isin(pil[pil < 3.0], knots)), P
        assert seg.size == ts.size == cfg.n_seg * (knots.size - 1)
        for i in range(knots.size - 1):
            t_i = ts[seg == i]
            assert t_i.size == cfg.n_seg and np.all(np.diff(t_i) > 0), (P, i)
            assert t_i[0] == knots[i] and t_i[-1] == knots[i + 1], (P, i)
        assert ts[-1] == 3.0


def test_escalation_margin_then_ssvi() -> None:
    """Isotonic pooling makes ``θ_T`` flat between the last two pillars: there ``w`` moves only
    through ``ρ_T``, whose effect has the sign of ``k``, so no positive margin is feasible.  The
    repair escalates to zero margin (step i), which the flat single-ρ input already meets
    (``∂_T w = θ' g h / 2 > 0`` with ``θ' = 2e-10``: proven, not merely sampled), and records it.
    With no stage allowed, an input violating inside ``±1`` runs through all steps and ends in
    the SSVI fallback (step iii), returning the input parameters unchanged."""
    assert CALENDAR_FALLBACKS == ("margin0", "k_abs_1", "ssvi")
    pil = PILLARS[7]
    theta_p = np.array([0.30, 0.29, 0.28, 0.27, 0.265, 0.26, 0.255]) ** 2 * pil
    theta_p[-1] = theta_p[-2] + 1e-10  # pooled, then jittered as pillar_atm_variances does
    rp = np.full(7, -0.6)
    g, e = 0.4, 0.8
    cfg = CalendarRepairConfig(max_stages=1)
    rep = _repair(pil, theta_p, rp, g, e, cfg)
    assert rep.fallback == "margin0" and rep.feasible and not rep.repaired
    assert rep.stages == 1 and rep.rhos is rp and rep.floor == 0.0 and rep.k_abs == 3.0
    assert rep.certificate is not None and rep.certificate.certified
    assert 0.0 < rep.min_dw_dt_after < cfg.margin

    p6, th6 = pil[:6], theta_p[:6]
    bad = np.array([-0.5, -0.55, -0.6, -0.6, -0.3, -0.95])
    assert _min_dw_dt(p6, th6, bad, g, 0.5, k_abs=1.0) < 0.0
    rep = repair_calendar(
        p6,
        th6,
        bad,
        g,
        0.5,
        _market_resid(p6, th6, bad, g, 0.5),
        gamma_bounds=GAMMA_BOUNDS,
        cfg=CalendarRepairConfig(max_stages=0),
        max_maturity=3.0,
    )
    assert rep.fallback == "ssvi" and not rep.feasible and not rep.repaired
    assert rep.rhos is bad and rep.stages == 0 and rep.min_dw_dt1_after < 0.0
    assert rep.certificate is None and rep.floor is None and np.isnan(rep.lower_bound)
