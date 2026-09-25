"""Presynaptic Dop2R autoreceptor feedback loop tests."""

import numpy as np
import pytest

from flyneuromod.neuromod import pharmacology
from flyneuromod.neuromod.autoreceptor import AutoreceptorFeedback, AutoreceptorParams
from flyneuromod.neuromod.constants import (
    DAN_AUTORECEPTOR,
    DAN_AUTORECEPTOR_FEEDBACK,
    MB_COMPARTMENT_RELEASE,
)
from flyneuromod.neuromod.field import DopamineField
from flyneuromod.neuromod.plasticity import PlasticityParams, SynapticPlasticity


def test_gain_at_zero_dopamine_and_monotonic_decrease():
    """Gain is 1 at zero dopamine and decreases monotonically with concentration."""
    params = DAN_AUTORECEPTOR_FEEDBACK
    auto = AutoreceptorFeedback(params, n_fields=1, dt=1e-3)

    # zero dopamine -> gain == 1
    gain0 = auto.step(np.zeros(1))
    assert gain0[0] == pytest.approx(1.0)

    # monotonically decreasing with concentration
    concentrations = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    gains = []
    for c in concentrations:
        auto.reset()
        for _ in range(10_000):
            g = auto.step(np.array([c]))
        gains.append(g[0])

    for i in range(len(gains) - 1):
        assert gains[i] > gains[i + 1]


def test_autoreceptor_lowers_steady_state_at_20hz_by_3_to_5_fold():
    """With autoreceptor, 20 Hz steady-state is 3-5x lower than with autoreceptor=None."""
    kinetics = MB_COMPARTMENT_RELEASE
    dt = 1e-3

    # With autoreceptor
    params = DAN_AUTORECEPTOR_FEEDBACK
    auto = AutoreceptorFeedback(params, n_fields=1, dt=dt)
    field_with = DopamineField(n_fields=1, kinetics=kinetics, dt=dt)

    gain = np.ones(1)
    for _ in range(20_000):
        c = field_with.step(np.array([20.0 * dt]), gain=gain)
        gain = auto.step(c)
    c_with = field_with.concentration[0]

    # Without autoreceptor
    field_without = DopamineField(n_fields=1, kinetics=kinetics, dt=dt)
    for _ in range(20_000):
        field_without.step(np.array([20.0 * dt]))
    c_without = field_without.concentration[0]

    # Concentration with autoreceptor is in 0.3 - 0.5 uM
    assert 0.3 <= c_with <= 0.5
    # Suppression ratio is between 3x and 5x
    ratio = c_without / c_with
    assert 3.0 <= ratio <= 5.0


def test_dop2r_block_restores_unsuppressed_level():
    """Blocking Dop2R with flupentixol restores the unsuppressed concentration."""
    kinetics = MB_COMPARTMENT_RELEASE
    dt = 1e-3

    # Dop2R blocked (flupentixol)
    params = DAN_AUTORECEPTOR_FEEDBACK
    auto_blocked = AutoreceptorFeedback(params, n_fields=1, dt=dt)
    ratio = pharmacology.FLUPENTIXOL.blocks["Dop2R"]
    auto_blocked.population.set_competitive_antagonist(concentration=ratio, k_i=1.0)
    field_blocked = DopamineField(n_fields=1, kinetics=kinetics, dt=dt)

    gain = np.ones(1)
    for _ in range(20_000):
        c = field_blocked.step(np.array([20.0 * dt]), gain=gain)
        gain = auto_blocked.step(c)
    c_blocked = field_blocked.concentration[0]

    # Unsuppressed (autoreceptor=None)
    field_none = DopamineField(n_fields=1, kinetics=kinetics, dt=dt)
    for _ in range(20_000):
        field_none.step(np.array([20.0 * dt]))
    c_none = field_none.concentration[0]

    assert c_blocked == pytest.approx(c_none, rel=0.05)


@pytest.mark.parametrize("coarse_dt", [2e-3, 5e-3])
def test_learned_weight_change_is_dt_invariant(coarse_dt):
    """Plasticity step is dt-invariant under integration."""
    def run_plasticity(dt):
        weights = np.array([10.0])
        params = PlasticityParams(tau_eligibility=2.0, rate_depression=2.0, rate_potentiation=0.8)
        plasticity = SynapticPlasticity(
            weights=weights,
            synapse_index=np.array([0]),
            presynaptic_index=np.array([0]),
            compartment_index=np.array([0]),
            n_compartments=1,
            params=params,
            dt=dt,
        )
        for _ in range(int(round(0.2 / dt))):
            plasticity.step(np.array([1.0]), gs_activation=np.zeros(1), ip3=np.zeros(1))
        for _ in range(int(round(0.5 / dt))):
            plasticity.step(np.zeros(1), gs_activation=np.array([0.8]), ip3=np.zeros(1))
        return weights[0]

    fine_w = run_plasticity(1e-3)
    coarse_w = run_plasticity(coarse_dt)
    assert coarse_w == pytest.approx(fine_w, rel=0.02)


def test_autoreceptor_params_validation():
    """AutoreceptorParams validates max_suppression and min_gain range."""
    with pytest.raises(ValueError):
        AutoreceptorParams(spec=DAN_AUTORECEPTOR, max_suppression=-0.1, min_gain=0.1)
    with pytest.raises(ValueError):
        AutoreceptorParams(spec=DAN_AUTORECEPTOR, max_suppression=1.1, min_gain=0.1)
    with pytest.raises(ValueError):
        AutoreceptorParams(spec=DAN_AUTORECEPTOR, max_suppression=0.5, min_gain=0.0)
    with pytest.raises(ValueError):
        AutoreceptorParams(spec=DAN_AUTORECEPTOR, max_suppression=0.5, min_gain=1.1)
