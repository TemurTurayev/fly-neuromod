"""Clearance is transporter uptake plus diffusion out of the compartment."""

import numpy as np
import pytest

from flyneuromod.neuromod.autoreceptor import AutoreceptorFeedback
from flyneuromod.neuromod.constants import DAN_AUTORECEPTOR_FEEDBACK, MB_COMPARTMENT_RELEASE
from flyneuromod.neuromod.field import DopamineField, ReleaseKinetics


def test_transporter_alone_saturates_and_accumulates():
    """Without diffusion, driving a field harder than V_max piles dopamine up."""
    kinetics = ReleaseKinetics(per_spike=0.01, v_max=0.3, k_m=1.3, k_diffusion=0.0)
    field = DopamineField(n_fields=1, kinetics=kinetics, dt=1e-3)
    release = np.array([1.0])  # 1 spike per ms = 10 uM/s, far above v_max
    for _ in range(5_000):
        field.step(release)
    assert field.concentration[0] > 10.0


def test_diffusion_bounds_the_concentration():
    """With a first-order escape term the same drive reaches a steady state."""
    kinetics = ReleaseKinetics(per_spike=0.01, v_max=0.3, k_m=1.3, k_diffusion=0.12)
    field = DopamineField(n_fields=1, kinetics=kinetics, dt=1e-3)
    release = np.array([1.0])
    for _ in range(200_000):
        field.step(release)
    steady = field.concentration[0]
    assert kinetics.clearance_at(steady) == pytest.approx(10.0, rel=0.02)


def test_default_kinetics_include_diffusion():
    assert MB_COMPARTMENT_RELEASE.k_diffusion > 0


def test_physiological_drive_stays_in_the_measured_range():
    """Two dopaminergic neurons at 20 Hz hold the compartment near the measured peak."""

    auto = AutoreceptorFeedback(DAN_AUTORECEPTOR_FEEDBACK, n_fields=1, dt=1e-3)
    field = DopamineField(n_fields=1, kinetics=MB_COMPARTMENT_RELEASE, dt=1e-3)
    release = np.array([2 * 20.0 * 1e-3])
    gain = np.ones(1)
    for _ in range(30_000):
        c = field.step(release, gain=gain)
        gain = auto.step(c)
    assert 0.3 <= field.concentration[0] <= 1.0


def test_transporter_block_leaves_diffusion_working():
    """Blocking uptake slows clearance but cannot abolish it, as in fumin flies."""
    blocked = MB_COMPARTMENT_RELEASE.evolve(k_m=MB_COMPARTMENT_RELEASE.k_m * 20)
    field = DopamineField(n_fields=1, kinetics=blocked, dt=1e-3)
    field.concentration[0] = 0.4
    for _ in range(20_000):
        field.step(np.zeros(1))
    assert 0.0 < field.concentration[0] < 0.1


def test_negative_diffusion_is_rejected():
    with pytest.raises(ValueError):
        ReleaseKinetics(per_spike=0.01, v_max=0.3, k_m=1.3, k_diffusion=-1.0)


def test_baseline_is_held_against_total_clearance():
    kinetics = ReleaseKinetics(
        per_spike=0.01, v_max=0.3, k_m=1.3, k_diffusion=0.12, baseline=0.05
    )
    field = DopamineField(n_fields=1, kinetics=kinetics, dt=1e-3)
    for _ in range(10_000):
        field.step(np.zeros(1))
    assert field.concentration[0] == pytest.approx(0.05, rel=1e-3)
