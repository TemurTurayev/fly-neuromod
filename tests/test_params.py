import pytest

from flyneuromod.engine.params import LIFParams, PoissonDrive


def test_defaults_match_reference_model():
    """Defaults are the published Shiu et al. (2024) constants (SI units)."""
    p = LIFParams()
    assert p.v_rest == pytest.approx(-0.052)
    assert p.v_threshold == pytest.approx(-0.045)
    assert p.tau_membrane == pytest.approx(0.020)
    assert p.tau_synapse == pytest.approx(0.005)
    assert p.w_synapse == pytest.approx(0.275e-3)


def test_step_counts_are_derived_from_dt():
    p = LIFParams(dt=0.1e-3)
    assert p.delay_steps == 18  # 1.8 ms
    assert p.refractory_steps == 22  # 2.2 ms


def test_delay_is_at_least_one_step():
    p = LIFParams(dt=1e-3, t_delay=0.0)
    assert p.delay_steps == 1


def test_evolve_returns_new_instance_and_leaves_original_untouched():
    p = LIFParams()
    q = p.evolve(w_synapse=0.5e-3)
    assert q.w_synapse == pytest.approx(0.5e-3)
    assert p.w_synapse == pytest.approx(0.275e-3)
    assert p is not q


def test_frozen_dataclass_rejects_mutation():
    p = LIFParams()
    with pytest.raises(Exception):
        p.w_synapse = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau_membrane": 0.0},
        {"tau_membrane": 0.005},  # equal to tau_synapse -> singular propagator
        {"dt": 0.0},
        {"dt": 0.01},  # larger than tau_synapse
        {"t_refractory": -1e-3},
        {"v_threshold": -0.060},  # below rest
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LIFParams(**kwargs)


def test_poisson_weight_scales_with_synaptic_weight():
    p = LIFParams()
    drive = PoissonDrive(rate=150.0, weight_factor=250.0)
    assert drive.weight(p) == pytest.approx(250 * 0.275e-3)


def test_invalid_poisson_drive_is_rejected():
    with pytest.raises(ValueError):
        PoissonDrive(rate=-1.0)
    with pytest.raises(ValueError):
        PoissonDrive(weight_factor=0.0)
