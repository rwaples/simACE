"""Focused tests for simulation parameter validation edge cases."""

import pytest

from simace.simulation.params import CTransmission, SimulationParams


class TestSimulationParamsValidationEdgeCases:
    """Exercise validation errors not hit by end-to-end simulation tests."""

    @staticmethod
    def _base(**overrides):
        params = dict(
            seed=1,
            N=10,
            G_ped=2,
            G_sim=2,
            mating_lambda=0.5,
            p_mztwin=0.0,
            A1=0.5,
            C1=0.0,
            A2=0.5,
            C2=0.0,
            E1=0.5,
            E2=0.5,
            rA=0.0,
            rC=0.0,
            rE=0.0,
        )
        params.update(overrides)
        return params

    def test_invalid_g_ped_rejected(self):
        with pytest.raises(ValueError, match="G_ped must be an integer"):
            SimulationParams.create(**self._base(G_ped=0, G_sim=0))

    def test_invalid_r_c_rejected(self):
        with pytest.raises(ValueError, match=r"rC must be in \[-1, 1\]"):
            SimulationParams.create(**self._base(rC=1.5))

    def test_invalid_per_generation_assort_value_rejected(self):
        with pytest.raises(ValueError, match=r"assort1\[1\] must be in \[-1, 1\]"):
            SimulationParams.create(**self._base(assort1={0: 0.0, 1: 1.5}))

    def test_assort_matrix_second_diagonal_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"assort_matrix\[1,1\]"):
            SimulationParams.create(**self._base(assort_matrix=[[0.2, 0.0], [0.0, -1.5]]))

    def test_c_transmission_is_validated(self):
        with pytest.raises(ValueError, match="exceeds 1"):
            SimulationParams.create(
                **self._base(C1=0.2, c_transmission=CTransmission(maternal=(0.8, 0), paternal=(0.8, 0)))
            )


class TestCTransmissionValidate:
    """The structural check on the configured C covariance (ADR 0025)."""

    def test_constant_c_under_full_copying_passes(self):
        CTransmission(maternal=(1.0, 1.0)).validate(0.2, 0.3, 0.5)

    def test_unequal_coefficients_under_full_c_correlation_are_rejected(self):
        with pytest.raises(ValueError, match=r"infeasible: .*eigenvalue -0\.005"):
            CTransmission(maternal=(0.1, 0.0)).validate(1.0, 1.0, 1.0)

    def test_from_params_defaults_missing_keys_to_zero(self):
        assert CTransmission.from_params({"Ctm_paternal2": 0.25}) == CTransmission(paternal=(0.0, 0.25))
        assert not CTransmission.from_params({}).active
