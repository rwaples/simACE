"""Unit tests for computation functions in simace.stats.

Each test builds a small DataFrame with known values so expected outputs
can be verified by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simace.analysis.stats import (
    compute_affected_correlations,
    compute_censoring_cascade,
    compute_censoring_confusion,
    compute_censoring_windows,
    compute_cross_trait_tetrachoric,
    compute_cumulative_incidence,
    compute_cumulative_incidence_by_sex,
    compute_cumulative_incidence_by_sex_generation,
    compute_joint_affection,
    compute_liability_correlations,
    compute_mate_correlation,
    compute_mean_family_size,
    compute_mortality,
    compute_observed_h2_estimators,
    compute_parent_offspring_affected_corr,
    compute_parent_offspring_corr,
    compute_parent_offspring_corr_by_sex,
    compute_parent_status,
    compute_person_years,
    compute_prevalence,
    compute_regression,
    compute_tetrachoric,
    compute_tetrachoric_by_generation,
    compute_tetrachoric_by_sex,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def person_years_df():
    """5 individuals across 2 generations with known death_ages and t_observed.

    Generation 1 (window [20, 80]):
      id=0: death_age=70, t_observed1=50, t_observed2=60
      id=1: death_age=90, t_observed1=85, t_observed2=40

    Generation 2 (window [0, 60]):
      id=2: death_age=55, t_observed1=30, t_observed2=50
      id=3: death_age=40, t_observed1=45, t_observed2=20
      id=4: death_age=100, t_observed1=10, t_observed2=70
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "generation": [1, 1, 2, 2, 2],
            "death_age": [70.0, 90.0, 55.0, 40.0, 100.0],
            "t_observed1": [50.0, 85.0, 30.0, 45.0, 10.0],
            "t_observed2": [60.0, 40.0, 50.0, 20.0, 70.0],
        }
    )


@pytest.fixture
def gen_censoring_basic():
    """Windows: gen 1 -> [20, 80], gen 2 -> [0, 60]."""
    return {1: [20.0, 80.0], 2: [0.0, 60.0]}


@pytest.fixture
def family_df():
    """2 families of size 2 and 1 family of size 3.

    Family A (mother=100, father=101): children 1, 2
    Family B (mother=102, father=103): children 3, 4
    Family C (mother=104, father=105): children 5, 6, 7

    Plus 6 founders (ids 100-105) with mother=-1, father=-1.
    """
    founders = pd.DataFrame(
        {
            "id": [100, 101, 102, 103, 104, 105],
            "mother": [-1, -1, -1, -1, -1, -1],
            "father": [-1, -1, -1, -1, -1, -1],
            "sex": [0, 1, 0, 1, 0, 1],
        }
    )
    children = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "mother": [100, 100, 102, 102, 104, 104, 104],
            "father": [101, 101, 103, 103, 105, 105, 105],
            "sex": [0, 1, 0, 1, 0, 1, 0],
        }
    )
    return pd.concat([founders, children], ignore_index=True)


@pytest.fixture
def confusion_df():
    """6 individuals for censoring confusion tests.

    censor_age = 80, all in generation 1, window [20, 80].

    id=0: t1=50 (true aff), affected1=True  (TP); t2=90 (not aff), affected2=False (TN)
    id=1: t1=30 (true aff), affected1=False (FN); t2=40 (true aff), affected2=True  (TP)
    id=2: t1=70 (true aff), affected1=True  (TP); t2=60 (true aff), affected2=True  (TP)
    id=3: t1=85 (not aff), affected1=False (TN); t2=75 (true aff), affected2=False (FN)
    id=4: t1=20 (true aff), affected1=True  (TP); t2=100(not aff), affected2=False (TN)
    id=5: t1=95 (not aff), affected1=True  (FP); t2=50 (true aff), affected2=True  (TP)
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "generation": [1, 1, 1, 1, 1, 1],
            "t1": [50.0, 30.0, 70.0, 85.0, 20.0, 95.0],
            "t2": [90.0, 40.0, 60.0, 75.0, 100.0, 50.0],
            "affected1": [True, False, True, False, True, True],
            "affected2": [False, True, True, False, False, True],
        }
    )


@pytest.fixture
def cascade_df():
    """8 individuals spanning 2 generations for censoring cascade tests.

    censor_age = 80
    Generation 1 window [20, 70]:
      id=0: t1=50, death_age=60  -> true_aff, in_window, death < t? no (60>=50) -> observed
      id=1: t1=30, death_age=25  -> true_aff, in_window, death < t? yes (25<30) -> death_censored
      id=2: t1=10, death_age=90  -> true_aff, t < lo(20) -> left_truncated
      id=3: t1=75, death_age=90  -> true_aff, t > hi(70) -> right_censored

    Generation 2 window [0, 60]:
      id=4: t1=40, death_age=50  -> true_aff, in_window, death >= t -> observed
      id=5: t1=55, death_age=45  -> true_aff, in_window, death < t -> death_censored
      id=6: t1=90, death_age=100 -> NOT true_aff (90 >= 80)
      id=7: t1=65, death_age=80  -> true_aff, t > hi(60) -> right_censored
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "generation": [1, 1, 1, 1, 2, 2, 2, 2],
            "death_age": [60.0, 25.0, 90.0, 90.0, 50.0, 45.0, 100.0, 80.0],
            "t1": [50.0, 30.0, 10.0, 75.0, 40.0, 55.0, 90.0, 65.0],
            "t2": [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
            "affected1": [True, False, False, False, True, False, False, False],
            "affected2": [False, False, False, False, False, False, False, False],
        }
    )


# ===================================================================
# Tests for compute_person_years
# ===================================================================


class TestComputePersonYears:
    def test_basic(self, person_years_df, gen_censoring_basic):
        """Verify exact person-year totals for known data.

        Gen 1 window [20, 80]:
          id=0: end = min(70, 80) = 70, total = 70 - 20 = 50
                trait1: min(50, 70, 80) - 20 = 30
                trait2: min(60, 70, 80) - 20 = 40
                death: 70 in [20, 80) -> yes
          id=1: end = min(90, 80) = 80, total = 80 - 20 = 60
                trait1: min(85, 90, 80) - 20 = 60
                trait2: min(40, 90, 80) - 20 = 20
                death: 90 not in [20, 80) -> no

        Gen 2 window [0, 60]:
          id=2: end = min(55, 60) = 55, total = 55 - 0 = 55
                trait1: min(30, 55, 60) - 0 = 30
                trait2: min(50, 55, 60) - 0 = 50
                death: 55 in [0, 60) -> yes
          id=3: end = min(40, 60) = 40, total = 40 - 0 = 40
                trait1: min(45, 40, 60) - 0 = 40
                trait2: min(20, 40, 60) - 0 = 20
                death: 40 in [0, 60) -> yes
          id=4: end = min(100, 60) = 60, total = 60 - 0 = 60
                trait1: min(10, 100, 60) - 0 = 10
                trait2: min(70, 100, 60) - 0 = 60
                death: 100 not in [0, 60) -> no

        Totals:
          total = 50 + 60 + 55 + 40 + 60 = 265
          deaths = 1 + 0 + 1 + 1 + 0 = 3
          trait1 = 30 + 60 + 30 + 40 + 10 = 170
          trait2 = 40 + 20 + 50 + 20 + 60 = 190
        """
        result = compute_person_years(person_years_df, censor_age=100.0, gen_censoring=gen_censoring_basic)
        assert result["total"] == pytest.approx(265.0, abs=0.1)
        assert result["deaths"] == 3
        assert result["trait1"] == pytest.approx(170.0, abs=0.1)
        assert result["trait2"] == pytest.approx(190.0, abs=0.1)

    def test_no_death_age(self, gen_censoring_basic):
        """Without death_age column, follow-up = hi - lo for each person.

        Gen 1 window [20, 80]: 2 people * 60 = 120
        Gen 2 window [0, 60]:  3 people * 60 = 180
        Total = 300, deaths = 0
        """
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "generation": [1, 1, 2, 2, 2],
                "t_observed1": [50.0, 85.0, 30.0, 45.0, 10.0],
                "t_observed2": [60.0, 40.0, 50.0, 20.0, 70.0],
            }
        )
        result = compute_person_years(df, censor_age=100.0, gen_censoring=gen_censoring_basic)
        assert result["total"] == pytest.approx(300.0, abs=0.1)
        assert result["deaths"] == 0
        # trait1: min(50,80)-20 + min(85,80)-20 + min(30,60) + min(45,60) + min(10,60)
        #       = 30 + 60 + 30 + 45 + 10 = 175
        assert result["trait1"] == pytest.approx(175.0, abs=0.1)
        # trait2: min(60,80)-20 + min(40,80)-20 + min(50,60) + min(20,60) + min(70,60)
        #       = 40 + 20 + 50 + 20 + 60 = 190
        assert result["trait2"] == pytest.approx(190.0, abs=0.1)

    def test_window_hi_le_lo(self):
        """Generation with hi <= lo is skipped entirely."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "generation": [1, 1],
                "death_age": [50.0, 60.0],
                "t_observed1": [30.0, 40.0],
                "t_observed2": [35.0, 45.0],
            }
        )
        # hi <= lo -> generation 1 skipped
        gen_censoring = {1: [80.0, 20.0]}
        result = compute_person_years(df, censor_age=100.0, gen_censoring=gen_censoring)
        assert result["total"] == pytest.approx(0.0, abs=0.1)
        assert result["deaths"] == 0
        assert result["trait1"] == pytest.approx(0.0, abs=0.1)
        assert result["trait2"] == pytest.approx(0.0, abs=0.1)

    def test_missing_generation_column(self):
        """Returns {} when generation column is absent."""
        df = pd.DataFrame({"id": [0, 1], "death_age": [50.0, 60.0]})
        result = compute_person_years(df, censor_age=100.0, gen_censoring={1: [0.0, 80.0]})
        assert result == {}

    def test_trait_column_missing(self):
        """Trait key still present but 0.0 when t_observed columns are absent."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "generation": [1, 1],
                "death_age": [50.0, 60.0],
            }
        )
        gen_censoring = {1: [0.0, 80.0]}
        result = compute_person_years(df, censor_age=100.0, gen_censoring=gen_censoring)
        # total: min(50,80)-0 + min(60,80)-0 = 50 + 60 = 110
        assert result["total"] == pytest.approx(110.0, abs=0.1)
        # t_observed columns are missing -> trait person-years stay at 0
        assert result["trait1"] == pytest.approx(0.0, abs=0.1)
        assert result["trait2"] == pytest.approx(0.0, abs=0.1)

    def test_no_gen_censoring_uses_default(self):
        """When gen_censoring is None, default window is [0, censor_age]."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "generation": [1, 1],
                "death_age": [50.0, 90.0],
                "t_observed1": [30.0, 70.0],
                "t_observed2": [40.0, 80.0],
            }
        )
        censor_age = 80.0
        result = compute_person_years(df, censor_age=censor_age, gen_censoring=None)
        # Window defaults to [0, 80]:
        # id=0: total = min(50,80) = 50; trait1 = min(30,50,80) = 30; trait2 = min(40,50,80) = 40
        #        death: 50 in [0,80) -> yes
        # id=1: total = min(90,80) = 80; trait1 = min(70,90,80) = 70; trait2 = min(80,90,80) = 80
        #        death: 90 not in [0,80) -> no
        assert result["total"] == pytest.approx(130.0, abs=0.1)
        assert result["deaths"] == 1
        assert result["trait1"] == pytest.approx(100.0, abs=0.1)
        assert result["trait2"] == pytest.approx(120.0, abs=0.1)


# ===================================================================
# Tests for compute_mean_family_size
# ===================================================================


class TestComputeMeanFamilySize:
    def test_basic(self, family_df):
        """2 families of size 2, 1 family of size 3 -> mean = 7/3, median = 2.

        Family sizes: [2, 2, 3]
        mean = 7/3 ~ 2.33
        median = 2.0
        q1 = 2.0
        q3 = 2.5 (numpy interpolation of [2, 2, 3])
        n_families = 3
        frac_with_full_sib: all 7 children are in families >= 2 -> 7/7 = 1.0
        size_dist: {1: 0/3=0, 2: 2/3, 3: 1/3, 4+: 0/3=0}
        """
        result = compute_mean_family_size(family_df)
        assert result["mean"] == pytest.approx(7 / 3, abs=0.01)
        assert result["median"] == pytest.approx(2.0, abs=0.1)
        assert result["n_families"] == 3
        assert result["frac_with_full_sib"] == pytest.approx(1.0, abs=0.0001)
        assert result["size_dist"]["1"] == pytest.approx(0.0, abs=0.0001)
        assert result["size_dist"]["2"] == pytest.approx(2 / 3, abs=0.0001)
        assert result["size_dist"]["3"] == pytest.approx(1 / 3, abs=0.0001)
        assert result["size_dist"]["4+"] == pytest.approx(0.0, abs=0.0001)

    def test_all_founders(self):
        """All individuals are founders (mother=-1) -> returns {}."""
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "mother": [-1, -1, -1],
                "father": [-1, -1, -1],
                "sex": [0, 1, 0],
            }
        )
        assert compute_mean_family_size(df) == {}

    def test_missing_columns(self):
        """Missing mother/father columns -> returns {}."""
        df = pd.DataFrame({"id": [0, 1], "sex": [0, 1]})
        assert compute_mean_family_size(df) == {}

    def test_single_child_family(self):
        """One family of size 1: frac_with_full_sib = 0."""
        df = pd.DataFrame(
            {
                "id": [100, 101, 1],
                "mother": [-1, -1, 100],
                "father": [-1, -1, 101],
                "sex": [0, 1, 0],
            }
        )
        result = compute_mean_family_size(df)
        assert result["mean"] == pytest.approx(1.0, abs=0.01)
        assert result["n_families"] == 1
        # child 1 is in a family of size 1 -> not in families >= 2
        assert result["frac_with_full_sib"] == pytest.approx(0.0, abs=0.0001)
        assert result["size_dist"]["1"] == pytest.approx(1.0, abs=0.0001)

    def test_mates_by_sex(self, family_df):
        """Each mother and father has exactly 1 mate."""
        result = compute_mean_family_size(family_df)
        assert result["mates_by_sex"]["female_mean"] == pytest.approx(1.0, abs=0.01)
        assert result["mates_by_sex"]["male_mean"] == pytest.approx(1.0, abs=0.01)
        assert result["mates_by_sex"]["female_1"] == pytest.approx(1.0, abs=0.0001)
        assert result["mates_by_sex"]["male_1"] == pytest.approx(1.0, abs=0.0001)


# ===================================================================
# Tests for compute_censoring_confusion
# ===================================================================


class TestComputeCensoringConfusion:
    def test_known_confusion(self, confusion_df):
        """Verify 2x2 confusion for each trait.

        censor_age = 80
        Trait 1:
          id=0: t1=50 < 80 (true_aff=T), affected1=T  -> TP
          id=1: t1=30 < 80 (true_aff=T), affected1=F  -> FN
          id=2: t1=70 < 80 (true_aff=T), affected1=T  -> TP
          id=3: t1=85 >= 80 (true_aff=F), affected1=F -> TN
          id=4: t1=20 < 80 (true_aff=T), affected1=T  -> TP
          id=5: t1=95 >= 80 (true_aff=F), affected1=T -> FP
          => TP=3, FN=1, FP=1, TN=1, n=6

        Trait 2:
          id=0: t2=90 >= 80 (true_aff=F), affected2=F -> TN
          id=1: t2=40 < 80 (true_aff=T), affected2=T  -> TP
          id=2: t2=60 < 80 (true_aff=T), affected2=T  -> TP
          id=3: t2=75 < 80 (true_aff=T), affected2=F  -> FN
          id=4: t2=100 >= 80 (true_aff=F), affected2=F -> TN
          id=5: t2=50 < 80 (true_aff=T), affected2=T  -> TP
          => TP=3, FN=1, FP=0, TN=2, n=6
        """
        gen_censoring = {1: [20.0, 80.0]}
        result = compute_censoring_confusion(confusion_df, censor_age=80.0, gen_censoring=gen_censoring)

        assert result["trait1"]["tp"] == 3
        assert result["trait1"]["fn"] == 1
        assert result["trait1"]["fp"] == 1
        assert result["trait1"]["tn"] == 1
        assert result["trait1"]["n"] == 6

        assert result["trait2"]["tp"] == 3
        assert result["trait2"]["fn"] == 1
        assert result["trait2"]["fp"] == 0
        assert result["trait2"]["tn"] == 2
        assert result["trait2"]["n"] == 6

    def test_all_observed_correctly(self):
        """No censoring effect: all truly affected are observed, no FPs or FNs.

        censor_age = 100
          id=0: t1=30 < 100 (true_aff), affected1=True  -> TP
          id=1: t1=40 < 100 (true_aff), affected1=True  -> TP
          id=2: t1=150 >= 100 (not aff), affected2=False -> TN
        """
        df = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "generation": [1, 1, 1],
                "t1": [30.0, 40.0, 150.0],
                "t2": [200.0, 200.0, 200.0],
                "affected1": [True, True, False],
                "affected2": [False, False, False],
            }
        )
        gen_censoring = {1: [0.0, 100.0]}
        result = compute_censoring_confusion(df, censor_age=100.0, gen_censoring=gen_censoring)
        assert result["trait1"]["tp"] == 2
        assert result["trait1"]["fn"] == 0
        assert result["trait1"]["fp"] == 0
        assert result["trait1"]["tn"] == 1

    def test_inactive_generation_excluded(self):
        """Generations with hi <= lo are excluded from the confusion matrix."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "generation": [1, 2],
                "t1": [30.0, 30.0],
                "t2": [200.0, 200.0],
                "affected1": [True, True],
                "affected2": [False, False],
            }
        )
        # Gen 1: active (hi > lo), gen 2: inactive (hi <= lo)
        gen_censoring = {1: [0.0, 80.0], 2: [80.0, 20.0]}
        result = compute_censoring_confusion(df, censor_age=80.0, gen_censoring=gen_censoring)
        # Only gen 1 individual included
        assert result["trait1"]["n"] == 1
        assert result["trait1"]["tp"] == 1


# ===================================================================
# Tests for compute_censoring_cascade
# ===================================================================


class TestComputeCensoringCascade:
    def test_known_fates(self, cascade_df):
        """Verify per-generation decomposition of true cases.

        censor_age = 80
        Gen 1, window [20, 70]:
          id=0: t1=50 < 80 -> true_aff, t in [20,70], death=60 >= 50 -> observed
          id=1: t1=30 < 80 -> true_aff, t in [20,70], death=25 < 30  -> death_censored
          id=2: t1=10 < 80 -> true_aff, t < 20                       -> left_truncated
          id=3: t1=75 < 80 -> true_aff, t > 70                       -> right_censored

          true_affected=4, observed=1, death_censored=1, left_truncated=1, right_censored=1
          sensitivity = 1/4 = 0.25

        Gen 2, window [0, 60]:
          id=4: t1=40 < 80 -> true_aff, t in [0,60], death=50 >= 40  -> observed
          id=5: t1=55 < 80 -> true_aff, t in [0,60], death=45 < 55   -> death_censored
          id=6: t1=90 >= 80 -> NOT true_aff
          id=7: t1=65 < 80 -> true_aff, t > 60                       -> right_censored

          true_affected=3, observed=1, death_censored=1, left_truncated=0, right_censored=1
          sensitivity = 1/3 ~ 0.333
        """
        gen_censoring = {1: [20.0, 70.0], 2: [0.0, 60.0]}
        result = compute_censoring_cascade(cascade_df, censor_age=80.0, gen_censoring=gen_censoring)

        g1 = result["trait1"]["gen1"]
        assert g1["true_affected"] == 4
        assert g1["observed"] == 1
        assert g1["death_censored"] == 1
        assert g1["left_truncated"] == 1
        assert g1["right_censored"] == 1
        assert g1["sensitivity"] == pytest.approx(0.25)
        assert g1["n_gen"] == 4

        g2 = result["trait1"]["gen2"]
        assert g2["true_affected"] == 3
        assert g2["observed"] == 1
        assert g2["death_censored"] == 1
        assert g2["left_truncated"] == 0
        assert g2["right_censored"] == 1
        assert g2["sensitivity"] == pytest.approx(1 / 3)
        assert g2["n_gen"] == 4

    def test_missing_generation_column(self):
        """Returns {} when generation column is absent."""
        df = pd.DataFrame({"id": [0], "t1": [50.0], "affected1": [True]})
        result = compute_censoring_cascade(df, censor_age=80.0, gen_censoring={1: [0.0, 80.0]})
        assert result == {}

    def test_empty_windows(self):
        """Returns {} when all windows have hi <= lo."""
        df = pd.DataFrame(
            {
                "id": [0],
                "generation": [1],
                "death_age": [50.0],
                "t1": [30.0],
                "t2": [200.0],
                "affected1": [True],
                "affected2": [False],
            }
        )
        gen_censoring = {1: [80.0, 20.0]}
        result = compute_censoring_cascade(df, censor_age=80.0, gen_censoring=gen_censoring)
        assert result == {}

    def test_no_death_age_column(self):
        """Without death_age, death_censored should be 0 and in-window cases are all observed."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "generation": [1, 1],
                "t1": [30.0, 50.0],
                "t2": [200.0, 200.0],
                "affected1": [True, True],
                "affected2": [False, False],
            }
        )
        gen_censoring = {1: [0.0, 80.0]}
        result = compute_censoring_cascade(df, censor_age=80.0, gen_censoring=gen_censoring)
        g1 = result["trait1"]["gen1"]
        assert g1["death_censored"] == 0
        # Both t1 values are in [0, 80], so both observed
        assert g1["observed"] == 2
        assert g1["true_affected"] == 2
        assert g1["sensitivity"] == pytest.approx(1.0)

    def test_no_true_affected(self):
        """When no individuals are truly affected, sensitivity is NaN."""
        df = pd.DataFrame(
            {
                "id": [0],
                "generation": [1],
                "death_age": [50.0],
                "t1": [90.0],  # 90 >= censor_age 80 -> not truly affected
                "t2": [200.0],
                "affected1": [False],
                "affected2": [False],
            }
        )
        gen_censoring = {1: [0.0, 80.0]}
        result = compute_censoring_cascade(df, censor_age=80.0, gen_censoring=gen_censoring)
        g1 = result["trait1"]["gen1"]
        assert g1["true_affected"] == 0
        assert np.isnan(g1["sensitivity"])

    def test_window_in_result(self, cascade_df):
        """Each generation result includes the window boundaries."""
        gen_censoring = {1: [20.0, 70.0], 2: [0.0, 60.0]}
        result = compute_censoring_cascade(cascade_df, censor_age=80.0, gen_censoring=gen_censoring)
        assert result["trait1"]["gen1"]["window"] == [20.0, 70.0]
        assert result["trait1"]["gen2"]["window"] == [0.0, 60.0]


# ===================================================================
# Tests for compute_censoring_windows
# ===================================================================


class TestComputeCensoringWindows:
    def test_structure(self):
        """Verify returned dict has expected top-level keys."""
        df = pd.DataFrame(
            {
                "id": [0, 1],
                "generation": [1, 1],
                "t1": [30.0, 50.0],
                "t_observed1": [30.0, 50.0],
                "affected1": [True, True],
                "death_censored1": [False, False],
                "t2": [200.0, 200.0],
                "t_observed2": [200.0, 200.0],
                "affected2": [False, False],
                "death_censored2": [False, False],
            }
        )
        gen_censoring = {1: [0.0, 80.0]}
        result = compute_censoring_windows(df, censor_age=80.0, gen_censoring=gen_censoring)
        assert result is not None
        assert "generations" in result
        assert "censoring_ages" in result
        assert "censor_age" in result
        assert "gen1" in result["generations"]

    def test_returns_none_without_generation(self):
        """Returns None when generation column is missing."""
        df = pd.DataFrame({"id": [0], "t1": [30.0]})
        result = compute_censoring_windows(df, censor_age=80.0, gen_censoring={1: [0.0, 80.0]})
        assert result is None

    def test_pct_affected(self):
        """Verify pct_affected matches manually computed fraction.

        2 out of 4 individuals are affected for trait1.
        """
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "generation": [1, 1, 1, 1],
                "t1": [30.0, 50.0, 90.0, 100.0],
                "t_observed1": [30.0, 50.0, 90.0, 100.0],
                "affected1": [True, True, False, False],
                "death_censored1": [False, False, False, False],
                "t2": [200.0, 200.0, 200.0, 200.0],
                "t_observed2": [200.0, 200.0, 200.0, 200.0],
                "affected2": [False, False, False, False],
                "death_censored2": [False, False, False, False],
            }
        )
        gen_censoring = {1: [0.0, 80.0]}
        result = compute_censoring_windows(df, censor_age=80.0, gen_censoring=gen_censoring)
        # 2/4 = 0.5 affected for trait1
        assert result["generations"]["gen1"]["trait1"]["pct_affected"] == pytest.approx(0.5)
        # 0/4 = 0.0 affected for trait2
        assert result["generations"]["gen1"]["trait2"]["pct_affected"] == pytest.approx(0.0)

    def test_empty_generation(self):
        """Generation with no matching individuals gets n=0 entry."""
        df = pd.DataFrame(
            {
                "id": [0],
                "generation": [1],
                "t1": [30.0],
                "t_observed1": [30.0],
                "affected1": [True],
                "death_censored1": [False],
                "t2": [200.0],
                "t_observed2": [200.0],
                "affected2": [False],
                "death_censored2": [False],
            }
        )
        # gen_censoring includes gen 2 but no data for it
        gen_censoring = {1: [0.0, 80.0], 2: [0.0, 60.0]}
        result = compute_censoring_windows(df, censor_age=80.0, gen_censoring=gen_censoring)
        assert result["generations"]["gen2"]["n"] == 0


# ===================================================================
# Module-scoped fixture: phenotyped pedigree for pair-based stats
# ===================================================================


_DEFAULT_SIM_PARAMS = dict(
    seed=42, N=1000, G_ped=3, G_sim=3, mating_lambda=0.5, p_mztwin=0.02,
    A1=0.5, C1=0.2, A2=0.5, C2=0.2, rA=0.3, rC=0.5, assort1=0.0, assort2=0.0,
)  # fmt: skip


@pytest.fixture(scope="module")
def phenotyped_df():
    """Simulated + thresholded pedigree with all columns needed by stats functions."""
    from simace.phenotyping.threshold import apply_threshold
    from simace.simulation.simulate import run_simulation

    ped = run_simulation(**_DEFAULT_SIM_PARAMS)
    gen = ped["generation"].values
    rng = np.random.default_rng(42)
    for t in [1, 2]:
        liab = ped[f"liability{t}"].values
        ped[f"affected{t}"] = apply_threshold(liab, gen, 0.10)
        aff = ped[f"affected{t}"].values
        ped[f"t_observed{t}"] = np.where(aff, rng.uniform(10, 70, len(ped)), 80.0)
        ped[f"t{t}"] = np.where(aff, ped[f"t_observed{t}"], rng.uniform(85, 200, len(ped)))
    ped["death_age"] = rng.uniform(40, 100, len(ped))
    return ped


@pytest.fixture(scope="module")
def extracted_pairs(phenotyped_df):
    """Pre-extracted relationship pairs."""
    from simace.core.pedigree_graph import extract_relationship_pairs

    return extract_relationship_pairs(phenotyped_df)


# ===================================================================
# Tests for compute_prevalence
# ===================================================================


class TestComputePrevalence:
    def test_known_values(self):
        df = pd.DataFrame(
            {
                "affected1": [True, True, False, False, False],
                "affected2": [False, True, True, True, False],
            }
        )
        result = compute_prevalence(df)
        assert result["trait1"] == pytest.approx(0.4)
        assert result["trait2"] == pytest.approx(0.6)

    def test_all_affected(self):
        df = pd.DataFrame({"affected1": [True, True], "affected2": [True, True]})
        result = compute_prevalence(df)
        assert result["trait1"] == pytest.approx(1.0)
        assert result["trait2"] == pytest.approx(1.0)

    def test_none_affected(self):
        df = pd.DataFrame({"affected1": [False, False], "affected2": [False, False]})
        result = compute_prevalence(df)
        assert result["trait1"] == pytest.approx(0.0)
        assert result["trait2"] == pytest.approx(0.0)


# ===================================================================
# Tests for compute_mortality
# ===================================================================


class TestComputeMortality:
    def test_known_rates(self):
        """5 people, censor_age=50, decade bins [0-10), [10-20), ..., [40-50)."""
        df = pd.DataFrame({"death_age": [5.0, 15.0, 25.0, 35.0, 95.0]})
        result = compute_mortality(df, censor_age=50)
        assert len(result["decade_labels"]) == 5
        assert result["decade_labels"][0] == "0-9"
        # Decade 0-10: alive=5, died=1 -> 0.2
        assert result["rates"][0] == pytest.approx(0.2)
        # Decade 10-20: alive=4, died=1 -> 0.25
        assert result["rates"][1] == pytest.approx(0.25)

    def test_no_deaths_before_censor(self):
        df = pd.DataFrame({"death_age": [100.0, 200.0]})
        result = compute_mortality(df, censor_age=80)
        assert all(r == pytest.approx(0.0) for r in result["rates"])


# ===================================================================
# Tests for compute_regression
# ===================================================================


class TestComputeRegression:
    def test_positive_slope(self):
        """Higher liability → earlier onset (negative slope with liability)."""
        df = pd.DataFrame(
            {
                "affected1": [True, True, True, True, False],
                "affected2": [True, True, True, True, False],
                "t_observed1": [20.0, 30.0, 40.0, 50.0, 80.0],
                "t_observed2": [25.0, 35.0, 45.0, 55.0, 80.0],
                "liability1": [3.0, 2.0, 1.0, 0.5, -1.0],
                "liability2": [3.0, 2.0, 1.0, 0.5, -1.0],
            }
        )
        result = compute_regression(df)
        assert result["trait1"] is not None
        assert result["trait1"]["n"] == 4
        assert result["trait1"]["r"] != 0

    def test_too_few_affected_returns_none(self):
        df = pd.DataFrame(
            {
                "affected1": [True, False],
                "affected2": [False, False],
                "t_observed1": [30.0, 80.0],
                "t_observed2": [80.0, 80.0],
                "liability1": [1.0, -1.0],
                "liability2": [1.0, -1.0],
            }
        )
        result = compute_regression(df)
        # Only 1 affected for trait1, 0 for trait2 → None
        assert result["trait1"] is None
        assert result["trait2"] is None

    def test_missing_liability_returns_none(self):
        df = pd.DataFrame(
            {
                "affected1": [True, True],
                "affected2": [True, True],
                "t_observed1": [30.0, 40.0],
                "t_observed2": [30.0, 40.0],
            }
        )
        result = compute_regression(df)
        assert result["trait1"] is None
        assert result["trait2"] is None


# ===================================================================
# Tests for compute_joint_affection
# ===================================================================


class TestComputeJointAffection:
    def test_known_counts(self):
        df = pd.DataFrame(
            {
                "affected1": [True, True, False, False, True],
                "affected2": [True, False, True, False, True],
                "sex": [0, 1, 0, 1, 0],
            }
        )
        result = compute_joint_affection(df)
        assert result["counts"]["both"] == 2
        assert result["counts"]["trait1_only"] == 1
        assert result["counts"]["trait2_only"] == 1
        assert result["counts"]["neither"] == 1
        assert result["n"] == 5
        assert result["proportions"]["both"] == pytest.approx(0.4)

    def test_by_sex_present(self):
        df = pd.DataFrame(
            {
                "affected1": [True, False, True, False],
                "affected2": [True, True, False, False],
                "sex": [0, 0, 1, 1],
            }
        )
        result = compute_joint_affection(df)
        assert "female" in result["by_sex"]
        assert "male" in result["by_sex"]
        # Female: id0 both -> 1/2 = 0.5
        assert result["by_sex"]["female"] == pytest.approx(0.5)
        # Male: no both -> 0/2 = 0.0
        assert result["by_sex"]["male"] == pytest.approx(0.0)


# ===================================================================
# Tests for compute_parent_status
# ===================================================================


class TestComputeParentStatus:
    def test_known_structure(self):
        """3 founders + 2 children: children have 2 parents each."""
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "mother": [-1, -1, -1, 0, 0],
                "father": [-1, -1, -1, 1, 1],
            }
        )
        result = compute_parent_status(df)
        # Founders (0,1,2) have 0 parents in phenotyped set (mother=-1)
        # Children (3,4) have 2 parents each (0 and 1 are in the set)
        assert result["phenotyped"]["0"] == 3
        assert result["phenotyped"]["2"] == 2

    def test_with_ped_df(self):
        df = pd.DataFrame(
            {
                "id": [3, 4],
                "mother": [0, 0],
                "father": [1, 1],
            }
        )
        df_ped = pd.DataFrame({"id": [0, 1, 2, 3, 4]})
        result = compute_parent_status(df, df_ped=df_ped)
        # Parents 0,1 not in df (phenotype), so phenotyped = 0 for both children
        assert result["phenotyped"]["0"] == 2
        # But parents 0,1 are in df_ped
        assert result["in_pedigree"]["2"] == 2

    def test_missing_columns(self):
        df = pd.DataFrame({"id": [0, 1]})
        assert compute_parent_status(df) == {}


# ===================================================================
# Tests for compute_cumulative_incidence (pair-based fixture)
# ===================================================================


class TestComputeCumulativeIncidence:
    def test_structure(self, phenotyped_df):
        result = compute_cumulative_incidence(phenotyped_df, censor_age=80)
        assert "trait1" in result
        assert "trait2" in result
        for trait in ["trait1", "trait2"]:
            assert "ages" in result[trait]
            assert "observed_values" in result[trait]
            assert "true_values" in result[trait]
            ages = result[trait]["ages"]
            assert ages[0] == pytest.approx(0.0)
            assert ages[-1] == pytest.approx(80.0)

    def test_values_non_decreasing(self, phenotyped_df):
        result = compute_cumulative_incidence(phenotyped_df, censor_age=80)
        for trait in ["trait1", "trait2"]:
            obs = result[trait]["observed_values"]
            assert all(obs[i] <= obs[i + 1] for i in range(len(obs) - 1))


# ===================================================================
# Tests for compute_cumulative_incidence_by_sex
# ===================================================================


class TestComputeCumulativeIncidenceBySex:
    def test_structure(self, phenotyped_df):
        result = compute_cumulative_incidence_by_sex(phenotyped_df, censor_age=80)
        for trait in ["trait1", "trait2"]:
            assert "female" in result[trait]
            assert "male" in result[trait]
            for sex in ["female", "male"]:
                assert result[trait][sex]["n"] > 0
                assert 0 <= result[trait][sex]["prevalence"] <= 1

    def test_missing_sex_returns_empty(self):
        df = pd.DataFrame({"affected1": [True], "affected2": [False], "t_observed1": [30.0], "t_observed2": [80.0]})
        assert compute_cumulative_incidence_by_sex(df, censor_age=80) == {}


# ===================================================================
# Tests for compute_cumulative_incidence_by_sex_generation
# ===================================================================


class TestComputeCumulativeIncidenceBySexGeneration:
    def test_structure(self, phenotyped_df):
        result = compute_cumulative_incidence_by_sex_generation(phenotyped_df, censor_age=80)
        assert "trait1" in result
        # Should have gen keys
        gen_keys = list(result["trait1"].keys())
        assert len(gen_keys) > 0
        for gk in gen_keys:
            assert "female" in result["trait1"][gk] or "male" in result["trait1"][gk]


# ===================================================================
# Tests for compute_liability_correlations
# ===================================================================


class TestComputeLiabilityCorrelations:
    def test_structure(self, phenotyped_df, extracted_pairs):
        result = compute_liability_correlations(phenotyped_df, pairs=extracted_pairs)
        assert "trait1" in result
        assert "trait2" in result
        for trait in ["trait1", "trait2"]:
            assert "MZ" in result[trait]
            assert "FS" in result[trait]

    def test_fs_positive(self, phenotyped_df, extracted_pairs):
        """Full siblings share ~0.5 of A — liability correlation should be positive."""
        result = compute_liability_correlations(phenotyped_df, pairs=extracted_pairs)
        fs = result["trait1"]["FS"]
        if fs is not None:
            assert fs > 0


# ===================================================================
# Tests for compute_tetrachoric
# ===================================================================


class TestComputeTetrachoric:
    def test_structure(self, phenotyped_df, extracted_pairs):
        result = compute_tetrachoric(phenotyped_df, pairs=extracted_pairs)
        assert "trait1" in result
        for ptype in ["MZ", "FS", "MO", "FO"]:
            entry = result["trait1"][ptype]
            assert "r" in entry
            assert "se" in entry
            assert "n_pairs" in entry

    def test_n_pairs_positive_for_common_types(self, phenotyped_df, extracted_pairs):
        result = compute_tetrachoric(phenotyped_df, pairs=extracted_pairs)
        for ptype in ["FS", "MO", "FO"]:
            assert result["trait1"][ptype]["n_pairs"] > 0


# ===================================================================
# Tests for compute_tetrachoric_by_generation
# ===================================================================


class TestComputeTetrachoricByGeneration:
    def test_structure(self, phenotyped_df, extracted_pairs):
        result = compute_tetrachoric_by_generation(phenotyped_df, pairs=extracted_pairs)
        assert len(result) > 0
        for gen_key, gen_data in result.items():
            assert gen_key.startswith("gen")
            assert "trait1" in gen_data
            assert "trait2" in gen_data

    def test_missing_generation_returns_empty(self):
        df = pd.DataFrame({"affected1": [True], "affected2": [False], "liability1": [1.0], "liability2": [1.0]})
        assert compute_tetrachoric_by_generation(df) == {}


# ===================================================================
# Tests for compute_cross_trait_tetrachoric
# ===================================================================


class TestComputeCrossTraitTetrachoric:
    def test_structure(self, phenotyped_df, extracted_pairs):
        result = compute_cross_trait_tetrachoric(phenotyped_df, pairs=extracted_pairs)
        assert "same_person" in result
        assert "r" in result["same_person"]
        assert "n" in result["same_person"]
        assert "same_person_by_generation" in result
        assert "cross_person" in result
        for ptype in ["FS", "MO", "FO"]:
            assert ptype in result["cross_person"]


# ===================================================================
# Tests for compute_parent_offspring_corr
# ===================================================================


class TestComputeParentOffspringCorr:
    def test_structure(self, phenotyped_df):
        result = compute_parent_offspring_corr(phenotyped_df)
        assert "trait1" in result
        assert "trait2" in result
        # Should have gen keys (gen1, gen2 for G_ped=3)
        assert "gen1" in result["trait1"] or "gen2" in result["trait1"]

    def test_positive_slope(self, phenotyped_df):
        """With A1=0.5, midparent-offspring slope should be positive."""
        result = compute_parent_offspring_corr(phenotyped_df)
        # Check the last generation where we expect enough pairs
        for gen_key in sorted(result["trait1"].keys()):
            entry = result["trait1"][gen_key]
            if entry["slope"] is not None:
                assert entry["slope"] > 0
                break

    def test_missing_generation_returns_empty(self):
        df = pd.DataFrame({"id": [0], "liability1": [1.0], "liability2": [1.0]})
        assert compute_parent_offspring_corr(df) == {}


# ===================================================================
# Tests for compute_tetrachoric_by_sex
# ===================================================================


class TestComputeTetrachoricBySex:
    def test_structure(self, phenotyped_df, extracted_pairs):
        result = compute_tetrachoric_by_sex(phenotyped_df, pairs=extracted_pairs)
        assert "female" in result
        assert "male" in result
        for sex in ["female", "male"]:
            assert "trait1" in result[sex]
            assert "trait2" in result[sex]


# ===================================================================
# Tests for compute_parent_offspring_corr_by_sex
# ===================================================================


class TestComputeParentOffspringCorrBySex:
    def test_structure(self, phenotyped_df):
        result = compute_parent_offspring_corr_by_sex(phenotyped_df)
        assert "female" in result
        assert "male" in result
        for sex in ["female", "male"]:
            assert "trait1" in result[sex]

    def test_missing_generation_returns_empty(self):
        df = pd.DataFrame({"id": [0], "sex": [0], "liability1": [1.0], "liability2": [1.0]})
        assert compute_parent_offspring_corr_by_sex(df) == {}


# ===================================================================
# Tests for compute_mate_correlation
# ===================================================================


class TestComputeMateCorrelation:
    def test_structure(self, phenotyped_df):
        result = compute_mate_correlation(phenotyped_df)
        assert "matrix" in result
        assert "n_pairs" in result
        assert len(result["matrix"]) == 2
        assert len(result["matrix"][0]) == 2
        assert result["n_pairs"] > 0

    def test_near_zero_for_random_mating(self, phenotyped_df):
        """With assort1=0, mate correlations should be near zero."""
        result = compute_mate_correlation(phenotyped_df)
        for i in range(2):
            for j in range(2):
                assert abs(result["matrix"][i][j]) < 0.15


# ===================================================================
# Tests for compute_affected_correlations
# ===================================================================


class TestComputeAffectedCorrelations:
    def test_known_phi(self):
        """Phi r on a synthetic MZ-pair set with hand-computable values.

        20 pairs with (a1, a2):
          4 (1,1), 2 (1,0), 2 (0,1), 12 (0,0)
        p(a1=1) = 6/20 = 0.3, p(a2=1) = 6/20 = 0.3
        p(1,1) = 4/20 = 0.2
        Cov = 0.2 - 0.3*0.3 = 0.11
        Var(a1) = Var(a2) = 0.3*0.7 = 0.21
        phi = 0.11 / 0.21 ≈ 0.5238
        """
        n_pairs = 20
        # 40 rows: first half = side A, second half = side B
        a_side = np.array([1] * 4 + [1] * 2 + [0] * 2 + [0] * 12, dtype=np.float64)
        b_side = np.array([1] * 4 + [0] * 2 + [1] * 2 + [0] * 12, dtype=np.float64)
        affected1 = np.concatenate([a_side, b_side])

        df = pd.DataFrame(
            {
                "id": np.arange(len(affected1), dtype=np.int64),
                "affected1": affected1.astype(bool),
                "affected2": np.zeros(len(affected1), dtype=bool),
            }
        )
        idx1 = np.arange(n_pairs, dtype=np.int64)
        idx2 = np.arange(n_pairs, 2 * n_pairs, dtype=np.int64)
        pairs = {
            "MZ": (idx1, idx2),
            "FS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "MO": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "FO": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "MHS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "PHS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "1C": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        }

        result = compute_affected_correlations(df, pairs=pairs)
        assert result["trait1"]["MZ"] == pytest.approx(0.11 / 0.21, abs=1e-6)
        # trait2 is constant (all unaffected): phi undefined -> None
        assert result["trait2"]["MZ"] is None
        # n_pairs < 10 -> None
        assert result["trait1"]["FS"] is None

    def test_constant_side_returns_none(self):
        """Phi is undefined when either indicator has zero variance."""
        df = pd.DataFrame(
            {
                "id": np.arange(30, dtype=np.int64),
                "affected1": np.array([True] * 15 + [False] * 15, dtype=bool),
                # Pair B (indices 15..29) is all-False -> constant -> None
                "affected2": np.zeros(30, dtype=bool),
            }
        )
        idx1 = np.arange(15, dtype=np.int64)
        idx2 = np.arange(15, 30, dtype=np.int64)
        pairs = {
            "MZ": (idx1, idx2),
            "FS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "MO": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "FO": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "MHS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "PHS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            "1C": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        }
        result = compute_affected_correlations(df, pairs=pairs)
        # Side A varies (15 True + 0 False in idx1 -> wait, idx1 is 0..14 which is all True)
        # Both sides constant -> None
        assert result["trait1"]["MZ"] is None

    def test_structure_matches_liability(self, phenotyped_df, extracted_pairs):
        """Affected-side structure mirrors compute_liability_correlations."""
        liab = compute_liability_correlations(phenotyped_df, pairs=extracted_pairs)
        aff = compute_affected_correlations(phenotyped_df, pairs=extracted_pairs)
        assert set(aff.keys()) == set(liab.keys())
        for trait_key in ["trait1", "trait2"]:
            assert set(aff[trait_key].keys()) == set(liab[trait_key].keys())


# ===================================================================
# Tests for compute_parent_offspring_affected_corr
# ===================================================================


class TestComputeParentOffspringAffectedCorr:
    def test_known_slope(self):
        """Trio df with hand-computable midparent-offspring regression slope.

        20 trios; the midparent indicator is 1 in 10 trios and 0 in 10.
        Offspring affected matches midparent in 16 of 20 trios and differs in 4.

        Midparent mean = 10/20 = 0.5, offspring mean = 10/20 = 0.5 (symmetric flips).
        Cov(mid, off) = (Σ (m_i - 0.5)(o_i - 0.5)) / (n - 1).
        With (m, o) counts: (1,1)=8, (1,0)=2, (0,1)=2, (0,0)=8:
          Σ (m-0.5)(o-0.5) = 8·0.25 + 2·(-0.25) + 2·(-0.25) + 8·0.25 = 3.0
        Var(mid) = Σ (m - 0.5)² / (n-1) = (20·0.25)/19 = 5/19
        slope = Cov / Var(mid) = (3/19) / (5/19) = 0.6
        """
        founder_ids = list(range(100, 140))  # 40 founders
        mother_ids = founder_ids[0:20]
        father_ids = founder_ids[20:40]
        child_ids = list(range(1, 21))

        # Parent pattern: first 10 trios have both parents True, last 10 both False.
        parent_aff = [True] * 10 + [False] * 10
        # Offspring pattern: flip 2 of the concordant-T trios to F and 2 of the
        # concordant-F trios to T.  16 match, 4 don't.
        child_aff = parent_aff[:]
        child_aff[0] = False
        child_aff[1] = False
        child_aff[10] = True
        child_aff[11] = True

        founders_aff = [False] * 40
        for i, m in enumerate(mother_ids):
            founders_aff[founder_ids.index(m)] = parent_aff[i]
        for i, f in enumerate(father_ids):
            founders_aff[founder_ids.index(f)] = parent_aff[i]
        ids = founder_ids + child_ids
        mothers = [-1] * 40 + mother_ids
        fathers = [-1] * 40 + father_ids
        affected1 = founders_aff + child_aff
        df = pd.DataFrame(
            {
                "id": ids,
                "mother": mothers,
                "father": fathers,
                "affected1": affected1,
                "affected2": [False] * len(ids),
            }
        )
        result = compute_parent_offspring_affected_corr(df)
        assert result["trait1"]["slope"] == pytest.approx(0.6, abs=1e-6)
        assert result["trait1"]["n_pairs"] == 20
        # trait2: all-unaffected midparents -> slope None
        assert result["trait2"]["slope"] is None

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"affected1": [True, False]})
        assert compute_parent_offspring_affected_corr(df) == {}

    def test_too_few_trios_returns_null(self):
        df = pd.DataFrame(
            {
                "id": [100, 101, 1, 2],
                "mother": [-1, -1, 100, 100],
                "father": [-1, -1, 101, 101],
                "affected1": [True, False, True, False],
                "affected2": [False, False, False, False],
            }
        )
        result = compute_parent_offspring_affected_corr(df)
        assert result["trait1"]["slope"] is None
        assert result["trait1"]["n_pairs"] == 2

    def test_smoke_on_pedigree(self, phenotyped_df):
        """On a simulated pedigree the PO slope should be finite and in [-0.5, 1.5]."""
        result = compute_parent_offspring_affected_corr(phenotyped_df)
        for trait_key in ["trait1", "trait2"]:
            entry = result[trait_key]
            slope = entry["slope"]
            if slope is not None:
                assert -0.5 <= slope <= 1.5


# ===================================================================
# Tests for compute_observed_h2_estimators
# ===================================================================


class TestComputeObservedH2Estimators:
    def test_closed_form(self):
        """Algebraic check: each estimator combines its inputs correctly."""
        stats = {
            "affected_correlations": {
                "trait1": {
                    "MZ": 0.40,
                    "FS": 0.15,
                    "MO": 0.10,
                    "FO": 0.10,
                    "MHS": 0.05,
                    "PHS": 0.05,
                    "1C": 0.02,
                },
                "trait2": {
                    "MZ": None,
                    "FS": 0.20,
                    "MO": None,
                    "FO": None,
                    "MHS": None,
                    "PHS": None,
                    "1C": None,
                },
            },
            "parent_offspring_affected_corr": {
                "trait1": {"slope": 0.12},
                "trait2": {"slope": None},
            },
        }
        result = compute_observed_h2_estimators(stats)
        t1 = result["trait1"]
        assert t1["falconer"] == pytest.approx(2.0 * (0.40 - 0.15))
        assert t1["sibs"] == pytest.approx(2.0 * 0.15)
        assert t1["po"] == pytest.approx(0.12)
        assert t1["hs"] == pytest.approx(4.0 * 0.05)
        assert t1["cousins"] == pytest.approx(8.0 * 0.02)

        t2 = result["trait2"]
        assert t2["falconer"] is None  # r_MZ is None
        assert t2["sibs"] == pytest.approx(2.0 * 0.20)
        assert t2["po"] is None
        assert t2["hs"] is None
        assert t2["cousins"] is None

    def test_hs_averages_available_sides(self):
        """When only one of MHS/PHS is present, HS uses that one × 4."""
        stats = {
            "affected_correlations": {
                "trait1": {"MZ": None, "FS": None, "MO": None, "FO": None, "MHS": 0.07, "PHS": None, "1C": None},
                "trait2": {"MZ": None, "FS": None, "MO": None, "FO": None, "MHS": None, "PHS": None, "1C": None},
            },
            "parent_offspring_affected_corr": {
                "trait1": {"slope": None},
                "trait2": {"slope": None},
            },
        }
        result = compute_observed_h2_estimators(stats)
        assert result["trait1"]["hs"] == pytest.approx(4.0 * 0.07)
        assert result["trait2"]["hs"] is None

    def test_missing_inputs_returns_all_none(self):
        """Empty stats dict yields all-None estimators."""
        result = compute_observed_h2_estimators({})
        assert result["trait1"]["falconer"] is None
        assert result["trait1"]["sibs"] is None
        assert result["trait1"]["po"] is None
        assert result["trait1"]["hs"] is None
        assert result["trait1"]["cousins"] is None
