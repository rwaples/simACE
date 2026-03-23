"""Unit tests for computation functions in sim_ace.stats.

Each test builds a small DataFrame with known values so expected outputs
can be verified by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim_ace.stats import (
    compute_censoring_cascade,
    compute_censoring_confusion,
    compute_censoring_windows,
    compute_mean_family_size,
    compute_person_years,
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
