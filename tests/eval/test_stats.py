import pytest

from arxiv_agent.eval.stats import bootstrap_ci


def test_the_interval_is_for_the_mean_not_for_the_values() -> None:
    # 100 alternating 0/1 values: mean 0.5, standard error 0.05, so a 95%
    # interval of the mean is about 0.40-0.60, far inside the values' 0-1 range.
    low, high = bootstrap_ci([0.0, 1.0] * 50)
    assert 0.38 <= low <= 0.42
    assert 0.58 <= high <= 0.62


def test_the_same_seed_gives_the_same_interval() -> None:
    values = [0.2, 0.9, 0.4, 1.0, 0.0, 0.7]
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


def test_constant_values_have_no_width() -> None:
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_no_values_have_no_interval() -> None:
    with pytest.raises(ValueError, match="no values"):
        bootstrap_ci([])
