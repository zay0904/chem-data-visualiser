"""Unit tests for src.analyzer."""
import numpy as np
import pytest

from src.analyzer import (
    calculate_rates,
    fit_linear,
    beer_lambert,
    half_life_first_order,
)


def test_calculate_rates_linear():
    # C = 1 - 0.5t  →  rate = 0.5 everywhere
    t = np.array([0, 1, 2, 3, 4], dtype=float)
    c = 1.0 - 0.5 * t
    rates = calculate_rates(c, t)
    assert np.allclose(rates, 0.5, atol=1e-9)


def test_calculate_rates_shape_mismatch():
    with pytest.raises(ValueError):
        calculate_rates([1, 2, 3], [0, 1])


def test_calculate_rates_non_increasing_time():
    with pytest.raises(ValueError):
        calculate_rates([1, 2, 3], [0, 5, 2])


def test_fit_linear_exact():
    # y = 3x + 2 (exact)
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = 3 * x + 2
    f = fit_linear(x, y)
    assert f.slope == pytest.approx(3.0, rel=1e-10)
    assert f.intercept == pytest.approx(2.0, rel=1e-10)
    assert f.r_squared == pytest.approx(1.0, rel=1e-10)


def test_beer_lambert():
    # c = A / (ε · l)
    c = beer_lambert(absorbance=0.45, molar_absorptivity=15000.0,
                     path_length_cm=1.0)
    assert c == pytest.approx(0.45 / 15000.0, rel=1e-12)


def test_beer_lambert_rejects_negative_absorbance():
    with pytest.raises(ValueError):
        beer_lambert(-0.1, 15000.0)


def test_half_life_first_order():
    # ln(2) / 0.0243 ≈ 28.52 s
    assert "28.5" in half_life_first_order(0.0243, "s")


def test_fit_linear_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_linear([1.0], [2.0])