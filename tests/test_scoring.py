import pytest
from risk_control.scoring import score_factors, size_for_risk


def test_five_bearish_one_neutral_is_not_reported_bullish():
    # Mirrors the reported run: price below AVWAP and 200 EMA, negative velocity/momentum/OBI-skew, CVD neutral
    s = score_factors(velocity=-5.8, recent_mom=-12.3, weighted_obi=-0.18,
                      is_macro_bull=False, flow_bull=False, cvd_vote=0, reversal_vote=0)
    assert (s.bear, s.bull, s.neutral) == (5, 0, 1)
    assert s.direction == 'BEARISH'
    assert s.conviction == pytest.approx(0.5 + 0.4 * 5 / 6)


def test_symmetry_mirror_image_gives_same_conviction():
    bear = score_factors(-1, -1, -0.2, False, False, 0, 0)
    bull = score_factors(1, 1, 0.2, True, True, 0, 0)
    assert bear.conviction == pytest.approx(bull.conviction)
    assert bull.direction == 'BULLISH' and bear.direction == 'BEARISH'


def test_tie_is_neutral_half_conviction():
    s = score_factors(1, 1, 0.2, False, False, cvd_vote=-1)  # 3 bull, 3 bear
    assert s.direction == 'NEUTRAL'
    assert s.conviction == pytest.approx(0.5)


def test_bearish_cvd_is_counted():
    s = score_factors(-1, -1, 0.0, False, False, cvd_vote=-1, reversal_vote=0)
    assert s.bear == 5 and s.neutral == 1


def test_counts_always_sum_to_total():
    s = score_factors(0.0, 0.0, 0.0, True, False, 1, 0)
    assert s.bull + s.bear + s.neutral == s.total == 6


def test_risk_is_measured_on_equity_not_on_notional():
    # INR 5,000 equity, 1.5% -> INR 75 budget. Stop 2% away -> max notional 3,750.
    r = size_for_risk(target_notional_inr=4000, entry=100.0, stop=102.0, equity_inr=5000,
                      risk_pct=0.015, max_notional_inr=5000, min_notional_inr=880)
    assert r.risk_budget_inr == pytest.approx(75.0)
    assert r.notional_inr == pytest.approx(3750.0)
    assert r.risk_inr <= 75.0 + 1e-9
    assert r.binding == 'risk budget'


def test_max_notional_binds_for_tight_stop():
    r = size_for_risk(4000, 77300.0, 77500.0, 5000, 0.015, 3000, 880)
    assert r.notional_inr == pytest.approx(3000.0)
    assert r.binding == 'max notional'
    assert r.risk_inr < 75.0


def test_below_exchange_minimum_flagged():
    r = size_for_risk(500, 77300.0, 77500.0, 5000, 0.015, 5000, 880)
    assert r.below_min_notional is True
