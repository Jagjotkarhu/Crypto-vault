"""Direction-aware factor scoring and equity-based risk sizing.

Pure functions (no I/O) so they are unit-testable and so the printed
table, the summary line and the sizing all derive from one source.
"""
from dataclasses import dataclass, field

OBI_THRESHOLD = 0.08


@dataclass
class FactorResult:
    name: str
    vote: int  # +1 bullish, -1 bearish, 0 neutral


@dataclass
class ScoreResult:
    factors: list = field(default_factory=list)
    bull: int = 0
    bear: int = 0
    neutral: int = 0
    direction: str = 'NEUTRAL'
    conviction: float = 0.5  # uncalibrated signal strength, NOT a probability

    @property
    def total(self) -> int:
        return len(self.factors)


def score_factors(velocity, recent_mom, weighted_obi, is_macro_bull,
                  flow_bull, cvd_vote=0, reversal_vote=0) -> ScoreResult:
    """Each factor votes +1 / -1 / 0. cvd_vote and reversal_vote are already
    signed (+1 bullish absorption/reversal, -1 bearish, 0 none)."""
    sixth = 1 if (cvd_vote > 0 or reversal_vote > 0) else (-1 if (cvd_vote < 0 or reversal_vote < 0) else 0)
    obi_vote = 1 if weighted_obi > OBI_THRESHOLD else (-1 if weighted_obi < -OBI_THRESHOLD else 0)
    factors = [
        FactorResult('Kalman velocity', 1 if velocity > 0 else (-1 if velocity < 0 else 0)),
        FactorResult('FracDiff momentum', 1 if recent_mom > 0 else (-1 if recent_mom < 0 else 0)),
        FactorResult('Order-book imbalance', obi_vote),
        FactorResult('200 EMA regime', 1 if is_macro_bull else -1),
        FactorResult('Anchored VWAP', 1 if flow_bull else -1),
        FactorResult('CVD / order-block reversal', sixth),
    ]
    bull = sum(1 for f in factors if f.vote > 0)
    bear = sum(1 for f in factors if f.vote < 0)
    neutral = sum(1 for f in factors if f.vote == 0)
    if bull > bear:
        direction = 'BULLISH'
    elif bear > bull:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'
    # Symmetric: depends on the margin between sides, never on one side alone.
    # 3-3 -> 0.50, 5-0 -> 0.833, 6-0 -> 0.90.
    conviction = 0.50 + 0.40 * (abs(bull - bear) / float(len(factors)))
    return ScoreResult(factors, bull, bear, neutral, direction, conviction)


@dataclass
class SizingResult:
    notional_inr: float
    risk_inr: float
    risk_budget_inr: float
    stop_pct: float
    binding: str  # which limit set the size
    below_min_notional: bool


def size_for_risk(target_notional_inr, entry, stop, equity_inr, risk_pct,
                  max_notional_inr, min_notional_inr) -> SizingResult:
    """Cap notional so that notional * stop_distance_pct <= equity * risk_pct."""
    risk_budget = equity_inr * risk_pct
    stop_pct = abs(entry - stop) / entry if entry > 0 else 0.0
    max_by_risk = risk_budget / stop_pct if stop_pct > 0 else float('inf')
    notional = min(abs(target_notional_inr), max_by_risk, max_notional_inr)
    if notional == max_by_risk and max_by_risk < min(abs(target_notional_inr), max_notional_inr):
        binding = 'risk budget'
    elif notional == max_notional_inr and max_notional_inr < abs(target_notional_inr):
        binding = 'max notional'
    else:
        binding = 'conviction target'
    return SizingResult(
        notional_inr=notional,
        risk_inr=notional * stop_pct,
        risk_budget_inr=risk_budget,
        stop_pct=stop_pct,
        binding=binding,
        below_min_notional=notional < min_notional_inr,
    )
