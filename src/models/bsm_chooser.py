"""
BSM-based Chooser Option Pricing via Monte Carlo and Analytic (Rubinstein) methods.

Reference: Huang, Wang & Wan (2021), "Exploration of JPMorgan Chooser Option Pricing"
           Rubinstein (1991), "Options for the Undecided"

A simple European chooser option gives the holder the right, at T1,
to choose whether it becomes a European call or put (same K, same T2).
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# GBM simulation
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    dt: float,
    n_paths: int,
    seed: Optional[int] = None,
    z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate terminal stock prices under risk-neutral GBM.

    S_T = S_0 * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    Args:
        s0: Initial stock price.
        r: Risk-free rate (annual).
        q: Dividend yield (annual).
        sigma: Volatility (annual).
        dt: Time interval (years).
        n_paths: Number of paths.
        seed: Random seed (ignored if z provided).
        z: Pre-drawn standard normals (overrides seed/n_paths).

    Returns:
        Array of terminal stock prices, shape (n_paths,).
    """
    if z is None:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    return s0 * np.exp(drift + diffusion)


# ---------------------------------------------------------------------------
# Chooser option payoff logic
# ---------------------------------------------------------------------------

def chooser_payoffs(
    s_t1: np.ndarray,
    s_t2: np.ndarray,
    k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute chooser option choices and payoffs.

    At T1: choose call if S_T1 > K, else put (paper's simplified rule).
    At T2: payoff depends on choice.

    Args:
        s_t1: Stock prices at decision date T1.
        s_t2: Stock prices at maturity T2.
        k: Strike price.

    Returns:
        (choices, payoffs):
            choices: bool array, True = call, False = put.
            payoffs: payoff at T2 for each path.
    """
    is_call = s_t1 > k
    call_payoff = np.maximum(s_t2 - k, 0.0)
    put_payoff = np.maximum(k - s_t2, 0.0)
    payoffs = np.where(is_call, call_payoff, put_payoff)
    return is_call, payoffs


def chooser_payoffs_proper(
    s_t1: np.ndarray,
    s_t2: np.ndarray,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chooser payoffs using proper BSM value comparison at T1.

    At T1 the holder compares BSM call value vs BSM put value
    for the remaining time tau = T2 - T1.

    Using put-call parity: choose call iff
        S_T1 * e^{-q*tau} > K * e^{-r*tau}
        i.e. S_T1 > K * e^{-(r-q)*tau}

    Args:
        s_t1: Stock prices at T1.
        s_t2: Stock prices at T2.
        k: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility (unused, kept for interface consistency).
        tau: T2 - T1.

    Returns:
        (choices, payoffs).
    """
    k_adj = k * np.exp(-(r - q) * tau)
    is_call = s_t1 > k_adj
    call_payoff = np.maximum(s_t2 - k, 0.0)
    put_payoff = np.maximum(k - s_t2, 0.0)
    payoffs = np.where(is_call, call_payoff, put_payoff)
    return is_call, payoffs


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------

def price_chooser_mc(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t1: float,
    t2: float,
    n_paths: int = 10000,
    seed: Optional[int] = None,
    use_proper_rule: bool = False,
) -> Dict:
    """
    Price a simple chooser option via Monte Carlo simulation.

    Two-period GBM:
        Period 1: S0 -> S_T1 over [0, T1]
        Period 2: S_T1 -> S_T2 over [T1, T2]

    Args:
        s0: Initial stock price.
        k: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        t1: Decision date (years).
        t2: Maturity date (years).
        n_paths: Number of MC paths.
        seed: Random seed.
        use_proper_rule: If True, use BSM-value comparison at T1;
                         if False, use paper's simplified S > K rule.

    Returns:
        Dict with keys: price, mean_payoff, std_payoff, se,
        call_ratio, s_t1, s_t2, choices, payoffs.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z2 = rng.standard_normal(n_paths)

    tau = t2 - t1

    # Period 1
    s_t1 = simulate_gbm_paths(s0, r, q, sigma, t1, n_paths, z=z1)

    # Period 2
    s_t2 = simulate_gbm_paths(s_t1, r, q, sigma, tau, n_paths, z=z2)

    # Payoffs
    if use_proper_rule:
        choices, payoffs = chooser_payoffs_proper(s_t1, s_t2, k, r, q, sigma, tau)
    else:
        choices, payoffs = chooser_payoffs(s_t1, s_t2, k)

    # Discount payoffs to present value
    discount = np.exp(-r * t2)
    pv_payoffs = discount * payoffs
    mean_pv = pv_payoffs.mean()
    std_pv = pv_payoffs.std()
    se = std_pv / np.sqrt(n_paths)

    return {
        "price": mean_pv,
        "mean_payoff": payoffs.mean(),
        "std_payoff": payoffs.std(),
        "se": se,
        "call_ratio": choices.mean(),
        "s_t1": s_t1,
        "s_t2": s_t2,
        "choices": choices,
        "payoffs": payoffs,
        "discount": discount,
    }


# ---------------------------------------------------------------------------
# Analytic BSM formulas
# ---------------------------------------------------------------------------

def bsm_call(s: float, k: float, r: float, q: float, sigma: float, t: float) -> float:
    """BSM European call price with continuous dividends."""
    if t <= 0:
        return max(s - k, 0.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def bsm_put(s: float, k: float, r: float, q: float, sigma: float, t: float) -> float:
    """BSM European put price with continuous dividends."""
    if t <= 0:
        return max(k - s, 0.0)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)


def rubinstein_chooser(
    s: float, k: float, r: float, q: float, sigma: float, t1: float, t2: float
) -> float:
    """
    Rubinstein (1991) closed-form simple chooser option price.

    V = S*e^{-q*T2}*N(d1) - K*e^{-r*T2}*N(d2)
        - S*e^{-q*T2}*N(-y1) + K*e^{-r*T2}*N(-y2)

    where:
        d1 = [ln(S/K) + (r-q+sigma^2/2)*T2] / (sigma*sqrt(T2))
        d2 = d1 - sigma*sqrt(T2)
        y1 = [ln(S/K) + (r-q)*T2 + (sigma^2/2)*T1] / (sigma*sqrt(T1))
        y2 = y1 - sigma*sqrt(T1)

    Equivalent to: Call(S,K,T2) + modified_put(T1).

    Args:
        s: Current stock price.
        k: Strike price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Volatility.
        t1: Decision date.
        t2: Maturity date.

    Returns:
        Chooser option price.
    """
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t2) / (sigma * np.sqrt(t2))
    d2 = d1 - sigma * np.sqrt(t2)

    y1 = (np.log(s / k) + (r - q) * t2 + 0.5 * sigma ** 2 * t1) / (sigma * np.sqrt(t1))
    y2 = y1 - sigma * np.sqrt(t1)

    chooser = (
        s * np.exp(-q * t2) * norm.cdf(d1)
        - k * np.exp(-r * t2) * norm.cdf(d2)
        - s * np.exp(-q * t2) * norm.cdf(-y1)
        + k * np.exp(-r * t2) * norm.cdf(-y2)
    )
    return chooser
