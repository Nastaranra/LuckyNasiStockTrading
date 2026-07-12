import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Trading Signal Research App", layout="wide")
st.title("📈 Trading Signal Research App")
st.caption("Technical regime + risk controls + news + market context + trade plan")
st.warning(
    "Educational research tool only. Signals are not financial advice and should "
    "be validated with paper trading and out-of-sample testing."
)

FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "SPY", "QQQ", "GLD", "IAU", "SLV", "TLT"
]

ETF_LIKE = {
    "SPY", "QQQ", "DIA", "IWM", "GLD", "IAU", "SLV", "TLT", "HYG",
    "LQD", "USO", "UNG", "XLE", "XLF", "XLK", "XLV", "XLP", "XLY",
    "XLU", "XLI", "XLB", "XLRE", "GDX", "GDXJ"
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def safe_num(value, default=np.nan):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_or(value, default):
    value = safe_num(value, np.nan)
    return default if not np.isfinite(value) else value


def is_intraday(interval: str) -> bool:
    return interval.endswith("m") or interval.endswith("h")


def bars_per_day(interval: str) -> int:
    mapping = {
        "1m": 390,
        "2m": 195,
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "60m": 7,
        "90m": 5,
        "1h": 7,
        "1d": 1,
    }
    return mapping.get(interval, 1)


def asset_profile(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in {"GLD", "IAU", "SGOL", "BAR"}:
        return "Gold ETF"
    if ticker in {"SLV", "SIVR"}:
        return "Silver ETF"
    if ticker in ETF_LIKE:
        return "ETF"
    return "Equity"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_all_tickers():
    tickers = []

    try:
        df = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            sep="|",
        )
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] == "N"]
        tickers.extend(df["Symbol"].astype(str).tolist())
    except Exception:
        pass

    try:
        df = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            sep="|",
        )
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] == "N"]
        tickers.extend(df["ACT Symbol"].astype(str).tolist())
    except Exception:
        pass

    clean = []
    for ticker in tickers:
        ticker = str(ticker).strip().replace(".", "-").upper()
        if (
            1 <= len(ticker) <= 6
            and "$" not in ticker
            and " " not in ticker
            and ticker != "FILE"
        ):
            clean.append(ticker)

    clean = sorted(set(clean)) or FALLBACK
    for ticker in FALLBACK:
        if ticker not in clean:
            clean.append(ticker)

    return clean, pd.DataFrame({"Ticker": clean})


def clean_yfinance_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return either (Price, Ticker) or (Ticker, Price).
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        level0 = set(map(str, df.columns.get_level_values(0)))
        level1 = set(map(str, df.columns.get_level_values(1)))
        if len(price_names & level0) >= len(price_names & level1):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    if "Date" not in df.columns:
        return pd.DataFrame()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in df.columns:
            df[column] = np.nan

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(ticker, period, interval):
    attempts = [(period, interval)]

    if is_intraday(interval):
        attempts.extend([("1mo", "15m"), ("3mo", "60m"), ("1y", "1d")])
    else:
        attempts.extend([("1y", "1d"), ("5y", "1d")])

    seen = set()
    for attempted_period, attempted_interval in attempts:
        key = (attempted_period, attempted_interval)
        if key in seen:
            continue
        seen.add(key)

        try:
            raw = yf.download(
                ticker,
                period=attempted_period,
                interval=attempted_interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                prepost=False,
            )
            cleaned = clean_yfinance_df(raw)
            if not cleaned.empty:
                cleaned.attrs["actual_period"] = attempted_period
                cleaned.attrs["actual_interval"] = attempted_interval
                return cleaned
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker):
    if asset_profile(ticker) != "Equity":
        return {}

    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "Company": info.get("longName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Market Cap": info.get("marketCap"),
            "P/E Ratio": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "Profit Margin": info.get("profitMargins"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Debt to Equity": info.get("debtToEquity"),
            "ROE": info.get("returnOnEquity"),
            "Beta": info.get("beta"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def load_news_sentiment(ticker):
    try:
        api_key = st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return pd.DataFrame(), 0.0, "Unavailable"

    try:
        today = datetime.now().date()
        start = today - timedelta(days=7)
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": str(start),
                "to": str(today),
                "token": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        news = response.json() or []
    except Exception:
        return pd.DataFrame(), 0.0, "News Error"

    positive_words = {
        "beat", "beats", "growth", "strong", "upgrade", "surge", "record",
        "profit", "higher", "bullish", "gain", "rally", "outperform",
        "raises", "increase", "partnership", "approval"
    }
    negative_words = {
        "miss", "misses", "drop", "fall", "lawsuit", "weak", "downgrade",
        "loss", "lower", "bearish", "decline", "cut", "warning",
        "investigation", "delay", "recall"
    }

    rows = []
    weighted_scores = []

    now_ts = datetime.now().timestamp()
    for item in news[:30]:
        headline = str(item.get("headline", ""))
        summary = str(item.get("summary", ""))
        tokens = set((headline + " " + summary).lower().replace(",", " ").split())

        raw_score = len(tokens & positive_words) - len(tokens & negative_words)
        item_ts = safe_num(item.get("datetime"), now_ts)
        age_days = max(0.0, (now_ts - item_ts) / 86400)
        recency_weight = math.exp(-age_days / 3.0)
        weighted_scores.append(raw_score * recency_weight)

        sentiment = (
            "Positive" if raw_score > 0
            else "Negative" if raw_score < 0
            else "Neutral"
        )
        rows.append({
            "Date": datetime.fromtimestamp(item_ts).strftime("%Y-%m-%d"),
            "Headline": headline,
            "Sentiment": sentiment,
            "Source": item.get("source"),
            "URL": item.get("url"),
        })

    if not rows:
        return pd.DataFrame(), 0.0, "No Recent News"

    score = float(np.mean(weighted_scores))
    label = (
        "Positive News" if score > 0.20
        else "Negative News" if score < -0.20
        else "Neutral News"
    )
    return pd.DataFrame(rows), score, label


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def wilder_rsi(close, length=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # A zero-loss run is genuinely strong, not missing.
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def add_adx(df, length=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    return tr, atr, plus_di, minus_di, adx


def rolling_vwap(df, interval):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"].fillna(0)

    if is_intraday(interval):
        session = df["Date"].dt.tz_localize(None).dt.date
        pv = (typical * volume).groupby(session).cumsum()
        vv = volume.groupby(session).cumsum()
        return pv / vv.replace(0, np.nan)

    # A rolling VWAP is more useful than a five-year cumulative VWAP.
    length = 20
    pv = (typical * volume).rolling(length, min_periods=5).sum()
    vv = volume.rolling(length, min_periods=5).sum()
    return pv / vv.replace(0, np.nan)


def add_indicators(df, interval):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    close = df["Close"]

    df["Return"] = close.pct_change()
    df["EMA9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()

    df["Return_5"] = close.pct_change(5)
    df["Return_20"] = close.pct_change(20)

    df["RSI"] = wilder_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    tr, atr, plus_di, minus_di, adx = add_adx(df, 14)
    df["TR"] = tr
    df["ATR"] = atr
    df["ATR_Pct"] = atr / close.replace(0, np.nan)
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    df["ADX"] = adx

    low14 = df["Low"].rolling(14, min_periods=14).min()
    high14 = df["High"].rolling(14, min_periods=14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3, min_periods=3).mean()

    df["Volume_MA20"] = df["Volume"].rolling(20, min_periods=5).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
    df["VWAP"] = rolling_vwap(df, interval)

    # Shifted levels prevent the current bar from defining its own support/resistance.
    lookback = max(20, min(60, bars_per_day(interval) if is_intraday(interval) else 30))
    prior_low = df["Low"].shift(1).rolling(lookback, min_periods=10).min()
    prior_high = df["High"].shift(1).rolling(lookback, min_periods=10).max()

    df["Support"] = prior_low
    df["Resistance"] = prior_high
    df["Distance_To_Support"] = (close - prior_low) / close.replace(0, np.nan)
    df["Distance_To_Resistance"] = (prior_high - close) / close.replace(0, np.nan)

    # Regime-adjusted realized volatility.
    annualization = math.sqrt(252 * bars_per_day(interval))
    df["Volatility"] = (
        df["Return"].rolling(20, min_periods=10).std() * annualization
    )

    # No bfill: bfill leaks future information into earlier rows.
    df = df.replace([np.inf, -np.inf], np.nan).ffill()

    required = [
        "Close", "EMA9", "EMA20", "EMA50", "RSI", "MACD",
        "MACD_Signal", "ATR", "ADX", "VWAP", "Support", "Resistance"
    ]
    return df.dropna(subset=required).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Market context and scoring
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def get_market_direction():
    try:
        spy = add_indicators(load_price_data("SPY", "1y", "1d"), "1d")
        qqq = add_indicators(load_price_data("QQQ", "1y", "1d"), "1d")

        if spy.empty or qqq.empty:
            return 0.0, "Unknown Market"

        score = 0
        for frame in (spy, qqq):
            latest = frame.iloc[-1]
            score += 1 if latest["Close"] > latest["EMA20"] else -1
            score += 1 if latest["EMA20"] > latest["EMA50"] else -1
            score += 1 if latest["MACD_Hist"] > 0 else -1

        normalized = score / 6
        if normalized >= 0.34:
            return normalized, "Bullish Market"
        if normalized <= -0.34:
            return normalized, "Bearish Market"
        return normalized, "Sideways Market"
    except Exception:
        return 0.0, "Unknown Market"


def fundamental_score_only(fundamentals):
    if not fundamentals:
        return 0.0, 0, ["Fundamentals not applicable or unavailable."]

    checks = []
    reasons = []

    metrics = {
        "P/E ratio is reasonable.": (
            safe_num(fundamentals.get("P/E Ratio")),
            lambda x: 0 < x < 45,
        ),
        "Forward P/E is reasonable.": (
            safe_num(fundamentals.get("Forward P/E")),
            lambda x: 0 < x < 45,
        ),
        "Profit margin is above 10%.": (
            safe_num(fundamentals.get("Profit Margin")),
            lambda x: x > 0.10,
        ),
        "Revenue growth is positive.": (
            safe_num(fundamentals.get("Revenue Growth")),
            lambda x: x > 0.05,
        ),
        "Debt-to-equity is manageable.": (
            safe_num(fundamentals.get("Debt to Equity")),
            lambda x: x < 220,
        ),
        "ROE is above 10%.": (
            safe_num(fundamentals.get("ROE")),
            lambda x: x > 0.10,
        ),
    }

    for reason, (value, test) in metrics.items():
        if np.isfinite(value):
            passed = bool(test(value))
            checks.append(1 if passed else -1)
            reasons.append(reason if passed else f"Not favorable: {reason}")

    if not checks:
        return 0.0, 0, ["Fundamental fields were unavailable."]

    return float(np.mean(checks)), len(checks), reasons


def technical_score_only(latest):
    close = finite_or(latest["Close"], 0)
    ema9 = finite_or(latest["EMA9"], close)
    ema20 = finite_or(latest["EMA20"], close)
    ema50 = finite_or(latest["EMA50"], close)
    rsi = finite_or(latest["RSI"], 50)
    macd_hist = finite_or(latest["MACD_Hist"], 0)
    vwap = finite_or(latest["VWAP"], close)
    volume_ratio = finite_or(latest["Volume_Ratio"], 1)
    adx = finite_or(latest["ADX"], 20)
    plus_di = finite_or(latest["Plus_DI"], 0)
    minus_di = finite_or(latest["Minus_DI"], 0)
    atr_pct = finite_or(latest["ATR_Pct"], 0.02)
    dist_support = finite_or(latest["Distance_To_Support"], 0.05)
    dist_resistance = finite_or(latest["Distance_To_Resistance"], 0.05)

    votes = []
    reasons = []

    def vote(condition, positive_reason, negative_reason):
        votes.append(1 if condition else -1)
        reasons.append(positive_reason if condition else negative_reason)

    vote(close > ema9, "Price is above EMA9.", "Price is below EMA9.")
    vote(ema9 > ema20, "EMA9 is above EMA20.", "EMA9 is below EMA20.")
    vote(ema20 > ema50, "EMA20 is above EMA50.", "EMA20 is below EMA50.")
    vote(close > vwap, "Price is above VWAP.", "Price is below VWAP.")
    vote(macd_hist > 0, "MACD momentum is positive.", "MACD momentum is negative.")
    vote(plus_di > minus_di, "+DI is above -DI.", "-DI is above +DI.")

    # RSI is scored contextually, not as “oversold = automatically bad.”
    if 45 <= rsi <= 65:
        votes.append(1)
        reasons.append("RSI is in a constructive range.")
    elif rsi >= 75:
        votes.append(-1)
        reasons.append("RSI is overbought.")
    elif rsi <= 25:
        votes.append(0)
        reasons.append("RSI is deeply oversold; reversal confirmation is required.")
    else:
        votes.append(0)
        reasons.append("RSI is neutral.")

    if volume_ratio >= 1.25:
        votes.append(1 if macd_hist > 0 else -1)
        reasons.append("Volume confirms the current momentum.")
    else:
        votes.append(0)
        reasons.append("Volume confirmation is limited.")

    if adx >= 25:
        votes.append(1 if plus_di > minus_di else -1)
        reasons.append("ADX indicates a meaningful trend.")
    else:
        votes.append(0)
        reasons.append("ADX indicates a weak or range-bound trend.")

    # Reward-location vote.
    if dist_support <= 0.012 and dist_resistance >= 0.025:
        votes.append(1)
        reasons.append("Price is close to support with room to resistance.")
    elif dist_resistance <= 0.012:
        votes.append(-1)
        reasons.append("Price is close to resistance.")
    else:
        votes.append(0)
        reasons.append("Price is between major support and resistance.")

    normalized_score = float(np.mean(votes)) if votes else 0.0

    if atr_pct >= 0.045:
        risk = "High"
    elif atr_pct >= 0.025:
        risk = "Medium"
    else:
        risk = "Low"

    return normalized_score, risk, reasons


def estimate_expected_return(df, horizon_bars):
    """
    Conservative, explainable estimate:
    trend drift + recent momentum, capped by ATR-based uncertainty.

    This is not presented as a price prediction model. It is a scenario estimate.
    """
    if len(df) < 60:
        return pd.DataFrame(), 0.0, "Insufficient Data"

    latest = df.iloc[-1]
    close = finite_or(latest["Close"], 0)
    atr_pct = finite_or(latest["ATR_Pct"], 0.02)

    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    recent = log_returns.tail(min(60, len(log_returns)))

    robust_drift = float(recent.median())
    momentum_5 = finite_or(latest["Return_5"], 0) / 5
    momentum_20 = finite_or(latest["Return_20"], 0) / 20

    per_bar_edge = 0.45 * robust_drift + 0.35 * momentum_5 + 0.20 * momentum_20
    raw_expected = math.exp(per_bar_edge * horizon_bars) - 1

    uncertainty = atr_pct * math.sqrt(max(horizon_bars, 1))
    cap = max(0.01, 1.5 * uncertainty)
    expected_return = float(np.clip(raw_expected, -cap, cap))

    base_price = close * (1 + expected_return)
    bull_price = close * (1 + expected_return + uncertainty)
    bear_price = close * (1 + expected_return - uncertainty)

    if expected_return >= 0.02:
        label = "Positive Scenario"
    elif expected_return <= -0.02:
        label = "Negative Scenario"
    else:
        label = "Neutral Scenario"

    out = pd.DataFrame({
        "Forecast Horizon": [f"{horizon_bars} bars"],
        "Current Price": [round(close, 2)],
        "Base Scenario": [round(base_price, 2)],
        "Bull Scenario": [round(bull_price, 2)],
        "Bear Scenario": [round(bear_price, 2)],
        "Estimated Return": [f"{expected_return:.2%}"],
        "Scenario Label": [label],
    })
    return out, expected_return, label


def score_asset(
    latest,
    fundamentals,
    expected_return,
    news_score,
    market_value,
    market_label,
    profile,
):
    tech_score, risk, tech_reasons = technical_score_only(latest)
    fund_score, fund_count, fund_reasons = fundamental_score_only(fundamentals)

    # Components are normalized to [-1, 1].
    forecast_score = float(np.clip(expected_return / 0.03, -1, 1))
    news_component = float(np.clip(news_score / 0.50, -1, 1))

    # Broad equity market direction is less relevant to gold ETFs.
    market_weight = 0.10 if profile == "Gold ETF" else 0.25
    fundamental_weight = 0.0 if profile != "Equity" or fund_count == 0 else 0.15
    news_weight = 0.10
    forecast_weight = 0.20
    technical_weight = 1.0 - (
        market_weight + fundamental_weight + news_weight + forecast_weight
    )

    weighted_score = (
        technical_weight * tech_score
        + fundamental_weight * fund_score
        + forecast_weight * forecast_score
        + news_weight * news_component
        + market_weight * float(np.clip(market_value, -1, 1))
    )

    close = finite_or(latest["Close"], 0)
    support = finite_or(latest["Support"], close)
    resistance = finite_or(latest["Resistance"], close)
    atr = finite_or(latest["ATR"], close * 0.02)

    long_reward = max(resistance - close, 0)
    long_risk = max(close - (support - 0.25 * atr), 0.01)
    reward_risk = long_reward / long_risk

    # Negative expectation cannot produce a Buy signal.
    if risk == "High":
        signal = "Avoid / High Risk"
    elif expected_return < -0.005 and weighted_score < 0.20:
        signal = "Sell / High Caution"
    elif weighted_score >= 0.45 and expected_return > 0.002 and reward_risk >= 1.4:
        signal = "Strong Buy"
    elif weighted_score >= 0.25 and expected_return > 0 and reward_risk >= 1.2:
        signal = "Buy Signal"
    elif weighted_score >= 0.10 and expected_return >= -0.002:
        signal = "Watch for Entry"
    elif weighted_score <= -0.25:
        signal = "Sell / High Caution"
    else:
        signal = "Hold / Wait"

    reasons = tech_reasons + fund_reasons
    reasons.extend([
        f"Expected-return component: {forecast_score:.2f}.",
        f"News component: {news_component:.2f}.",
        f"Market context: {market_label}.",
        f"Estimated upside/downside ratio: {reward_risk:.2f}.",
    ])

    return {
        "technical_score": tech_score,
        "fundamental_score": fund_score,
        "forecast_score": forecast_score,
        "news_score_component": news_component,
        "market_score_component": float(market_value),
        "weighted_score": weighted_score,
        "reward_risk": reward_risk,
        "risk": risk,
        "signal": signal,
        "reasons": reasons,
    }


def confidence_score(scores, expected_return, latest, data_rows):
    """
    Confidence measures signal agreement and data quality.
    It is not a calibrated probability of profit.
    """
    components = np.array([
        scores["technical_score"],
        scores["forecast_score"],
        scores["news_score_component"],
        scores["market_score_component"],
    ], dtype=float)

    direction = np.sign(scores["weighted_score"])
    if direction == 0:
        agreement = 0.0
    else:
        agreement = float(np.mean(np.sign(components) == direction))

    strength = min(abs(scores["weighted_score"]), 1.0)
    adx = finite_or(latest["ADX"], 20)
    trend_quality = float(np.clip((adx - 15) / 25, 0, 1))
    data_quality = float(np.clip(data_rows / 250, 0.4, 1.0))

    confidence = (
        35
        + 25 * strength
        + 15 * agreement
        + 10 * trend_quality
        + 10 * data_quality
    )

    if scores["risk"] == "High":
        confidence -= 15
    if abs(expected_return) < 0.002:
        confidence -= 7
    if scores["signal"] in {"Hold / Wait", "Watch for Entry"}:
        confidence = min(confidence, 69)

    return int(np.clip(round(confidence), 30, 90))


def trade_plan(latest, signal, confidence, expected_return, horizon_label):
    close = finite_or(latest["Close"], 0)
    atr = finite_or(latest["ATR"], close * 0.02)
    support = finite_or(latest["Support"], close - atr)
    resistance = finite_or(latest["Resistance"], close + atr)

    if signal in {"Strong Buy", "Buy Signal"}:
        entry_low = max(support, close - 0.50 * atr)
        entry_high = min(close + 0.10 * atr, resistance)
        stop = min(support - 0.25 * atr, entry_low - 0.75 * atr)
        target = max(resistance, close + 1.50 * atr)
        action = "LONG SETUP"
    elif signal == "Watch for Entry":
        entry_low = max(support, close - 0.80 * atr)
        entry_high = close - 0.15 * atr
        stop = support - 0.35 * atr
        target = max(resistance, close + 1.25 * atr)
        action = "WAIT FOR PULLBACK"
    elif "Sell" in signal or "Avoid" in signal:
        entry_low = np.nan
        entry_high = np.nan
        stop = close + atr
        target = min(support, close - 1.25 * atr)
        action = "NO NEW LONG / EXIT REVIEW"
    else:
        # A Hold/Wait signal should not display a fake buy zone.
        entry_low = np.nan
        entry_high = np.nan
        stop = np.nan
        target = np.nan
        action = "NO TRADE — WAIT"

    risk_amount = (
        entry_high - stop
        if np.isfinite(entry_high) and np.isfinite(stop)
        else np.nan
    )
    reward_amount = (
        target - entry_high
        if np.isfinite(entry_high) and np.isfinite(target)
        else np.nan
    )
    rr = (
        reward_amount / risk_amount
        if np.isfinite(risk_amount) and risk_amount > 0
        else np.nan
    )

    return pd.DataFrame({
        "Action": [action],
        "Entry Low": [round(entry_low, 2) if np.isfinite(entry_low) else None],
        "Entry High": [round(entry_high, 2) if np.isfinite(entry_high) else None],
        "Target": [round(target, 2) if np.isfinite(target) else None],
        "Stop Loss": [round(stop, 2) if np.isfinite(stop) else None],
        "Risk/Reward": [round(rr, 2) if np.isfinite(rr) else None],
        "Expected Hold": [horizon_label],
        "Confidence*": [f"{confidence}%"],
        "Estimated Return": [f"{expected_return:.2%}"],
    })


def make_price_chart(df, ticker, title_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["EMA9"], mode="lines", name="EMA9"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["EMA50"], mode="lines", name="EMA50"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Support"], mode="lines", name="Support",
        line={"dash": "dot"}
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Resistance"], mode="lines", name="Resistance",
        line={"dash": "dot"}
    ))
    fig.update_layout(
        title=f"{ticker} Price + Regime {title_suffix}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    return fig


def scanner_score(ticker, period, interval, market_value, market_label):
    df = add_indicators(load_price_data(ticker, period, interval), interval)
    if df.empty:
        return None

    latest = df.iloc[-1]
    profile = asset_profile(ticker)

    # Scanner uses a fast path. Expensive fundamentals/news calls are omitted.
    fundamentals = {}
    news_score = 0.0
    _, expected_return, _ = estimate_expected_return(df, 5)

    scores = score_asset(
        latest,
        fundamentals,
        expected_return,
        news_score,
        market_value,
        market_label,
        profile,
    )
    confidence = confidence_score(scores, expected_return, latest, len(df))

    return {
        "Ticker": ticker,
        "Asset": profile,
        "Price": round(finite_or(latest["Close"], 0), 2),
        "Signal": scores["signal"],
        "Confidence*": confidence,
        "Risk": scores["risk"],
        "Expected Return %": round(expected_return * 100, 2),
        "Score": round(scores["weighted_score"], 3),
        "R/R": round(scores["reward_risk"], 2),
        "RSI": round(finite_or(latest["RSI"], 50), 1),
        "ADX": round(finite_or(latest["ADX"], 20), 1),
    }


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
tickers, ticker_df = get_all_tickers()
market_value, market_label = get_market_direction()

st.sidebar.header("Settings")
mode = st.sidebar.selectbox(
    "Trading mode",
    ["Intraday / Short-term", "Swing / Multi-day", "Historical / Long-term"],
    index=0,
)

if mode == "Intraday / Short-term":
    period, interval = "5d", "5m"
    horizon_options = [3, 6, 12, 24, 39]
    default_horizon = 12
elif mode == "Swing / Multi-day":
    period, interval = "1y", "1d"
    horizon_options = [1, 3, 5, 10, 14]
    default_horizon = 5
else:
    period, interval = "5y", "1d"
    horizon_options = [20, 60, 90, 120, 180]
    default_horizon = 60

horizon_bars = st.sidebar.selectbox(
    "Forecast horizon (bars)",
    horizon_options,
    index=horizon_options.index(default_horizon),
)

scan_count = st.sidebar.selectbox(
    "How many tickers to scan?",
    [10, 25, 50, 100],
    index=1,
)
selected_ticker = st.sidebar.selectbox(
    "Select ticker",
    tickers,
    index=tickers.index("GLD") if "GLD" in tickers else 0,
)

st.sidebar.write(f"Available tickers: {len(tickers):,}")
st.sidebar.write(f"Market context: {market_label}")
st.sidebar.caption(
    "Large scans are intentionally limited because free market-data endpoints "
    "are rate-limited."
)

tab1, tab2, tab3 = st.tabs(["Single Asset", "Fast Scanner", "Ticker List"])


# -----------------------------------------------------------------------------
# Single asset
# -----------------------------------------------------------------------------
with tab1:
    raw_df = load_price_data(selected_ticker, period, interval)

    if raw_df.empty:
        st.error("Price data could not be loaded. Try another mode or ticker.")
    else:
        actual_interval = raw_df.attrs.get("actual_interval", interval)
        df = add_indicators(raw_df, actual_interval)

        if df.empty:
            st.error("Not enough clean observations to calculate the indicators.")
        else:
            latest = df.iloc[-1]
            profile = asset_profile(selected_ticker)
            fundamentals = load_fundamentals(selected_ticker)
            news_df, news_score, news_label = load_news_sentiment(selected_ticker)

            estimate_df, expected_return, forecast_label = estimate_expected_return(
                df, horizon_bars
            )
            scores = score_asset(
                latest,
                fundamentals,
                expected_return,
                news_score,
                market_value,
                market_label,
                profile,
            )
            confidence = confidence_score(
                scores, expected_return, latest, len(df)
            )

            if is_intraday(actual_interval):
                expected_hold = f"{horizon_bars} bars / same day preferred"
            elif horizon_bars <= 5:
                expected_hold = "1–5 trading days"
            elif horizon_bars <= 20:
                expected_hold = "1–4 weeks"
            else:
                expected_hold = "1–6 months"

            plan = trade_plan(
                latest,
                scores["signal"],
                confidence,
                expected_return,
                expected_hold,
            )

            st.subheader(f"{selected_ticker} — {profile}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Price", f"${latest['Close']:.2f}")
            c2.metric("Signal", scores["signal"])
            c3.metric("Confidence*", f"{confidence}%")
            c4.metric("Risk", scores["risk"])
            c5.metric("RSI", f"{latest['RSI']:.1f}")
            c6.metric("ADX", f"{latest['ADX']:.1f}")

            st.caption(
                "*Confidence is a model-agreement score, not a calibrated probability "
                "of profit."
            )

            if raw_df.attrs.get("actual_interval") != interval:
                st.warning(
                    f"Requested {interval} data was unavailable; the app used "
                    f"{raw_df.attrs.get('actual_interval')} data instead."
                )

            st.markdown("### Trade Plan")
            st.dataframe(plan, use_container_width=True, hide_index=True)

            if scores["signal"] in {"Hold / Wait", "Watch for Entry"}:
                st.info(
                    "The model does not currently identify a confirmed long entry. "
                    "Wait for price/volume confirmation rather than treating Low Risk "
                    "as a Buy signal."
                )

            st.markdown("### Key Levels and Indicators")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Support", f"${latest['Support']:.2f}")
            k2.metric("Resistance", f"${latest['Resistance']:.2f}")
            k3.metric("ATR", f"${latest['ATR']:.2f}")
            k4.metric("Stoch K", f"{latest['Stoch_K']:.1f}")
            k5.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}")
            k6.metric("Model Score", f"{scores['weighted_score']:.2f}")

            st.plotly_chart(
                make_price_chart(df, selected_ticker, f"({actual_interval})"),
                use_container_width=True,
            )

            st.markdown("### Scenario Estimate")
            st.dataframe(estimate_df, use_container_width=True, hide_index=True)
            st.caption(
                "The scenario estimate is based on robust recent drift, momentum and "
                "ATR uncertainty. It is not a guaranteed price forecast."
            )

            st.markdown("### Why This Signal?")
            for reason in scores["reasons"]:
                st.write(f"- {reason}")

            st.markdown("### Fundamentals")
            if fundamentals:
                display_fundamentals = pd.DataFrame({
                    "Metric": list(fundamentals.keys()),
                    "Value": list(fundamentals.values()),
                })
                st.dataframe(
                    display_fundamentals,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Fundamental scoring is skipped for ETFs/commodities or when "
                    "reliable fields are unavailable."
                )

            st.markdown("### Recent News")
            st.write(f"News status: **{news_label}**")
            if news_df.empty:
                st.info(
                    "No news was loaded. Add FINNHUB_API_KEY to Streamlit secrets "
                    "to enable this section."
                )
            else:
                st.dataframe(news_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Scanner
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Fast Scanner")
    st.write(
        f"Scanning the first {scan_count} tickers using price/technical data. "
        "Fundamentals and news are intentionally excluded from the scanner for speed."
    )

    if st.button("Run Scanner", type="primary"):
        rows = []
        progress = st.progress(0)

        for i, ticker in enumerate(tickers[:scan_count]):
            result = scanner_score(
                ticker,
                period,
                interval,
                market_value,
                market_label,
            )
            if result is not None:
                rows.append(result)
            progress.progress((i + 1) / scan_count)

        if rows:
            scanner_df = pd.DataFrame(rows)
            order = {
                "Strong Buy": 1,
                "Buy Signal": 2,
                "Watch for Entry": 3,
                "Hold / Wait": 4,
                "Sell / High Caution": 5,
                "Avoid / High Risk": 6,
            }
            scanner_df["_sort"] = scanner_df["Signal"].map(order).fillna(99)
            scanner_df = (
                scanner_df.sort_values(
                    ["_sort", "Confidence*", "Score"],
                    ascending=[True, False, False],
                )
                .drop(columns="_sort")
                .reset_index(drop=True)
            )
            st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Scanner Results",
                scanner_df.to_csv(index=False).encode("utf-8"),
                "trading_scanner_results.csv",
                "text/csv",
            )
        else:
            st.warning("No valid scanner results were returned.")


# -----------------------------------------------------------------------------
# Ticker list
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Ticker List")
    st.write(f"Total tickers loaded: {len(ticker_df):,}")
    st.dataframe(ticker_df, use_container_width=True, hide_index=True)