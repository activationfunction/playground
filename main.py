from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger("recursive_trading_system")
DEFAULT_MODEL = "gpt-4o-mini"
TRADING_DAYS_PER_YEAR = 365
DEPENDENCY_IMPORT_ERROR: ModuleNotFoundError | None = None

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from openai import OpenAI
except ModuleNotFoundError as exc:
    DEPENDENCY_IMPORT_ERROR = exc
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    yf = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sum": sum,
    "zip": zip,
}


@dataclass(slots=True)
class AppConfig:
    symbol: str = "BTC-USD"
    days: int = 365
    iterations: int = 5
    model: str = DEFAULT_MODEL
    output_plot: Path = Path("best_equity_curve.png")


@dataclass(slots=True)
class BacktestResult:
    metrics: dict[str, float]
    equity_curve: pd.Series
    strategy_returns: pd.Series
    signals: pd.Series


@dataclass(slots=True)
class EvaluatedStrategy:
    code: str
    result: BacktestResult


@dataclass(slots=True)
class StrategyIteration:
    iteration: int
    code: str
    metrics: dict[str, float]
    improvement_pct: float
    score: float


@dataclass(slots=True)
class OptimizationSummary:
    best_iteration: StrategyIteration
    best_result: BacktestResult
    all_iterations: list[StrategyIteration] = field(default_factory=list)


class ExecutionAdapter(Protocol):
    def handle_latest_signal(
        self,
        symbol: str,
        latest_signal: int,
        latest_price: float,
        latest_timestamp: pd.Timestamp,
    ) -> None:
        """Future live or paper trading hook."""


class NoOpExecutionAdapter:
    def handle_latest_signal(
        self,
        symbol: str,
        latest_signal: int,
        latest_price: float,
        latest_timestamp: pd.Timestamp,
    ) -> None:
        LOGGER.debug(
            "Execution adapter placeholder | symbol=%s signal=%s price=%.2f timestamp=%s",
            symbol,
            latest_signal,
            latest_price,
            latest_timestamp,
        )


def ensure_dependencies() -> None:
    if DEPENDENCY_IMPORT_ERROR is None:
        return
    missing = DEPENDENCY_IMPORT_ERROR.name or "required package"
    raise RuntimeError(
        f"Missing Python dependency '{missing}'. Install the packages from requirements.txt first."
    ) from DEPENDENCY_IMPORT_ERROR


class MarketDataProvider:
    def fetch_daily_ohlcv(self, symbol: str, days: int) -> pd.DataFrame:
        history_kwargs: dict[str, Any] = {"interval": "1d", "auto_adjust": False}
        if days == 365:
            history_kwargs["period"] = "1y"
        else:
            history_kwargs["period"] = f"{max(days, 30)}d"

        history = yf.Ticker(symbol).history(**history_kwargs)
        if history.empty:
            raise ValueError(f"No market data returned for symbol '{symbol}'.")

        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        normalized = history.rename(columns=str.lower).copy()
        expected = {"open", "high", "low", "close", "volume"}
        missing = expected.difference(normalized.columns)
        if missing:
            raise ValueError(f"Market data is missing required columns: {sorted(missing)}")

        normalized = normalized.loc[:, ["open", "high", "low", "close", "volume"]]
        normalized.index = pd.to_datetime(normalized.index)
        normalized = normalized.sort_index().dropna()

        if normalized.empty:
            raise ValueError("Market data became empty after normalization.")

        return normalized


class OpenAIStrategyGenerator:
    SYSTEM_PROMPT = (
        "You are an expert quant researcher. Output only valid Python code for a single function "
        "named generate_signals(df). The input df is a pandas DataFrame with lowercase columns "
        "open, high, low, close, volume. Return a pandas Series aligned to df.index containing "
        "only -1, 0, or 1. Use only pandas and numpy operations. Do not include markdown, "
        "explanations, tests, classes, or backtesting code."
    )

    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    def generate_initial_strategy(self, symbol: str, days: int) -> str:
        prompt = (
            f"Create an initial BTC-style strategy function for {symbol} using daily candles across "
            f"roughly {days} days. It must combine momentum and RSI logic, stay simple, and manage "
            "risk with a regime or trend filter."
        )
        return self._request_code(prompt)

    def improve_strategy(self, previous_code: str, metrics: dict[str, float]) -> str:
        prompt = (
            "Previous strategy code:\n"
            f"{previous_code}\n\n"
            "Backtest metrics JSON:\n"
            f"{json.dumps(metrics, indent=2)}\n\n"
            "Improve for higher Sharpe ratio and lower max drawdown. Output ONLY the new "
            "generate_signals(df) function code."
        )
        return self._request_code(prompt)

    def repair_strategy(self, broken_code: str, error_message: str) -> str:
        prompt = (
            "The previous strategy function failed validation or execution.\n\n"
            "Broken code:\n"
            f"{broken_code}\n\n"
            f"Error:\n{error_message}\n\n"
            "Return a corrected generate_signals(df) function only. Keep the idea but fix the bug."
        )
        return self._request_code(prompt)

    def _request_code(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        cleaned = strip_code_fences(content).strip()
        if not cleaned:
            raise ValueError("LLM returned empty strategy code.")
        return cleaned


class StrategyEvaluator:
    def __init__(self, generator: OpenAIStrategyGenerator) -> None:
        self.generator = generator

    def evaluate(self, strategy_code: str, df: pd.DataFrame) -> EvaluatedStrategy:
        repaired_code = strategy_code
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                strategy_func = load_strategy_function(repaired_code)
                signals = validate_signals(strategy_func(df.copy()), df.index)
                return EvaluatedStrategy(
                    code=repaired_code,
                    result=backtest_strategy(df, signals),
                )
            except Exception as exc:  # noqa: BLE001 - surface runtime errors from generated code
                last_error = exc
                LOGGER.warning(
                    "Strategy evaluation failed on attempt %s: %s",
                    attempt + 1,
                    exc,
                )
                if attempt == 1:
                    break
                repaired_code = self.generator.repair_strategy(repaired_code, str(exc))

        raise RuntimeError(f"Unable to evaluate generated strategy: {last_error}") from last_error


class RecursiveStrategyOptimizer:
    def __init__(
        self,
        config: AppConfig,
        data_provider: MarketDataProvider,
        generator: OpenAIStrategyGenerator,
        evaluator: StrategyEvaluator,
        execution_adapter: ExecutionAdapter | None = None,
    ) -> None:
        self.config = config
        self.data_provider = data_provider
        self.generator = generator
        self.evaluator = evaluator
        self.execution_adapter = execution_adapter or NoOpExecutionAdapter()

    def run(self) -> OptimizationSummary:
        df = self.data_provider.fetch_daily_ohlcv(self.config.symbol, self.config.days)
        current_code = self.generator.generate_initial_strategy(
            symbol=self.config.symbol,
            days=self.config.days,
        )

        iterations: list[StrategyIteration] = []
        best_iteration: StrategyIteration | None = None
        best_result: BacktestResult | None = None
        previous_score: float | None = None

        for iteration_number in range(1, self.config.iterations + 1):
            evaluated = self.evaluator.evaluate(current_code, df)
            current_code = evaluated.code
            result = evaluated.result
            score = composite_score(result.metrics)
            improvement_pct = calculate_improvement(previous_score, score)

            iteration = StrategyIteration(
                iteration=iteration_number,
                code=current_code,
                metrics=result.metrics,
                improvement_pct=improvement_pct,
                score=score,
            )
            iterations.append(iteration)
            log_iteration(iteration)

            if best_iteration is None or score > best_iteration.score:
                best_iteration = iteration
                best_result = result

            latest_signal = int(result.signals.iloc[-1])
            self.execution_adapter.handle_latest_signal(
                symbol=self.config.symbol,
                latest_signal=latest_signal,
                latest_price=float(df["close"].iloc[-1]),
                latest_timestamp=df.index[-1],
            )

            previous_score = score
            if iteration_number < self.config.iterations:
                current_code = self.generator.improve_strategy(current_code, result.metrics)

        assert best_iteration is not None
        assert best_result is not None
        return OptimizationSummary(
            best_iteration=best_iteration,
            best_result=best_result,
            all_iterations=iterations,
        )


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped


def load_strategy_function(code: str):
    parsed = ast.parse(code)
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Strategy code must not contain import statements.")

    namespace = {"pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    local_vars: dict[str, Any] = {}
    exec(compile(parsed, filename="<generated_strategy>", mode="exec"), namespace, local_vars)
    strategy_func = local_vars.get("generate_signals") or namespace.get("generate_signals")
    if not callable(strategy_func):
        raise ValueError("Strategy code must define a callable generate_signals(df) function.")
    return strategy_func


def validate_signals(signals: Any, index: pd.Index) -> pd.Series:
    if isinstance(signals, pd.DataFrame):
        if signals.shape[1] != 1:
            raise ValueError("Strategy returned a DataFrame with multiple columns.")
        signals = signals.iloc[:, 0]

    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=index)

    normalized = signals.reindex(index).fillna(0.0).astype(float)
    normalized = normalized.clip(-1.0, 1.0)
    normalized = pd.Series(np.sign(normalized).astype(int), index=index, name="signal")
    return normalized


def backtest_strategy(df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
    close_returns = df["close"].pct_change().fillna(0.0)
    positions = signals.shift(1).fillna(0.0).astype(float).clip(-1.0, 1.0)
    strategy_returns = positions * close_returns
    equity_curve = (1.0 + strategy_returns).cumprod()

    volatility = float(strategy_returns.std(ddof=0))
    annual_volatility = volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        float(strategy_returns.mean()) / volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
        if volatility > 0
        else 0.0
    )
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    max_drawdown = float(drawdown.min())
    total_return = float(equity_curve.iloc[-1] - 1.0)

    periods = max(len(df), 1)
    annual_return = float(equity_curve.iloc[-1] ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)
    trade_count = int(positions.diff().abs().fillna(0.0).sum())
    win_rate = float((strategy_returns > 0).mean())

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": float(trade_count),
    }
    metrics["composite_score"] = composite_score(metrics)

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        signals=signals,
    )


def composite_score(metrics: dict[str, float]) -> float:
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    annual_return = float(metrics.get("annual_return", 0.0))
    return sharpe + (annual_return * 0.5) - (max_drawdown * 2.0)


def calculate_improvement(previous_score: float | None, current_score: float) -> float:
    if previous_score is None:
        return 0.0
    if previous_score == 0:
        return 100.0 if current_score > 0 else 0.0
    return ((current_score - previous_score) / abs(previous_score)) * 100.0


def log_iteration(iteration: StrategyIteration) -> None:
    LOGGER.info(
        "Iteration %s complete | improvement=%.2f%% | metrics=%s",
        iteration.iteration,
        iteration.improvement_pct,
        json.dumps(iteration.metrics, sort_keys=True),
    )
    LOGGER.info("Iteration %s strategy code:\n%s", iteration.iteration, iteration.code)


def plot_equity_curve(equity_curve: pd.Series, output_path: Path, symbol: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, label="Strategy Equity", linewidth=2.0)
    plt.title(f"Best Equity Curve | {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info("Saved equity curve plot to %s", output_path)


def print_summary(summary: OptimizationSummary) -> None:
    best = summary.best_iteration
    print("\n=== Best Strategy Summary ===")
    print(f"Best iteration: {best.iteration}")
    print("Metrics:")
    print(json.dumps(best.metrics, indent=2))
    print("\nBest strategy code:\n")
    print(best.code)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursive self-improving AI trading prototype for BTC-USD."
    )
    parser.add_argument("--symbol", default="BTC-USD", help="Ticker symbol to analyze.")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Daily lookback window. A value of 365 uses yfinance 1y daily data.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of recursive strategy improvement iterations.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model for generating strategy functions.",
    )
    parser.add_argument(
        "--output-plot",
        default="best_equity_curve.png",
        help="File path for the saved equity curve plot.",
    )
    return parser


def resolve_repo_output_path(raw_path: str) -> Path:
    repo_root = Path.cwd().resolve()
    candidate = Path(raw_path)
    resolved = (repo_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if repo_root not in resolved.parents and resolved != repo_root:
        raise ValueError("Output path must stay within the repository.")
    return resolved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_arg_parser().parse_args()
    ensure_dependencies()
    config = AppConfig(
        symbol=args.symbol,
        days=args.days,
        iterations=args.iterations,
        model=args.model,
        output_plot=resolve_repo_output_path(args.output_plot),
    )

    if config.days <= 0:
        raise ValueError("--days must be positive.")
    if config.iterations <= 0:
        raise ValueError("--iterations must be positive.")

    generator = OpenAIStrategyGenerator(model=config.model)
    optimizer = RecursiveStrategyOptimizer(
        config=config,
        data_provider=MarketDataProvider(),
        generator=generator,
        evaluator=StrategyEvaluator(generator=generator),
        execution_adapter=NoOpExecutionAdapter(),
    )

    summary = optimizer.run()
    plot_equity_curve(summary.best_result.equity_curve, config.output_plot, config.symbol)
    print_summary(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        LOGGER.exception("Application failed: %s", exc)
        raise
