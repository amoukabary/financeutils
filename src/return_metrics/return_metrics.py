from abc import ABC, abstractmethod
import pandas as pd

# fluent interface for chaining statistical computations and appending new ones
# example
# builder = StatsBuilder(data)
# stats = builder.mean().std().skew() --> should return a new builder with mean, std, skew columns


class ReturnMetrics:

    # expose only these methods (read-only) from the results DataFrame
    _proxy = {
        "head",
        "tail",
        "shape",
        "columns",
        "index",
        "dtypes",
        "T",
        "info",
        "describe",
        "to_csv",
        "to_excel",
        "to_dict",
        "to_json",
        "to_numpy",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        annualization: int = 1,
        results: pd.DataFrame | None = None,
    ):
        self.data = data
        self.annualization = annualization
        self.results = (
            results.copy(deep=True)
            if results is not None
            else pd.DataFrame(index=data.columns)
        )

        self._index_type = self.data.index.dtype

    # ------------------------------------------------------------------ #
    # Proxy read-only DataFrame attributes
    # ------------------------------------------------------------------ #

    def __getattr__(self, name):
        if name in self._proxy:
            return getattr(self.results, name)
        if name.startswith("_repr_") and hasattr(self.results, name):
            return getattr(self.results, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            "Use `.df` for full DataFrame access."
        ) from None

    def __dir__(self):
        base = set(dir(type(self))) | set(self.__dict__.keys())
        reprs = {a for a in dir(self.results) if a.startswith("_repr_")}
        return sorted(base | self._proxy | reprs)

    @property
    def df(self) -> pd.DataFrame:
        return self.results.copy(deep=True)

    # ------------------------------------------------------------------ #
    # Helper – create a new instance with an extra column
    # ------------------------------------------------------------------ #
    def _with_column(self, name: str, series: pd.Series) -> ReturnMetrics:
        new_results = self.results.copy(deep=True)
        new_results[name] = series
        return self.__class__(
            data=self.data,
            annualization=self.annualization,
            results=new_results,
        )

    def _with_columns(self, new_series: dict[str, pd.Series]) -> ReturnMetrics:
        new_results = self.results.copy(deep=True)
        for name, series in new_series.items():
            new_results[name] = series
        return self.__class__(
            data=self.data,
            annualization=self.annualization,
            results=new_results,
        )

    # ------------------------------------------------------------------ #
    # Statistical methods
    # ------------------------------------------------------------------ #

    def mean(self, name: str = "mean") -> ReturnMetrics:
        mean = self.data.mean() * self.annualization
        return self._with_column(name, mean)

    def std(self, name: str = "std") -> ReturnMetrics:
        std = self.data.std() * (self.annualization**0.5)
        return self._with_column(name, std)

    def skew(self, name: str = "skew") -> ReturnMetrics:
        skew = self.data.skew()
        return self._with_column(name, skew)

    def kurtosis(self, name: str = "kurtosis") -> ReturnMetrics:
        kurtosis = self.data.kurtosis()
        return self._with_column(name, kurtosis)

    def VaR(self, quantile: float = 0.05, name: str = "VaR") -> ReturnMetrics:
        var = self.data.quantile(quantile)
        return self._with_column(name, var)

    def CVaR(self, quantile: float = 0.05, name: str = "CVaR") -> ReturnMetrics:
        cvar = self.data.loc[self.data <= self.data.quantile(quantile)].mean()
        return self._with_column(name, cvar)

    def sharpe(
        self, risk_free_rate: float = 0.0, name: str = "sharpe"
    ) -> ReturnMetrics:
        excess_return = self.data.mean() - risk_free_rate / self.annualization
        sharpe = excess_return / self.data.std() * (self.annualization**0.5)
        return self._with_column(name, sharpe)

    def max_drawdown(
        self,
        name: str = "max_drawdown",
        details: bool = False,
        data_type: str = "returns",
    ) -> ReturnMetrics:

        if data_type == "prices":
            raise NotImplementedError(
                "max_drawdown with data_type='prices' is not implemented yet."
            )

        cumulative = (1 + self.data).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        new_data = {name: max_drawdown}

        if details:
            peak_dates = pd.Series(index=self.data.columns, dtype=self._index_type)
            through_dates = pd.Series(index=self.data.columns, dtype=self._index_type)
            recovery_dates = pd.Series(index=self.data.columns, dtype=self._index_type)
            for col in self.data.columns:
                dd_series = drawdown[col]
                trough_date = dd_series.idxmin()
                peak_date = cumulative[col][:trough_date].idxmax()
                recovery_date = (
                    cumulative[col][trough_date:]
                    .loc[cumulative[col] >= peak[col][peak_date]]
                    .first_valid_index()
                )

                peak_dates.loc[col] = peak_date
                through_dates.loc[col] = trough_date
                recovery_dates.loc[col] = recovery_date

            new_data.update(
                {
                    f"{name}_peak_date": peak_dates,
                    f"{name}_trough_date": through_dates,
                    f"{name}_recovery_date": recovery_dates,
                }
            )

        return self._with_columns(new_data)

    # some composite methods for common groups
    def basic_stats(self) -> ReturnMetrics:
        return self.mean().std().sharpe()

    def tail_risk(self, quantile: float = 0.05) -> ReturnMetrics:
        return self.VaR(quantile).CVaR(quantile)
