import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
from datetime import datetime, timedelta

class Portfolio:
    def __init__(self, tickers: list):
        """Initialize portfolio with tickers."""
        self.tickers = tickers
        self.data = None
        self.weights = None
        self.portfolio_value = None
        self.current_prices = None

    def fetch_portfolio_data(self, lookback_years: int, end_date=datetime.today().strftime("%Y-%m-%d")):
        """Fetch historical data and current prices from yfinance, excluding tickers with incomplete data."""
        if lookback_years not in [5, 10, 15, 20]:
            raise ValueError("Lookback period must be 5, 10, 15 or 20 years")
        
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=lookback_years * 365)
        start_date = start.strftime("%Y-%m-%d")
        
        data = yf.download(self.tickers + ["SPY"], start=start_date, end=end_date, group_by="ticker")
        
        if len(self.tickers) == 1:
            if "Close" not in data or data["Close"].isna().all():
                warnings.warn(f"No data available for {self.tickers[0]} from {start_date} to {end_date}")
                raise ValueError("No valid data for the ticker")
            returns = data["Close"].pct_change()
            returns = returns.to_frame()
            current_price = data["Close"].iloc[-1] if not pd.isna(data["Close"].iloc[-1]) else np.nan
            current_prices = np.array([current_price])
        else:
            returns = data.xs("Close", level=1, axis=1).pct_change()
            current_prices = np.array([
                data[ticker]["Close"].iloc[-1] if ticker in data and not pd.isna(data[ticker]["Close"].iloc[-1]) else np.nan
                for ticker in self.tickers
            ])
        
        valid_tickers = []
        valid_prices = []
        start_date_dt = pd.to_datetime(start_date)
        for i, ticker in enumerate(self.tickers):
            if (ticker in returns and 
                not returns[ticker].isna().all() and 
                not np.isnan(current_prices[i])):
                first_valid_date = returns[ticker].first_valid_index()
                if first_valid_date is None or first_valid_date > start_date_dt + timedelta(days=30):
                    warnings.warn(f"Insufficient data for {ticker}: data starts at {first_valid_date}, required from {start_date}")
                    continue
                valid_tickers.append(ticker)
                valid_prices.append(current_prices[i])
            else:
                warnings.warn(f"No sufficient data or current price for {ticker} from {start_date} to {end_date}")
        
        if not valid_tickers:
            raise ValueError("No valid data for any tickers")
        
        self.tickers = valid_tickers
        self.current_prices = np.array(valid_prices)
        
        weights = np.random.random(len(self.tickers))
        weights /= weights.sum()
        self.weights = weights
        self.portfolio_value = np.sum(self.current_prices * self.weights)
        
        returns = returns[valid_tickers + ["SPY"]].dropna()
        
        self.data = {
            "returns": returns[valid_tickers],
            "market_returns": returns["SPY"],
            "current_prices": self.current_prices,
            "weights": self.weights,
            "portfolio_value": self.portfolio_value,
            "tickers": self.tickers
        }

    def calculate_var(self, confidence_level: float = 0.99, time_horizon: float = 1/252, n_simulations: int = 10000):
        """Calculate baseline VaR and CVaR using Monte Carlo simulation."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        current_prices = self.data["current_prices"]
        weights = self.data["weights"]
        portfolio_value = self.data["portfolio_value"]
        
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        L = np.linalg.cholesky(cov_matrix)
        
        n_assets = len(weights)
        z = np.random.normal(0, 1, (n_simulations, n_assets))
        correlated_z = z @ L
        dt = time_horizon
        
        terminal_prices = np.zeros((n_simulations, n_assets))
        for i in range(n_assets):
            drift = (mu.iloc[i] - 0.5 * sigma.iloc[i]**2) * dt
            diffusion = sigma.iloc[i] * np.sqrt(dt) * correlated_z[:, i]
            terminal_prices[:, i] = current_prices[i] * np.exp(drift + diffusion)
        
        portfolio_values_T = terminal_prices @ weights
        returns_T = (portfolio_values_T - portfolio_value) / portfolio_value
        returns_T_sorted = np.sort(returns_T)
        
        alpha = 1 - confidence_level
        k = int(n_simulations * alpha)
        var = -returns_T_sorted[k] * portfolio_value
        cvar = -np.mean(returns_T_sorted[:k]) * portfolio_value
        
        return {"VaR": var, "CVaR": cvar}

    def scenario_analysis(self, scenario: dict, time_horizon: float = 1/252, n_simulations: int = 10000):
        """Simulate portfolio returns under user-defined scenarios with optional correlation shocks."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        current_prices = self.data["current_prices"]
        weights = self.data["weights"]
        portfolio_value = self.data["portfolio_value"]
        tickers = self.data["tickers"]
        
        # Validate scenario inputs
        for ticker, params in scenario.items():
            if ticker not in tickers and ticker != "correlation_shock":
                raise ValueError(f"Ticker {ticker} not in portfolio")
            if ticker != "correlation_shock" and isinstance(params, dict):  # Only check for ticker-specific shocks
                if "price_shock" in params and (1 + params["price_shock"]) <= 0:
                    raise ValueError(f"Price shock for {ticker} results in non-positive price")
                if "vol_shock" in params and params["vol_shock"] <= 0:
                    raise ValueError(f"Volatility shock for {ticker} must be positive")
        
        # Calculate historical parameters
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        
        # Apply shocks
        scenario_prices = current_prices.copy()
        scenario_mu = mu.copy()
        scenario_sigma = sigma.copy()
        scenario_cov_matrix = cov_matrix.copy()
        
        for ticker, params in scenario.items():
            if ticker in tickers and isinstance(params, dict):
                idx = tickers.index(ticker)
                if "price_shock" in params:
                    scenario_prices[idx] *= (1 + params["price_shock"])
                if "vol_shock" in params:
                    scenario_sigma.iloc[idx] *= params["vol_shock"]
                    # Scale covariance matrix rows/columns for this asset
                    scenario_cov_matrix.iloc[idx, :] *= params["vol_shock"]
                    scenario_cov_matrix.iloc[:, idx] *= params["vol_shock"]
                if "return_shock" in params:
                    scenario_mu.iloc[idx] += params["return_shock"]
        
        # Apply correlation shocks
        if "correlation_shock" in scenario:
            correlation_shock = scenario["correlation_shock"]
            if isinstance(correlation_shock, float):
                # Global correlation scaling (e.g., 1.2 to increase correlations by 20%)
                corr_matrix = np.corrcoef(returns, rowvar=False)
                n_assets = len(tickers)
                for i in range(n_assets):
                    for j in range(i + 1, n_assets):
                        corr_matrix[i, j] = min(corr_matrix[i, j] * correlation_shock, 1.0)  # Cap at 1
                        corr_matrix[j, i] = corr_matrix[i, j]
                # Reconstruct covariance matrix
                std_matrix = np.diag(scenario_sigma)
                scenario_cov_matrix = std_matrix @ corr_matrix @ std_matrix
            elif isinstance(correlation_shock, dict):
                # Pair-specific correlation shocks
                corr_matrix = np.corrcoef(returns, rowvar=False)
                for (ticker1, ticker2), new_corr in correlation_shock.items():
                    if ticker1 not in tickers or ticker2 not in tickers:
                        raise ValueError(f"Invalid ticker pair: ({ticker1}, {ticker2})")
                    i, j = tickers.index(ticker1), tickers.index(ticker2)
                    corr_matrix[i, j] = min(new_corr, 1.0)  # Cap at 1
                    corr_matrix[j, i] = new_corr
                std_matrix = np.diag(scenario_sigma)
                scenario_cov_matrix = std_matrix @ corr_matrix @ std_matrix
        
        # Ensure positive definite covariance matrix
        try:
            L = np.linalg.cholesky(scenario_cov_matrix)
        except np.linalg.LinAlgError:
            min_eig = np.min(np.linalg.eigvals(scenario_cov_matrix))
            if min_eig < 0:
                scenario_cov_matrix += np.eye(scenario_cov_matrix.shape[0]) * (-min_eig + 1e-6)
            L = np.linalg.cholesky(scenario_cov_matrix)
        
        # Monte Carlo simulation
        n_assets = len(weights)
        z = np.random.normal(0, 1, (n_simulations, n_assets))
        correlated_z = z @ L
        dt = time_horizon
        
        scenario_terminal_prices = np.zeros((n_simulations, n_assets))
        for i in range(n_assets):
            drift = (scenario_mu.iloc[i] - 0.5 * scenario_sigma.iloc[i]**2) * dt
            diffusion = scenario_sigma.iloc[i] * np.sqrt(dt) * correlated_z[:, i]
            scenario_terminal_prices[:, i] = scenario_prices[i] * np.exp(drift + diffusion)
        
        scenario_values = scenario_terminal_prices @ weights
        scenario_returns = (scenario_values - portfolio_value) / portfolio_value
        
        return {
            "Mean_Return": np.mean(scenario_returns) * portfolio_value,
            "Worst_Return": np.min(scenario_returns) * portfolio_value
        }

    def stress_test(self, stress: dict, confidence_level: float = 0.99, time_horizon: float = 1/252, n_simulations: int = 10000):
        """Calculate stressed VaR with optional correlation shocks."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        current_prices = self.data["current_prices"]
        weights = self.data["weights"]
        portfolio_value = self.data["portfolio_value"]
        tickers = self.data["tickers"]
        
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        
        stress_sigma = sigma.copy()
        stress_cov_matrix = cov_matrix.copy()
        
        # Apply stress conditions
        for key, value in stress.items():
            if key not in ["vol_multiplier", "use_historical_worst", "correlation_shock"]:
                raise ValueError(f"Invalid stress parameter: {key}")
            if key == "vol_multiplier":
                if value <= 0:
                    raise ValueError("Volatility multiplier must be positive")
                stress_sigma *= value
                stress_cov_matrix *= value**2
            elif key == "use_historical_worst":
                mu = returns.min() * 252
        
        # Apply correlation shock
        if "correlation_shock" in stress:
            correlation_shock = stress["correlation_shock"]
            if isinstance(correlation_shock, float):
                # Global correlation scaling (e.g., 1.2 to increase correlations by 20%)
                corr_matrix = np.corrcoef(returns, rowvar=False)
                n_assets = len(tickers)
                for i in range(n_assets):
                    for j in range(i + 1, n_assets):
                        corr_matrix[i, j] = min(corr_matrix[i, j] * correlation_shock, 1.0)
                        corr_matrix[j, i] = corr_matrix[i, j]
                std_matrix = np.diag(stress_sigma)
                stress_cov_matrix = std_matrix @ corr_matrix @ std_matrix
            elif isinstance(correlation_shock, dict):
                # Pair-specific correlation shocks
                corr_matrix = np.corrcoef(returns, rowvar=False)
                for (ticker1, ticker2), new_corr in correlation_shock.items():
                    if ticker1 not in tickers or ticker2 not in tickers:
                        raise ValueError(f"Invalid ticker pair: ({ticker1}, {ticker2})")
                    i, j = tickers.index(ticker1), tickers.index(ticker2)
                    corr_matrix[i, j] = min(new_corr, 1.0)
                    corr_matrix[j, i] = new_corr
                std_matrix = np.diag(stress_sigma)
                stress_cov_matrix = std_matrix @ corr_matrix @ std_matrix
        
        # Ensure positive definite covariance matrix
        try:
            L = np.linalg.cholesky(stress_cov_matrix)
        except np.linalg.LinAlgError:
            min_eig = np.min(np.linalg.eigvals(stress_cov_matrix))
            if min_eig < 0:
                stress_cov_matrix += np.eye(stress_cov_matrix.shape[0]) * (-min_eig + 1e-6)
            L = np.linalg.cholesky(stress_cov_matrix)
        
        # Monte Carlo simulation
        n_assets = len(weights)
        z = np.random.normal(0, 1, (n_simulations, n_assets))
        correlated_z = z @ L
        dt = time_horizon
        
        stress_terminal_prices = np.zeros((n_simulations, n_assets))
        for i in range(n_assets):
            drift = (mu.iloc[i] - 0.5 * stress_sigma.iloc[i]**2) * dt
            diffusion = stress_sigma.iloc[i] * np.sqrt(dt) * correlated_z[:, i]
            stress_terminal_prices[:, i] = current_prices[i] * np.exp(drift + diffusion)
        
        stress_values = stress_terminal_prices @ weights
        stress_returns = (stress_values - portfolio_value) / portfolio_value
        stress_returns_sorted = np.sort(stress_returns)
        
        alpha = 1 - confidence_level
        k = int(n_simulations * alpha)
        stressed_var = -stress_returns_sorted[k] * portfolio_value
        
        return {"Stressed_VaR": stressed_var}

    def calculate_beta(self, selloff_threshold: float = -0.02):
        """Calculate normal and stressed portfolio beta."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        market_returns = self.data["market_returns"]
        weights = self.data["weights"]
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        slope, _, _, _, _ = stats.linregress(market_returns, portfolio_returns)
        normal_beta = slope
        
        selloff_days = market_returns < selloff_threshold
        if selloff_days.sum() > 10:
            slope, _, _, _, _ = stats.linregress(market_returns[selloff_days], portfolio_returns[selloff_days])
            stressed_beta = slope
        else:
            stressed_beta = None
        
        return {"Normal_Beta": normal_beta, "Stressed_Beta": stressed_beta}

    def calculate_es_multiple_levels(self, confidence_levels: list = [0.95, 0.99, 0.995], time_horizon: float = 1/252, n_simulations: int = 10000):
        """Calculate Expected Shortfall (CVaR) at multiple confidence levels."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        current_prices = self.data["current_prices"]
        weights = self.data["weights"]
        portfolio_value = self.data["portfolio_value"]
        
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        L = np.linalg.cholesky(cov_matrix)
        
        n_assets = len(weights)
        z = np.random.normal(0, 1, (n_simulations, n_assets))
        correlated_z = z @ L
        dt = time_horizon
        
        terminal_prices = np.zeros((n_simulations, n_assets))
        for i in range(n_assets):
            drift = (mu.iloc[i] - 0.5 * sigma.iloc[i]**2) * dt
            diffusion = sigma.iloc[i] * np.sqrt(dt) * correlated_z[:, i]
            terminal_prices[:, i] = current_prices[i] * np.exp(drift + diffusion)
        
        portfolio_values_T = terminal_prices @ weights
        returns_T = (portfolio_values_T - portfolio_value) / portfolio_value
        returns_T_sorted = np.sort(returns_T)
        
        es_results = {}
        for cl in confidence_levels:
            alpha = 1 - cl
            k = int(n_simulations * alpha)
            es = -np.mean(returns_T_sorted[:k]) * portfolio_value
            es_results[f"ES_{cl*100}%"] = es
        
        return es_results

    def calculate_max_drawdown(self):
        """Calculate historical maximum drawdown."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        weights = self.data["weights"]
        
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative_value = (1 + portfolio_returns).cumprod()
        
        running_max = np.maximum.accumulate(cumulative_value)
        drawdowns = (running_max - cumulative_value) / running_max
        max_drawdown = np.max(drawdowns)
        
        return {"Max_Drawdown": max_drawdown * self.data["portfolio_value"]}

    def calculate_marginal_var(self, confidence_level: float = 0.99, time_horizon: float = 1/252, n_simulations: int = 10000):
        """Calculate Marginal VaR for each asset."""
        if self.data is None:
            raise ValueError("Must fetch portfolio data first")
        
        returns = self.data["returns"]
        current_prices = self.data["current_prices"]
        weights = self.data["weights"]
        portfolio_value = self.data["portfolio_value"]
        
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        L = np.linalg.cholesky(cov_matrix)
        
        n_assets = len(weights)
        z = np.random.normal(0, 1, (n_simulations, n_assets))
        correlated_z = z @ L
        dt = time_horizon
        
        terminal_prices = np.zeros((n_simulations, n_assets))
        for i in range(n_assets):
            drift = (mu.iloc[i] - 0.5 * sigma.iloc[i]**2) * dt
            diffusion = sigma.iloc[i] * np.sqrt(dt) * correlated_z[:, i]
            terminal_prices[:, i] = current_prices[i] * np.exp(drift + diffusion)
        
        portfolio_values_T = terminal_prices @ weights
        returns_T = (portfolio_values_T - portfolio_value) / portfolio_value
        returns_T_sorted = np.sort(returns_T)
        
        alpha = 1 - confidence_level
        k = int(n_simulations * alpha)
        var = -returns_T_sorted[k] * portfolio_value
        
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        marginal_vars = []
        for i in range(n_assets):
            cov_asset_portfolio = np.cov(returns.iloc[:, i], portfolio_returns)[0, 1] * 252
            mvar = weights[i] * cov_asset_portfolio / portfolio_vol * var / portfolio_value
            marginal_vars.append(mvar * portfolio_value)
        
        return {f"Marginal_VaR_{self.tickers[i]}": marginal_vars[i] for i in range(n_assets)}

    def risk_model(self, lookback_years: int = 20, confidence_level: float = 0.99, time_horizon: float = 1/252, 
                   n_simulations: int = 10000, scenario: dict = None, stress: dict = None):
        """Run comprehensive risk model for the portfolio."""
        self.fetch_portfolio_data(lookback_years)
        
        results = {
            "Portfolio_Value": self.portfolio_value,
            "Weights": dict(zip(self.tickers, self.weights)),
            "Current_Prices": dict(zip(self.tickers, self.current_prices))
        }
        
        var_results = self.calculate_var(confidence_level, time_horizon, n_simulations)
        results.update(var_results)
        
        es_results = self.calculate_es_multiple_levels(confidence_levels=[0.95, 0.99, 0.995], time_horizon=time_horizon, n_simulations=n_simulations)
        results.update(es_results)
        
        mdd_result = self.calculate_max_drawdown()
        results.update(mdd_result)
        
        mvar_result = self.calculate_marginal_var(confidence_level, time_horizon, n_simulations)
        results.update(mvar_result)
        
        if scenario:
            scenario_results = self.scenario_analysis(scenario, time_horizon, n_simulations)
            results["Scenario_Results"] = scenario_results
        
        if stress:
            stress_results = self.stress_test(stress, confidence_level, time_horizon, n_simulations)
            results.update(stress_results)
        
        beta_results = self.calculate_beta()
        results.update(beta_results)
        
        return results

if __name__ == "__main__":
    """
    tickers : list, list of tickers in a portfolio
    scenario : dict, in the format {ticker : {"price_shock/vol_shock/return_shock" : value}}.  
        To introduce correlation shocks, for eg, 
            Global correlation increase by 20%:
            scenario = {
                "AAPL": {"price_shock": -0.1},
                "MSFT": {"vol_shock": 1.5},
                "correlation_shock": 1.2
            }
            Pair-specific correlation shock (if more than 1 pair):
            scenario = {
                "AAPL": {"price_shock": -0.1},
                "MSFT": {"vol_shock": 1.5},
                "correlation_shock": {("AAPL", "MSFT"): 0.9, ("AAPL", "JPM") : 0.7}
            }
    stress : dict, in the format {"vol_multiplier/use_historical_worst" : value}
        To introduce correlation shocks, for eg,
            Global 20% correlation increase:
            stress = {
                "vol_multiplier": 2.0,
                "correlation_shock": 1.2
            }
            Stress test with specific correlation shock
            stress = {
                "vol_multiplier": 2.0,
                "correlation_shock": {("AAPL", "MSFT"): 0.9, ("AAPL", "JPM") : 0.7}
            }
    """
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "C", "WMT"]
    portfolio = Portfolio(tickers)
    scenario = {"AAPL": {"price_shock": -0.1}, "MSFT": {"vol_shock": 1.5}, "correlation_shock": {("AAPL", "MSFT"): 0.9, ("AAPL", "NVDA") : 0.7}}
    stress = {"vol_multiplier": 2.0, "correlation_shock": 1.2}
    
    try:
        results = portfolio.risk_model(lookback_years=20, scenario=scenario, stress=stress)
        
        # Formatted output for Microsoft Word
        print("Portfolio Risk Report")
        print("=" * 50)
        print(f"Date: {datetime.today().strftime('%Y-%m-%d')}")
        print(f"Portfolio Value: ${results['Portfolio_Value']:.2f}")
        print("\nTickers and Weights:")
        print("-" * 30)
        for ticker, weight in results["Weights"].items():
            print(f"{ticker:<10} | Weight: {weight:.4f}")
        print("\nCurrent Prices:")
        print("-" * 30)
        for ticker, price in results["Current_Prices"].items():
            print(f"{ticker:<10} | Price: ${price:.2f}")
        print("\nRisk Metrics:")
        print("-" * 30)
        print(f"1-day VaR (99%):         ${results['VaR']:.2f}")
        print(f"1-day CVaR (99%):        ${results['CVaR']:.2f}")
        print(f"1-day ES (95%):          ${results['ES_95.0%']:.2f}")
        print(f"1-day ES (99%):          ${results['ES_99.0%']:.2f}")
        print(f"1-day ES (99.5%):        ${results['ES_99.5%']:.2f}")
        print(f"Maximum Drawdown:        ${results['Max_Drawdown']:.2f}")
        print("\nMarginal VaR Contributions:")
        print("-" * 30)
        for ticker in results["Weights"].keys():
            print(f"{ticker:<10} | Marginal VaR: ${results[f'Marginal_VaR_{ticker}']:.2f}")
        
        # Scenario Analysis with dynamic shock description
        print("\nScenario Analysis:")
        print("-" * 30)
        scenario_desc = []
        for ticker, params in scenario.items():
            if ticker != "correlation_shock" and isinstance(params, dict):
                shocks = []
                if "price_shock" in params:
                    shocks.append(f"{ticker} {params['price_shock']*100:+.1f}% price")
                if "vol_shock" in params:
                    shocks.append(f"{ticker} {params['vol_shock']:.2f}x vol")
                if "return_shock" in params:
                    shocks.append(f"{ticker} {params['return_shock']*100:+.1f}% return")
                if shocks:
                    scenario_desc.append(", ".join(shocks))
            elif ticker == "correlation_shock":
                if isinstance(params, float):
                    scenario_desc.append(f"Global correlation {params:.2f}x")
                elif isinstance(params, dict):
                    pair_desc = [f"{t1}-{t2} corr {corr:.2f}" for (t1, t2), corr in params.items()]
                    scenario_desc.append(", ".join(pair_desc))
        print(f"Conditions: {'; '.join(scenario_desc)}")
        print(f"Mean Return:             -${-results['Scenario_Results']['Mean_Return']:.2f}")
        print(f"Worst Return:            -${-results['Scenario_Results']['Worst_Return']:.2f}")
        
        # Stress Test with dynamic stress description
        print("\nStress Test:")
        print("-" * 30)
        stress_desc = []
        for key, value in stress.items():
            if key == "vol_multiplier":
                stress_desc.append(f"{value:.2f}x volatility")
            elif key == "use_historical_worst":
                stress_desc.append("Historical worst returns")
            elif key == "correlation_shock":
                if isinstance(value, float):
                    stress_desc.append(f"Global correlation {value:.2f}x")
                elif isinstance(value, dict):
                    pair_desc = [f"{t1}-{t2} corr {corr:.2f}" for (t1, t2), corr in value.items()]
                    stress_desc.append(", ".join(pair_desc))
        print(f"Conditions: {'; '.join(stress_desc)}")
        print(f"Stressed VaR (99%):      ${results['Stressed_VaR']:.2f}")
        
        print("\nBeta Metrics:")
        print("-" * 30)
        print(f"Normal Beta:             {results['Normal_Beta']:.2f}")
        print(f"Stressed Beta:           {results['Stressed_Beta']:.2f}" if results['Stressed_Beta'] is not None else "Stressed Beta:           Insufficient selloff data")
        print("=" * 50)
    
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")