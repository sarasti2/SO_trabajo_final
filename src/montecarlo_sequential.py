import csv
import os
import time

import numpy as np
import psutil
import yfinance as yf
from scipy.stats import norm


class ResourceMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_memory = self.process.memory_info().rss
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.used_memory = (self.end_memory - self.start_memory) / 1024**2

    def get_metrics(self):
        return self.elapsed_time, self.used_memory


def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data["Close"]


def calculate_daily_returns(prices):
    return prices.pct_change().dropna()


def simulate_prices(start_price, daily_returns, iterations):
    results = []
    for _ in range(iterations):
        random_returns = np.random.choice(daily_returns, size=len(daily_returns))
        simulated_price = start_price * np.cumprod(1 + random_returns)[-1]
        results.append(simulated_price)

    mean_price = np.mean(results)
    std_dev = np.std(results)

    # Intervalos de confianza
    z = 1.96  # 95%
    margin_error = z * (std_dev / np.sqrt(iterations))
    conf_interval = (mean_price - margin_error, mean_price + margin_error)

    return mean_price, std_dev, conf_interval


def run_sequential(portfolio, iterations, results_dir, case, mode):
    all_results = []

    with ResourceMonitor() as monitor:
        for ticker in portfolio:
            print(f"\nFetching data for {ticker}...")
            prices = fetch_stock_data(ticker)
            daily_returns = calculate_daily_returns(prices)
            start_price = prices.iloc[-1]

            print(f"Running sequential simulation for {ticker}...")
            mean_price, std_dev, conf_interval = simulate_prices(start_price, daily_returns, iterations)
            profit_pct = ((mean_price - start_price) / start_price) * 100
            all_results.append([
                ticker,
                mean_price,
                std_dev,
                profit_pct,
                conf_interval[0],
                conf_interval[1],
            ])

    elapsed_time, used_memory = monitor.get_metrics()
    cpu_usage = psutil.cpu_percent(interval=1)

    # Añadir métricas al final de cada resultado
    for result in all_results:
        result.extend([elapsed_time, used_memory, cpu_usage])

    return all_results
