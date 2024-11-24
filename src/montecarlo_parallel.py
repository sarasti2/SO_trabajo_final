import csv
import os
import threading
import time
from multiprocessing import Pool

import numpy as np
import psutil
import yfinance as yf
from scipy.stats import norm


class ResourceMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()
        self.cpu_usage = []  # Lista para almacenar el uso de CPU a lo largo del tiempo
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.start()
        return self

    def _monitor_cpu(self):
        """Monitoriza el uso de CPU en intervalos regulares mientras se ejecuta."""
        while self.monitoring:
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))  # Intervalo de 100ms
            time.sleep(0.1)

    def __exit__(self, exc_type, exc_value, traceback):
        self.monitoring = False
        self.monitor_thread.join()
        self.end_memory = self.process.memory_info().rss
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.used_memory = (self.end_memory - self.start_memory) / 1024**2
        self.avg_cpu_usage = np.mean(self.cpu_usage) if self.cpu_usage else 0

    def get_metrics(self):
        return self.elapsed_time, self.used_memory, self.avg_cpu_usage


def fetch_stock_data(ticker, period="1y"):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data["Close"]


def calculate_daily_returns(prices):
    """Calculate daily returns from historical prices."""
    return prices.pct_change().dropna()


def montecarlo_task(daily_returns, start_price):
    """Simulate a single Monte Carlo run."""
    random_returns = np.random.choice(daily_returns, size=len(daily_returns))
    return start_price * np.cumprod(1 + random_returns)[-1]


def simulate_prices_parallel(start_price, daily_returns, iterations, threads):
    """Simulate future prices using Monte Carlo in parallel."""
    with Pool(threads) as pool:
        results = pool.starmap(montecarlo_task, [(daily_returns, start_price)] * iterations)

    mean_price = np.mean(results)
    std_dev = np.std(results)

    # Intervalos de confianza
    z = 1.96  # Para un intervalo de confianza del 95%
    margin_error = z * (std_dev / np.sqrt(iterations))
    conf_interval = (mean_price - margin_error, mean_price + margin_error)

    return mean_price, std_dev, conf_interval


def run_parallel(portfolio, iterations, threads, results_dir, case):
    all_results = []

    with ResourceMonitor() as monitor:
        for ticker in portfolio:
            print(f"\nFetching data for {ticker}...")
            prices = fetch_stock_data(ticker)
            daily_returns = calculate_daily_returns(prices)
            start_price = prices.iloc[-1]

            print(f"Running parallel simulation for {ticker} with {threads} threads...")
            mean_price, std_dev, conf_interval = simulate_prices_parallel(
                start_price, daily_returns, iterations, threads
            )
            profit_pct = ((mean_price - start_price) / start_price) * 100

            all_results.append([
                ticker,
                mean_price,
                std_dev,
                profit_pct,
                conf_interval[0],
                conf_interval[1],
            ])

    elapsed_time, used_memory, avg_cpu_usage = monitor.get_metrics()

    # Añadir métricas al final de cada resultado
    for result in all_results:
        result.extend([elapsed_time, used_memory, avg_cpu_usage])

    return all_results

