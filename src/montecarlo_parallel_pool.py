import os
import threading
import time
from multiprocessing import Pool, shared_memory

import numpy as np
import psutil
import yfinance as yf


class ResourceMonitor:
    def __enter__(self):
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()
        self.cpu_usage = []  # List to store CPU usage over time
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.start()
        return self

    def _monitor_cpu(self):
        """Monitor CPU usage at regular intervals."""
        while self.monitoring:
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))  # 100ms interval
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


def montecarlo_batch_shared(batch_size, start_price, shm_name, shape, dtype):
    """Simulate Monte Carlo runs for a batch using shared memory."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_daily_returns = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    random_returns = np.random.choice(shared_daily_returns, size=(batch_size, len(shared_daily_returns)))
    existing_shm.close()
    return start_price * np.prod(1 + random_returns, axis=1)


def simulate_prices_parallel(start_price, daily_returns, iterations, threads):
    """Simulate future prices using Monte Carlo in parallel with batching."""
    batch_size = iterations // threads
    extra = iterations % threads

    # Shared memory setup
    shm = shared_memory.SharedMemory(create=True, size=daily_returns.nbytes)
    shm_array = np.ndarray(daily_returns.shape, dtype=daily_returns.dtype, buffer=shm.buf)
    np.copyto(shm_array, daily_returns)

    tasks = [(batch_size, start_price, shm.name, daily_returns.shape, daily_returns.dtype)] * threads
    if extra:
        tasks.append((extra, start_price, shm.name, daily_returns.shape, daily_returns.dtype))

    with Pool(threads) as pool:
        results = np.concatenate(pool.starmap(montecarlo_batch_shared, tasks))

    # Cleanup shared memory
    shm.close()
    shm.unlink()

    # Calculate statistics
    mean_price = np.mean(results)
    std_dev = np.std(results)

    # Confidence intervals
    z = 1.96
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

    # Add metrics to the end of each result
    for result in all_results:
        result.extend([elapsed_time, used_memory, avg_cpu_usage])

    return all_results
