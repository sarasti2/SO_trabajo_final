import csv
import os

from montecarlo_parallel_pool import run_parallel
from montecarlo_sequential import run_sequential


def run_multiple_experiments(portfolio, initial_iterations, max_iterations, step, results_file, num_experiments):
    """Run experiments with increasing iterations and save results to a single file."""
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Header for the CSV file
        writer.writerow([
            "Experiment", "Method", "Threads", "Ticker", 
            "Iterations", "Mean Price", "Std Dev", "Profit (%)", 
            "Conf Interval Lower", "Conf Interval Upper", 
            "Elapsed Time (s)", "Memory Used (MB)", "CPU Used (%)"
        ])

        current_iterations = initial_iterations
        experiment_count = 1

        while current_iterations <= max_iterations:
            print(f"\nRunning experiments for {current_iterations} iterations...")

            for experiment in range(1, num_experiments + 1):
                print(f"\nExperiment {experiment}/{num_experiments}, Iterations: {current_iterations}")

                # Sequential execution
                print("Sequential execution:")
                for ticker in portfolio:
                    results = run_sequential(
                        portfolio=[ticker],
                        iterations=current_iterations,
                        results_dir=None,  # No need for separate directories
                        case=f"exp{experiment}_iter{current_iterations}",
                        mode="sequential"
                    )
                    for result in results:
                        writer.writerow(
                            [experiment, "Sequential", 1, ticker, current_iterations] + result
                        )

                # Parallel execution
                print("\nParallel execution:")
                for threads in [1, 2, 4]:  # Test with 1, 2, and 4 threads
                    print(f"Using {threads} threads:")
                    for ticker in portfolio:
                        results = run_parallel(
                            portfolio=[ticker],
                            iterations=current_iterations,
                            threads=threads,
                            results_dir=None,
                            case=f"exp{experiment}_iter{current_iterations}"
                        )
                        for result in results:
                            writer.writerow(
                                [experiment, "Parallel", threads, ticker, current_iterations] + result
                            )

            # Increase iterations for the next round
            current_iterations += step
            experiment_count += 1


if __name__ == "__main__":
    portfolio = ["AAPL", "MSFT", "GOOGL"]  # Example tickers
    initial_iterations = 10_000
    max_iterations = 2_000_000
    step = 100_000  # Increment step
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"incremental_experiments.csv")
    run_multiple_experiments(
        portfolio=portfolio,
        initial_iterations=initial_iterations,
        max_iterations=max_iterations,
        step=step,
        results_file=results_file,
        num_experiments=1  # Number of experiments per iteration
    )
