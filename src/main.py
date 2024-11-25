import csv
import os

from montecarlo_parallel import run_parallel
from montecarlo_sequential import run_sequential


def run_multiple_experiments(portfolio, iterations, results_file, case, num_experiments):
    """Ejecuta múltiples experimentos para un caso dado y guarda todos los resultados en un único archivo."""
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Cabecera del archivo CSV
        writer.writerow([
            "Experiment", "Method", "Threads", "Ticker", 
            "Mean Price", "Std Dev", "Profit (%)", 
            "Conf Interval Lower", "Conf Interval Upper", 
            "Elapsed Time (s)", "Memory Used (MB)", "CPU Used (%)"
        ])

        # Realiza 1000 experimentos
        for experiment in range(86, num_experiments + 1):
            print(f"\nRunning experiment {experiment}/{num_experiments} for case {case}...")

            # Ejecución secuencial
            print("Sequential execution:")
            for ticker in portfolio:
                results = run_sequential(
                    portfolio=[ticker],
                    iterations=iterations,
                    results_dir=None,  # Ya no se necesita
                    case=f"{case}_exp{experiment}",
                    mode="sequential"
                )
                # Escribe los resultados en el archivo único
                for result in results:
                    writer.writerow(
                        [experiment, "Sequential", 1] + result
                    )

            # Ejecución paralela
            print("\nParallel execution:")
            for threads in [1, 2, 4]:
                print(f"Using {threads} threads:")
                for ticker in portfolio:
                    results = run_parallel(
                        portfolio=[ticker],
                        iterations=iterations,
                        threads=threads,
                        results_dir=None,  # Ya no se necesita
                        case=f"{case}_exp{experiment}"
                    )
                    # Escribe los resultados en el archivo único
                    for result in results:
                        writer.writerow(
                            [experiment, "Parallel", threads] + result
                        )


if __name__ == "__main__":
    portfolio = ["AAPL", "MSFT", "GOOGL"]  # Tickers de ejemplo
    cases = {
        "large": 100_000,
    }
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)

    # Ejecuta experimentos para cada caso y guarda en un solo archivo por caso
    for case, iterations in cases.items():
        print(f"\nRunning {case} case with {iterations} iterations for portfolio: {portfolio}.")
        results_file = os.path.join(results_dir, f"{case}_experiments.csv")
        run_multiple_experiments(portfolio, iterations, results_file, case, num_experiments=256)
