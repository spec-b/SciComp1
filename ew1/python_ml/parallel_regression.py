# parallel_regression.py

from mpi4py import MPI
from regression import fetch_data, run_regression

def main(symbols):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute symbols among processes (simple round-robin)
    my_symbols = [symbols[i] for i in range(len(symbols)) if i % size == rank]
    my_results = {}

    for symbol in my_symbols:
        try:
            data = fetch_data(symbol)
            coef, intercept, _ = run_regression(data)
            my_results[symbol] = {
                "coefficient": coef,
                "intercept": intercept,
                "status": "success"
            }
        except Exception as e:
            my_results[symbol] = {
                "error": str(e),
                "status": "failed"
            }

    # Gather all results at root process
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        # Combine results from all processes
        final_results = {}
        for proc_result in all_results:
            final_results.update(proc_result)
        print("\n=== Regression Results ===")
        for symbol, result in final_results.items():
            if result["status"] == "success":
                print(f"{symbol}: Coefficient={result['coefficient']:.4f}, Intercept={result['intercept']:.2f}")
            else:
                print(f"{symbol}: ERROR: {result['error']}")

if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:]
    if not symbols:
        print("Usage: mpirun -np 4 python parallel_regression.py SYMBOL1 SYMBOL2 ...")
    else:
        main(symbols)
