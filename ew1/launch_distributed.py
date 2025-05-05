from multiprocessing import Manager, Process

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python_ml.parameter_server import ParameterServer
from python_ml.distributed_worker import worker

import numpy as np

def main():
    num_workers = 4
    staleness = 1  # S in SSP
    num_iters = 10

    # Generate synthetic data for demo
    np.random.seed(42)
    X_full = np.linspace(0, 10, 100)
    y_full = 3.0 * X_full + 5.0 + np.random.randn(100)

    # Split data among workers
    X_splits = np.array_split(X_full, num_workers)
    y_splits = np.array_split(y_full, num_workers)

    manager = Manager()

    params = manager.dict({'w': 0.0, 'b': 0.0})
    iterations = manager.dict({wid: 0 for wid in range(num_workers)})

    processes = []
    for wid in range(num_workers):
        p = Process(target=worker, args=(wid, params, iterations, staleness, X_splits[wid], y_splits[wid], num_iters))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nFinal parameters after distributed training:")
    print(dict(params))

if __name__ == "__main__":
    main()
