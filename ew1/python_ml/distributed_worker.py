import time
import numpy as np

def worker(worker_id, params, iterations, staleness, data_X, data_y, num_iters=5, lr=0.001):
    n = len(data_y)
    if n == 0:
        print(f"Worker {worker_id} has no data, exiting.")
        return
    max_wait = 30  # seconds
    for iter in range(num_iters):
        start_time = time.time() 
        while True:
            min_iter = min(iterations.values())
            if (iterations[worker_id] - min_iter) <= staleness:
                break
            if time.time() - start_time > max_wait:
                print(f"Worker {worker_id} breaking wait after {max_wait} seconds")
                return  # or break out of loop
            time.sleep(0.05)
            
        # Get current params
        w = params['w']
        b = params['b']

        # Compute prediction and gradients
        y_pred = w * data_X + b
        grad_w = (2/n) * np.sum((y_pred - data_y) * data_X)
        grad_b = (2/n) * np.sum(y_pred - data_y)

        # Update params
        params['w'] = w - lr * grad_w
        params['b'] = b - lr * grad_b
        iterations[worker_id] = iter

        print(f"Worker {worker_id} Iter {iter}: w={params['w']:.4f}, b={params['b']:.4f}")

    print(f"Worker {worker_id} finished.")


