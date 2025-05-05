from multiprocessing import Manager, Lock

class ParameterServer:
    def __init__(self, initial_params, staleness):
        self.manager = Manager()
        self.params = self.manager.dict(initial_params)
        self.iterations = self.manager.dict()  # worker_id -> iteration
        self.staleness = staleness
        self.lock = Lock()

    def get_params(self, worker_id):
        with self.lock:
            # In a real SSP, you could return a cached (stale) version.
            # Here, we return the latest for simplicity.
            return dict(self.params)

    def update_params(self, worker_id, delta):
        with self.lock:
            for k, v in delta.items():
                self.params[k] += v

    def can_proceed(self, worker_id):
        with self.lock:
            if not self.iterations:
                return True
            min_iter = min(self.iterations.values())
            return (self.iterations[worker_id] - min_iter) <= self.staleness

    def update_iteration(self, worker_id, iteration):
        with self.lock:
            self.iterations[worker_id] = iteration
