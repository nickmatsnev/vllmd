from time import time
from vllmd.models import Worker


class WorkerRegistry:
    def __init__(self):
        self.workers: dict[str, Worker] = {}

    def register(self, worker: Worker):
        self.workers[worker.id] = worker

    def heartbeat(self, worker_id: str):
        worker = self.workers.get(worker_id)

        if worker:
            worker.last_seen = time()
            worker.healthy = True

    def remove_dead_workers(self, timeout=15):
        now = time()

        for worker in self.workers.values():
            if now - worker.last_seen > timeout:
                worker.healthy = False

    def healthy_workers(self):
        return [
            worker
            for worker in self.workers.values()
            if worker.healthy
        ]
