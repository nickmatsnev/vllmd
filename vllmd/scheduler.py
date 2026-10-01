from vllmd.registry import WorkerRegistry


class Scheduler:
    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    def choose_worker(self):
        workers = self.registry.healthy_workers()

        if not workers:
            raise RuntimeError("No healthy workers available")

        # stupid but perfectly acceptable MVP scheduler
        return min(
            workers,
            key=lambda worker: worker.active_requests,
        )
