from scipy.optimize import OptimizeResult


class BundleAdjustmentResult:
    # TODO(andrei): Add other fields useful in benchmarking.

    def __init__(self, cost):
        self.cost = cost


def from_scipy(sres: OptimizeResult) -> BundleAdjustmentResult:
    # TODO(andrei): Implement clean adapter.
    raise NotImplementedError()

