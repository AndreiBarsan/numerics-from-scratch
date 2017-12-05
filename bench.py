"""Bundle Adjustment Benchmark

Compares Ceres, SciPy, custom Python, and TensorFlow nonlinear optimizers on
several problems ranging from toy function to bundle adjustments.
"""

from problem import ToyProblem, BALBundleAdjustmentProblem, Problem

from scipy_ba import solve as scipy_solve


class Result(object):
    """The result of an experiment run under some configuration."""

    def __init__(self):
        # TODO Want: learning curve over time, final optimum, time taken,
        # plus all the stuff from the default SciPy OptimizationResult.
        pass



class ExperimentConfig(object):

    def __init__(self, name, args):
        self.name = name
        self.args = args


    def run(self, problem: Problem):
        pass


class SciPyConfig(ExperimentConfig):
    """Solves problems using SciPy."""

    def __init__(self, name, args):
        ExperimentConfig.__init__(self, name, args)

    def run(self, problem: Problem):
        if type(problem) is BALBundleAdjustmentProblem:
            result = scipy_solve(problem)
        else:
            raise ValueError("Unsupported problem type for the SciPy driver.")


def set_nested(m, value, *keys):
    if len(keys) == 0:
        raise ValueError("No keys specified")
    else:
        key = keys[0]
        if key not in m:
            m[key] = {}

        rest_keys = keys[1:]
        if len(rest_keys) == 0:
            m[key] = value
        else:
            return set_nested(m[key], value, *rest_keys)


def main():
    configs = [
        SciPyConfig("SciPy-LM", None)
        # TODO(andrei): Config for my Python implementation.
        # TODO(andrei): Config for my TensorFlow implementation.
        # TODO(andrei): Config for Ceres implementation.
    ]

    problems = [
        # ToyProblem("Quadratic"),
        # At 31k observations, this is the SMALLEST problem in the dataset,
        # but still very challenging. However, if the TF implementation
        # manages to get a decent result on this in less than a few minutes,
        # I will be very happy.
        # BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
        # FILE_NAME = "problem-49-7776-pre.txt.bz2"
        BALBundleAdjustmentProblem("LadyBug", "data/small/problem-49-7776-pre.txt"),
    ]

    result_map = {}
    for config in configs:
        for problem in problems:
            print("\nEvaluating [{}] using [{}]...".format(problem.name, config.name))
            result = config.run(problem)
            set_nested(result_map, result, config.name, problem.name)

    # TODO automatic comparative analysis
    # TODO plot loss function over time for each method
    # TODO dump data as, e.g., CSV from Ceres.



if __name__ == '__main__':
    main()
