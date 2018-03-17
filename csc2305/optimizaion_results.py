from csc2305.utils import l2_norm


class OptimizationResults:
    """Logs detailed information about an optimization procedure.

    Attributes
        known_optimum: The a priori known value of the global optimum of the
                       function being optimized.
        iterates: The parameter values over time.
        values: The function values over time.
        ratios: This and quad_ratios correspond to Equation (1) from the CSC2305
                Assignment 03 handout.
        quad_ratios: See above.
    """

    def __init__(self, known_optimum, x_0, f_0, norm=l2_norm):
        self.known_optimum = known_optimum
        self.norm = norm
        self.iterates = [x_0]
        self.values = [f_0]
        self.alphas = []
        self.ratios = []
        self.quad_ratios = []

    def __getitem__(self, item):
        if item == 0:
            return self.iterates[item], self.values[item], None, None, None
        else:
            return self.iterates[item], self.values[item], self.alphas[item], self.ratios[item], self.quad_ratios[item]

    def __len__(self):
        return len(self.iterates)

    def record(self, iterate, value, alpha):
        if len(self.iterates) == 0:
            raise ValueError("record() must be called after x_0 and f_0 are "
                             "recorded.")

        previous = self.iterates[-1]
        current_gap = self.norm(iterate - self.known_optimum)
        previous_gap = self.norm(previous - self.known_optimum)

        ratio = current_gap / previous_gap
        ratio_quad = current_gap / (previous_gap ** 2)

        self.ratios.append(ratio)
        self.quad_ratios.append(ratio_quad)
        self.iterates.append(iterate)
        self.values.append(value)
        self.alphas.append(value)
