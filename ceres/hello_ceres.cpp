/*
 * Very simple function optimization using Ceres.
 *
 * We wish to optimize:
 *
 *  $$ 0.5 * (10 - x)^2 $$
 */

#include <ceres/ceres.h>
#include <glog/logging.h>

using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct CostFunctor {
    template <typename T>
    // The type parameters essentially means we can use this function both
    // for computing the cost value, as well as its analytic Jacobian. Yay!
    bool operator()(const T* const x, T* residual) const {
        *residual = T(10.0) - *x;

        // TODO(andrei): What does this bool mean?
        return true;
    }
};

struct NumericDiffCostFunctor {
    bool operator()(const double* const x, double *residual) const {
        *residual = 10.0 - *x;
        return true;
    }
};


void simple_analytical_derivative() {
    double initial_x = 5.0;
    double x = initial_x;

    Problem problem;

    CostFunction *cost_function = new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    // cost function, loss (robustification part), initial value of this residual block
    problem.AddResidualBlock(cost_function, nullptr, &x);

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    cout << "Optimization results when using analytical differentiation to estimate the Jacobian:" << endl;
    cout << summary.BriefReport() << endl;
    cout << "x :" << initial_x << " -> " << x << endl;
    cout << endl;
}

void simple_numeric_derivative() {
    double initial_x = 5.0;
    double x = initial_x;

    Problem problem;
    // NumericDiffCostFunction<CostFormula, est. method, nr. residuals, nr. params in block 0, ...>
    // The method for estimating numerical derivatives can be one of: central, forward (less accurate than central) and
    // Ridder's method (more accurate than central differences, but needs more function evaluations).
    CostFunction *cost_function = new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
            new NumericDiffCostFunctor());

    problem.AddResidualBlock(cost_function, nullptr, &x);

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    cout << "Optimization results when using numerical differentiation to estimate the Jacobian:" << endl;
    cout << summary.BriefReport() << endl;
    cout << "x: " << initial_x << " -> " << x << endl;
    cout << endl;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    simple_analytical_derivative();
    simple_numeric_derivative();

    return 0;
}

