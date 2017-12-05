#include <ceres/ceres.h>
#include <glog/logging.h>

using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct F1 {
    template <typename T>
    bool operator()(const T* const x1, const T* const x2, T* residual) const {
        residual[0] = x1[0] + T(10.0) * x2[0];
        return true;
    }
};

struct F2 {
    template <typename T>
    bool operator()(const T* const x3, const T* const x4, T* residual) const {
        residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
        return true;
    }
};

struct F3 {
    template <typename T>
    bool operator()(const T* const x2, const T* const x3, T* residual) const {
        residual[0] = (x2[0] - T(2.0) * x3[0]) * (x2[0] - T(2.0) * x3[0]);
        return true;
    }
};

struct F4 {
    template <typename T>
    bool operator()(const T* const x1, const T* x4, T* residual) const {
        residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
        return true;
    }
};

void solve_powells_function() {
    double x1 = 3.0;
    double x2 = -1.0;
    double x3 = 0.0;
    double x4 = 1.0;

    Problem problem;
    problem.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2);
    problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4);
    problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3);
    problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4);

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    cout << "Powell's function optimization result:" << endl;
    cout << summary.FullReport() << endl;
    cout << "x1 = " << x1 << "\t x2 = " << x2 << "\t x3 = " << x3 << "\t x4 = " << x4 << endl;
    cout << endl;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    solve_powells_function();
}
