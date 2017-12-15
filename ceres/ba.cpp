#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "BALProblem.h"

using namespace std;
using ceres::AngleAxisRotatePoint;
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


// TODO(andrei): Consider experimenting with Ceres...
// Ya know, I could just implement my Jacobian in Ceres and see if it works, easily comparing it with the autodiff one,
// as well as the numerical one, all while running really quickly...

// TODO(andrei): Check out the NIST example!!

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y, bool enable_radial, bool reparametrized)
            : observed_x(observed_x),
              observed_y(observed_y),
              enable_radial(enable_radial),
              reparametrized(reparametrized)
    {}

    // Note that this whole method *is diffable*, thanks to the 'AngleAxisRotatePoint' helper!
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        // camera[0, 1, 2] represent the rotation (angle-axis);
        // Pc = R ( Pw + c )
        // TODO(andrei): What happens when we reparametrize this?
        T p[3];

        if (reparametrized) {
            // p = R'(p - t)
            // Translate first
            p[0] -= camera[3];
            p[1] -= camera[4];
            p[2] -= camera[5];

            // Then rotate (using the transpose of the rortation matrix)
            T neg_axis_angle[3];
            neg_axis_angle[0] = camera[0];
            neg_axis_angle[1] = camera[1];
            neg_axis_angle[2] = camera[2];
            ceres::AngleAxisRotatePoint(neg_axis_angle, point, p);
        }
        else {
            throw runtime_error("NO");
            // p = Rp + t
            // Rotate the point
            ceres::AngleAxisRotatePoint(camera, point, p);

            // Apply the translation
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];
        }

        // Apply the projection
        //   The sign change comes from the camera model assumed by the
        //   bundler tool for which this dataset was originally designed,
        //   which had a negative z-axis, that is, -z was in front of the
        //   camera.
        T x_center = -p[0] / p[2];
        T y_center = -p[1] / p[2];

        T distortion = T(1.0);
        if (enable_radial) {
            // Apply the radial distortion
            //  (the second and fourth order radial distortion)
            const T &l1 = camera[7];
            const T &l2 = camera[8];
            T r2 = x_center * x_center + y_center * y_center;
            T r4 = r2 * r2;
            distortion += r2 * l1 + r4 * l2;
        }

        // Compute the final projected point position
        // It seems that the system just assumes the principal point is zero.
        // This is just the convention used by the BAL dataset.
        const T& focal = camera[6];
        T predicted_x_2d = x_center * focal * distortion;
        T predicted_y_2d = y_center * focal * distortion;

        // TODO(andrei): What happens if we group these together into a distance, and then robustify?
        // Ceres seems to not be hindered by the fact that x and y are separate when adding
        // a robust estimator. Is the SciPy solver then weird, or does Ceres do anything special?
        residuals[0] = predicted_x_2d - T(observed_x);
        residuals[1] = predicted_y_2d - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const bool enable_radial,
                                       const bool reparametrized
    ) {
        // Two residuals and two param blocks, with 9 (camera) and 3 (3D point) params, respectively,
        // hence the 2, 9, 3 template args.
        auto *res = new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y, enable_radial, reparametrized));
        return res;
    }

    double observed_x;
    double observed_y;
    bool enable_radial;
    bool reparametrized;
};


void solve_almost_not_toy_ba() {
    // Solves a problem from the BAL dataset.

    // TODO(andrei): Use this data as a benchmark for your own TF implementation.
    // The first "ladybug" sequence from the BAL dataset.
    const string kBALLadybugSimple = "../../data/small/problem-49-7776-pre.txt";
    const string kBALProblemFpath = kBALLadybugSimple;
    // Whether to also account for the cameras' radial distortion parameter.
    const bool enable_radial = false;
    const bool reparametrize = true;

    ceres::Problem problem;
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(kBALProblemFpath.c_str(), reparametrize)) {
        LOG(ERROR) << "Could not load data from file: " << kBALProblemFpath << endl;
        return;
    }

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // Each residual depends on a 3D point and 9-param camera (calibration not assumed).
        double obs_x = bal_problem.observations()[2 * i + 0];
        double obs_y = bal_problem.observations()[2 * i + 1];
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
                obs_x,
                obs_y,
                enable_radial,
                reparametrize
        );

        // Regular squared loss (no robust estimators, for simplicity).
        ceres::LossFunction* loss_function = nullptr;

        problem.AddResidualBlock(
                cost_function,
                loss_function,
                bal_problem.mutable_camera_for_observation(i),
                bal_problem.mutable_point_for_observation(i));
    }

    LOG(INFO) << "Finished preparing problem (" << bal_problem.num_observations() << " observations)." << endl;

    ceres::Solver::Options options;
    // TODO(andrei): What happens when we use ceres::DENSE_NORMAL_CHOLESKY?
    // TODO(andrei): We may need some sort of sparse solver implemented in TF.
    // The iterative Shur one seems the best in Ceres. No idea if it would be implementable and diffable in TensorFlow.
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 6;
    options.max_num_iterations = 1000;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << endl;
}


int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO);
    solve_almost_not_toy_ba();
    return 0;
}
