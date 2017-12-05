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

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}

    // Note that this whole method *is diffable*, thanks to the 'AngleAxisRotatePoint' helper!
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        // camera[0, 1, 2] represent the rotation (angle-axis);
        T p[3];
        // Rotate the point
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Apply the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Apply the radial distortion
        //  a) Compute the center of distortion.
        //     The sign change comes from the camera model assumed by the
        //     bundler tool for which this dataset was originally designed,
        //     which had a negative z-axis, that is, -z was in front of the
        //     camera.
        T x_center = - p[0] / p[2];
        T y_center = - p[1] / p[2];

        //  b) Apply the second and fourth order radial distortion
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = x_center * x_center + y_center * y_center;
        T r4 = r2 * r2;
        T distortion = T(1.0) + r2 * l1 + r4 * l2;

        // Compute the final projected point position
        // It seems that the system just assumes the principal point is zero?
        const T& focal = camera[6];
        T predicted_x_2d = x_center * focal * distortion;
        T predicted_y_2d = y_center * focal * distortion;

        residuals[0] = predicted_x_2d - T(observed_x);
        residuals[1] = predicted_y_2d - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        // Two residuals and two param blocks, with 9 (camera) and 3 (3D point) params, respectively,
        // hence the 2, 9, 3 template args.
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)
        ));
    }

    double observed_x;
    double observed_y;
};


void solve_almost_not_toy_ba() {
    // Solves a problem from the BAL dataset.

    // TODO(andrei): Use this data as a benchmark for your own TF implementation.
    // The first "ladybug" sequence from the BAL dataset.
    const string kBALLadybugSimple = "../../data/small/problem-49-7776-pre.txt";
    const string kBALProblemFpath = kBALLadybugSimple;

    // TODO(andrei): Check out the NIST example!!

    // Each residual depends on a 3D point and 9-param camera (calibration
    // not assumed).

    ceres::Problem problem;
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(kBALProblemFpath.c_str())) {
        LOG(ERROR) << "Could not load data from file: " << kBALProblemFpath << endl;
        return;
    }

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // TODO(andrei): Finish!
        double obs_x = bal_problem.observations()[2 * i + 0];
        double obs_y = bal_problem.observations()[2 * i + 1];
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(obs_x, obs_y);

        problem.AddResidualBlock(
                cost_function,
                nullptr,            // Regular squared loss
                bal_problem.mutable_camera_for_observation(i),
                bal_problem.mutable_point_for_observation(i)
        );
    }


    LOG(INFO) << "Finished preparing problem (" << bal_problem.num_observations() << " observations)." << endl;

    ceres::Solver::Options options;
    // TODO(andrei): What happens when we use ceres::DENSE_NORMAL_CHOLESKY?
    // TODO(andrei): We may need some sort of sparse solver implemented in TF.
    // The iterative Shur one seems the best in Ceres. No idea if it would be implementable and diffable in TensorFlow.
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
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
