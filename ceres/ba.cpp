#include <iomanip>
#include <memory>
#include <string>

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

template <typename T>
bool compute_residual(const double observed_x, const double observed_y, const T *camera, const T *point, T *residuals,
                      bool enable_radial, bool reparametrized) {
    // camera[0, 1, 2] represent the rotation (angle-axis);
    // Pc = R ( Pw + c )
    T p[3];

    if (reparametrized) {
        // p = R'(p - t)
        // Translate first
        p[0] = point[0] - camera[3];
        p[1] = point[1] - camera[4];
        p[2] = point[2] - camera[5];

        // Then rotate (using the transpose of the rortation matrix)
        T neg_axis_angle[3];
        neg_axis_angle[0] = -camera[0];
        neg_axis_angle[1] = -camera[1];
        neg_axis_angle[2] = -camera[2];
        T p_cpy[3];
        p_cpy[0] = p[0];
        p_cpy[1] = p[1];
        p_cpy[2] = p[2];
        AngleAxisRotatePoint(neg_axis_angle, p_cpy, p);
    }
    else {
        // p = Rp + t
        // Rotate the point
        AngleAxisRotatePoint(camera, point, p);

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


struct TestParams {
    bool enable_radial;
    bool reparametrize;
    bool handcrafted_jacobian;

    TestParams(bool enable_radial, bool reparametrize, bool handcrafted_jacobian) : enable_radial(enable_radial),
                                                                                    reparametrize(reparametrize),
                                                                                    handcrafted_jacobian(
                                                                                            handcrafted_jacobian) {}
};

struct SnavelyCostFunction : public ceres::SizedCostFunction<2, 9, 3> {
public:
    virtual ~SnavelyCostFunction() = default;

//    virtual bool Evaluate(double const* camera, double const* point, double *residuals, double **jacobians) {
    virtual bool Evaluate(double const* const* params, double *residuals, double **jacobians) const override {

        cout << "Evaluate()" << endl;

        // TODO(andrei): Code from SnavelyReprojectionError(), do not duplicate it.

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            cout << "Computing analytical Jacobian over here!" << endl;

            // TODO(andrei): Implement analytical Jacobian, even if it means
            // doing a lot of work with Eigen matrices.
        }

        return true;
    }
};


using SummaryPtr = shared_ptr<ceres::Solver::Summary>;

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
        return compute_residual(observed_x, observed_y, camera, point, residuals, enable_radial, reparametrized);
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const TestParams& test_params
    ) {
        // Two residuals and two param blocks, with 9 (camera) and 3 (3D point) params, respectively,
        // hence the 2, 9, 3 template args.
        if (test_params.handcrafted_jacobian) {
            return new SnavelyCostFunction();
        }
        else {
            return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                    new SnavelyReprojectionError(observed_x, observed_y, test_params.enable_radial,
                                                 test_params.reparametrize));
        }
    }

    double observed_x;
    double observed_y;
    bool enable_radial;
    bool reparametrized;
};


SummaryPtr solve_almost_not_toy_ba(const TestParams &test_params) {
    // Solves a problem from the BAL dataset.

    // TODO(andrei): Use this data as a benchmark for your own TF implementation.
    // The first "ladybug" sequence from the BAL dataset.
    const string kBALLadybugSimple = "../../data/small/problem-49-7776-pre.txt";
    const string kBALProblemFpath = kBALLadybugSimple;
    // Whether to also account for the cameras' radial distortion parameter.
    const bool enable_radial = test_params.enable_radial;
    const bool reparametrize = test_params.reparametrize;

    ceres::Problem problem;
    BALProblem bal_problem;
    if (!bal_problem.LoadFile(kBALProblemFpath.c_str(), reparametrize)) {
        LOG(ERROR) << "Could not load data from file: " << kBALProblemFpath << endl;
        return nullptr;
    }

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        // Each residual depends on a 3D point and 9-param camera (calibration not assumed).
        double obs_x = bal_problem.observations()[2 * i + 0];
        double obs_y = bal_problem.observations()[2 * i + 1];
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(obs_x, obs_y, test_params);

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
    options.max_num_iterations = 100;
    auto summary = make_shared<ceres::Solver::Summary>();
    ceres::Solve(options, &problem, summary.get());
    std::cout << summary->FullReport() << endl;

    return summary;
}


int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO);

    auto handcrafted_jacobians = {true, false};
    auto reparam = {true, false};
    auto enable_radial = {true, false};

    vector<SummaryPtr> results;
    for (auto hc_jacobian : handcrafted_jacobians) {
        for (auto rep : reparam) {
            for (auto rad : enable_radial) {
                TestParams tp(rad, rep, hc_jacobian);
                results.push_back(solve_almost_not_toy_ba(tp));
            }
        }
    }

    string labels_outer[] = {"Handcrafted Jacobian", "Autodiff Jacobian"};
    string labels_inner[] = {"rep and radial", "rep no radial", "no rep and radial", "no rep no radial"};
    int idx = 0;
    for(auto result : results) {
        cout << labels_outer[idx / 4] << ", " << labels_inner[idx % 4] << ": \t"
             << result->initial_cost << " --> " << setprecision(12) << result->final_cost
             << " in [" << result->num_successful_steps << "] succesful. steps "
             << endl;
        idx += 1;
    }

    return 0;
}
