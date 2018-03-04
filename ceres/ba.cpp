#include <iomanip>
#include <memory>
#include <string>

#include <Eigen/Eigen>
#include "third_party/Sophus/sophus/so3.hpp"
#include "third_party/Sophus/sophus/se3.hpp"

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

// TODO(andrei): Very important: once you implement the analytic jacobians, use check_gradients!
// See: https://groups.google.com/forum/#!topic/ceres-solver/QMjKE7y5Q7M

// TODO(andrei): Consider experimenting with Ceres...
// Ya know, I could just implement my Jacobian in Ceres and see if it works, easily comparing it with the autodiff one,
// as well as the numerical one, all while running really quickly...

// TODO(andrei): Check out the NIST example!!

template<typename T>
void transform_point(const T* camera, const T* point_world, bool reparametrized, T* result) {
    if (reparametrized) {
        // p = R'(p - t)
        // Translate first
        result[0] = point_world[0] - camera[3];
        result[1] = point_world[1] - camera[4];
        result[2] = point_world[2] - camera[5];

        // Then rotate (using the transpose of the rotation matrix)
        T neg_axis_angle[3];
        neg_axis_angle[0] = -camera[0];
        neg_axis_angle[1] = -camera[1];
        neg_axis_angle[2] = -camera[2];
        T p_cpy[3];
        p_cpy[0] = result[0];
        p_cpy[1] = result[1];
        p_cpy[2] = result[2];
        AngleAxisRotatePoint(neg_axis_angle, p_cpy, result);
    }
    else {
        // p = Rp + t
        // Rotate the point
        AngleAxisRotatePoint(camera, point_world, result);

        // Apply the translation
        result[0] += camera[3];
        result[1] += camera[4];
        result[2] += camera[5];
    }
}

template <typename T>
bool compute_reprojection_residual(const double observed_x, const double observed_y, const T *camera, const T *point_world,
                                   T *residuals,
                                   bool enable_radial, bool reparametrized) {
    // camera[0, 1, 2] represents the rotation (angle-axis);
    // Pc = R ( Pw + c )
    T p[3];
    transform_point(camera, point_world, reparametrized, p);

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

using Matrix34d = Eigen::Matrix<double, 3, 4>;
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Vector7d = Eigen::Matrix<double, 1, 7>;

Matrix23d jac_pproj(const Eigen::Vector3d &P) {
    double Px = P[0];
    double Py = P[1];
    double Pz = P[2];

    double Pz2 = Pz * Pz;

    Matrix23d jac;
    jac << -1.0 / Pz,       0.0, Px / Pz2,
                 0.0, -1.0 / Pz, Py / Pz2;

    return jac;
};

Eigen::Matrix3d skew(Eigen::Vector3d x) {
    Eigen::Matrix3d res;
    res <<     0,  -x[2],   x[1],
            x[2],      0,  -x[0],
           -x[1],   x[0],      0;
    return res;
}



struct SnavelyCostFunction : public ceres::SizedCostFunction<2, 9, 3> {
public:
    SnavelyCostFunction(double observed_x, double observed_y, bool enable_radial, bool reparametrize) : observed_x(
            observed_x), observed_y(observed_y), enable_radial(enable_radial), reparametrize(reparametrize) {}

    virtual ~SnavelyCostFunction() = default;

    virtual bool Evaluate(double const* const* params, double *residuals, double **jacobians) const override {

        cout << "Evaluate()" << endl;
        if (this->enable_radial) {
            cout << "Radial distortion optimization not yet supported. Skipping." << endl;
            return true;
        }

        assert(this->parameter_block_sizes().size() == 2);

        // We operate on a per-residual-block basis. And since a residual block has two parameter blocks, there are two
        // entries, in the params array. In other words, this function is called for every (observation, parameter-block)
        // combination, computing the residual for one observation of one 3D point in one camera.
        const double* camera = params[0];
        const double* point = params[1];

        // Can use this if sizes are not known in advance.
        // auto sizes = this->parameter_block_sizes();

        compute_reprojection_residual(
                this->observed_x,
                this->observed_y,
                camera,
                point,
                residuals,
                this->enable_radial,
                this->reparametrize
        );

        if (jacobians != nullptr) {
            const double *jac_cam = jacobians[0];
            const double *jac_point = jacobians[1];

            double f = camera[6];

            double point_in_cam_frame_raw[3] = {0, 0, 0};
            transform_point<double>(camera, point, this->reparametrize, point_in_cam_frame_raw);

            Eigen::Vector3d point_in_cam_frame;
            point_in_cam_frame << point_in_cam_frame_raw[0], point_in_cam_frame_raw[1], point_in_cam_frame_raw[2];

            Eigen::Vector2d point_projected;
            point_projected << point_in_cam_frame[0] / point_in_cam_frame[2],
                               point_in_cam_frame[1] / point_in_cam_frame[2];

            Matrix23d J_proj_wrt_P = jac_pproj(point_in_cam_frame);
            if (jac_cam != nullptr) {
                cout << "Computing an-Jac of camera." << endl;

                // TODO(andrei): Use the absolute latest version from the Python version.
                Eigen::Matrix3d J_trans_wrt_omega = skew(point_in_cam_frame);
                Eigen::Matrix3d J_trans_wrt_nu;
                J_trans_wrt_nu.setIdentity();
                J_trans_wrt_nu *= -1;

                Eigen::Matrix<double, 3, 6> J_transform_wrt_twist;
                J_transform_wrt_twist << J_trans_wrt_omega, J_trans_wrt_nu;

                Eigen::Matrix<double, 2, 7> J;
                // TODO(andrei): Implement analytical Jacobian, even if it means
                // doing a lot of work with Eigen matrices.

                auto h1 = f * J_proj_wrt_P * J_transform_wrt_twist;
                // TODO set jacobian
                auto h3 = point_projected;
            }

            if (jac_point != nullptr) {
                cout << "Computing an-Jac of 3D point." << endl;

                Eigen::Vector3d omega;
                omega << camera[0], camera[1], camera[2];
                Eigen::Matrix3d R = Sophus::SO3d::exp(omega).matrix();

                Eigen::Matrix3d J_transform_wrt_delta_3d = R.transpose();

                auto h2 = f * J_proj_wrt_P * J_transform_wrt_delta_3d;

                // TODO set jacobian
            }
        }

        return true;
    }


    double observed_x;
    double observed_y;
    bool enable_radial;
    bool reparametrize;
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
        return compute_reprojection_residual(observed_x, observed_y, camera, point, residuals, enable_radial,
                                             reparametrized);
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const TestParams& test_params
    ) {
        // Two residuals and two param blocks, with 9 (camera) and 3 (3D point) params, respectively,
        // hence the 2, 9, 3 template args.
        if (test_params.handcrafted_jacobian) {
            return new SnavelyCostFunction(observed_x, observed_y, test_params.enable_radial,
                                           test_params.reparametrize);
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
    options.num_threads = 1;
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
