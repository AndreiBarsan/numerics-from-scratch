#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#include <Eigen/Eigen>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <fstream>

#include "BALProblem.h"
#include "CsvWriter.h"

using namespace std;
using ceres::AngleAxisRotatePoint;
using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

DEFINE_string(dataset_root, "../data",
              "The root folder where the BAL datasets are present. (See 'get_data.py' for more info.)");
DEFINE_string(problem_list,
              "trafalgar:ALL",
              "Indices of the problems to solve. The format should be 'nameA:1,2,..,n;nameB:1,2,...,m;...'. For "
                  "instance, 'trafalgar:1,3,5' runs the first, third, and fifth problems from the trafalgar set, while "
                  "'venice:ALL; trafalgar:2,5' runs all the venice sequences and sequences 2 and 5 from the trafalgar "
                  "set.");

// TODO(andreib): Basic header with info about WTF is going on.

// Problem file name structure: problem-IMAGES-POINTS-OBSERVATIONS
std::vector<std::string> kVeniceFiles = {
    "problem-52-64053-pre.txt",
    "problem-89-110973-pre.txt",
    "problem-245-198739-pre.txt",
    "problem-427-310384-pre.txt",
    "problem-744-543562-pre.txt",
    "problem-951-708276-pre.txt",
    "problem-1158-802917-pre.txt",
    "problem-1288-866452-pre.txt",
    "problem-1408-912229-pre.txt",
    "problem-1490-935273-pre.txt",
// TODO(andreib): Mention removed ones.
};

std::vector<std::string> kTrafalgarFiles = {
    "problem-21-11315-pre.txt",
    "problem-39-18060-pre.txt",
    "problem-50-20431-pre.txt",
    "problem-126-40037-pre.txt",
    "problem-138-44033-pre.txt",
    "problem-161-48126-pre.txt",
    "problem-170-49267-pre.txt",
    "problem-174-50489-pre.txt",
    "problem-193-53101-pre.txt",
    "problem-201-54427-pre.txt",
    "problem-206-54562-pre.txt",
    "problem-215-55910-pre.txt",
    "problem-225-57665-pre.txt",
    "problem-257-65132-pre.txt",
};

std::vector<std::string> kDubrovnikFiles = {
    "problem-16-22106-pre.txt",
    "problem-88-64298-pre.txt",
    "problem-135-90642-pre.txt",
    "problem-142-93602-pre.txt",
    "problem-150-95821-pre.txt",
    "problem-161-103832-pre.txt",
    "problem-173-111908-pre.txt",
    "problem-182-116770-pre.txt",
    "problem-202-132796-pre.txt",
    "problem-237-154414-pre.txt",
    "problem-253-163691-pre.txt",
    "problem-262-169354-pre.txt",
    "problem-273-176305-pre.txt",
    "problem-287-182023-pre.txt",
    "problem-308-195089-pre.txt",
    "problem-356-226730-pre.txt",
};

// I am extremely grateful for the existence of map literals in modern C++...
std::map<std::string, std::vector<std::string>> kProblemFiles = {
    {"venice", kVeniceFiles},
    {"trafalgar", kTrafalgarFiles},
    {"dubrovnik", kDubrovnikFiles},
};

template<typename T>
void transform_point(const T *camera, const T *point_world, bool reparametrized, T *result) {
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
  } else {
    // p = Rp + t
    // Rotate the point
    AngleAxisRotatePoint(camera, point_world, result);

    // Apply the translation
    result[0] += camera[3];
    result[1] += camera[4];
    result[2] += camera[5];
  }
}

template<typename T>
bool compute_reprojection_residual(const double observed_x,
                                   const double observed_y,
                                   const T *camera,
                                   const T *point_world,
                                   T *residuals,
                                   bool enable_radial,
                                   bool reparametrized) {
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
  const T &focal = camera[6];
  T predicted_x_2d = x_center * focal * distortion;
  T predicted_y_2d = y_center * focal * distortion;

  // TODO(andrei): What happens if we group these together into a distance, and then robustify?
  // Ceres seems to not be hindered by the fact that x and y are separate when adding
  // a robust estimator. Is the SciPy solver then weird, or does Ceres do anything special?
  residuals[0] = predicted_x_2d - T(observed_x);
  residuals[1] = predicted_y_2d - T(observed_y);

  return true;
}

enum StrategyType {
  LM,

  // Only experimental support in Ceres; should only be used with SPARSE_SCHUR,
  // DENSE_SCHUR, DENSE_QR, and SPARSE_NORMAL_CHOLESKY.
      DOGLEG_TRADITIONAL,

  DOGLEG_SUBSPACE,
};

struct ExperimentParams {
  const bool enable_radial;
  const bool reparametrize;

  // LINE_SEARCH or TRUST_REGION are the two major categories.
//  const ceres::MinimizerType minimizer_type;

  // DENSE_NORMAL_CHOLESY
  // DENSE_QR
  // SPARSE_NORMAL_CHOLESKY
  // DENSE_SCHUR
  // SPARSE_SCHUR
  // CGNR
  // Note: only used in trust region methods, right? Seems so.
//  const ceres::LinearSolverType solver_type;

  // Note: only used in line search methods.
  //  NONLINEAR_CONJUGATE_GRADIENT
  //  BFGS
  //  LBFGS
//  const ceres::LineSearchDirectionType line_search_type;

  // Used in line search with NONLINEAR_CONJUGATE_GRADIENT
//  const ceres::NonlinearConjugateGradientType nlcg_type;

//  const ceres::TrustRegionStrategyType tr_strategy;

//    const ceres::PreconditionerType  preconditioner_type = ...;

  const ceres::Solver::Options solver_options;

  ExperimentParams(const bool enable_radial,
                   const bool reparametrize,
                   const ceres::Solver::Options &solver_options)
      : enable_radial(enable_radial),
        reparametrize(reparametrize),
        solver_options(solver_options) {}

  /// Returns a representation of this object suitable for including as part of a file name.
  std::string get_label() const {
    // TODO(andreib): Don't bother with parsing this in detail when processing the data. The complete info should be
    // in the meta file anyway.
    using namespace ceres;
    std::stringstream out_ss;
    out_ss << MinimizerTypeToString(solver_options.minimizer_type) << "-";
    if (solver_options.minimizer_type == MinimizerType::TRUST_REGION) {
      out_ss << TrustRegionStrategyTypeToString(solver_options.trust_region_strategy_type) << "-";
      if (solver_options.trust_region_strategy_type == TrustRegionStrategyType::DOGLEG) {
        out_ss << DoglegTypeToString(solver_options.dogleg_type) << "-";
      }
      out_ss << LinearSolverTypeToString(solver_options.linear_solver_type) << "-";
    } else {
      out_ss << LineSearchDirectionTypeToString(solver_options.line_search_direction_type) << "-";
      if (solver_options.line_search_direction_type == LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT) {
        out_ss << NonlinearConjugateGradientTypeToString(solver_options.nonlinear_conjugate_gradient_type) << "-";
      }
    }

    out_ss << "params";
    return out_ss.str();
  }

  /// Returns a detailed and easy-to-parse representation of this object.
  std::string get_details() const {
    using namespace ceres;
    std::stringstream out_ss;
    out_ss << MinimizerTypeToString(solver_options.minimizer_type) << ",";
    out_ss << TrustRegionStrategyTypeToString(solver_options.trust_region_strategy_type) << ",";
    out_ss << DoglegTypeToString(solver_options.dogleg_type) << ",";
    out_ss << LinearSolverTypeToString(solver_options.linear_solver_type) << ",";
    out_ss << LineSearchDirectionTypeToString(solver_options.line_search_direction_type) << ",";
    out_ss << NonlinearConjugateGradientTypeToString(solver_options.nonlinear_conjugate_gradient_type);
    return out_ss.str();
  }
};

// TODO(andreib): Organize your configs into functions, then put all together,
// maybe based on command line flags, and run. (Each config on a subset of venice.)

ceres::TrustRegionStrategyType kTrStrategies[] = {
    ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT,
    ceres::TrustRegionStrategyType::DOGLEG
};
ceres::LinearSolverType kTrSolvers[] = {
//    ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY,
//    ceres::LinearSolverType::DENSE_QR,
//    ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY,
    ceres::LinearSolverType::DENSE_SCHUR,
    ceres::LinearSolverType::SPARSE_SCHUR,
    ceres::LinearSolverType::CGNR
};

std::vector<ExperimentParams> get_lm_configs(const ceres::Solver::Options &base_options) {
  using namespace ceres;
  std::vector<ExperimentParams> out;
  for (LinearSolverType solver : kTrSolvers) {
    ceres::Solver::Options options = base_options;
    options.minimizer_type = MinimizerType::TRUST_REGION;
    options.trust_region_strategy_type = TrustRegionStrategyType::LEVENBERG_MARQUARDT;
    options.linear_solver_type = solver;
    out.emplace_back(true, true, options);
  }
  return out;
}

using Matrix34d = Eigen::Matrix<double, 3, 4>;
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Vector7d = Eigen::Matrix<double, 1, 7>;
using SummaryPtr = shared_ptr<ceres::Solver::Summary>;

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y, bool enable_radial, bool reparametrized)
      : observed_x(observed_x),
        observed_y(observed_y),
        enable_radial(enable_radial),
        reparametrized(reparametrized) {}

  // Note that *this whole method is auto-diffable*, thanks to the 'AngleAxisRotatePoint' helper!
  template<typename T>
  bool operator()(const T *const camera, const T *const point, T *residuals) const {
    return compute_reprojection_residual(observed_x, observed_y, camera, point, residuals, enable_radial,
                                         reparametrized);
  }

  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y,
                                     const ExperimentParams &test_params
  ) {
    // Two residuals and two param blocks, with 9 (camera) and 3 (3D point) params, respectively,
    // hence the 2, 9, 3 template args.
    return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y, test_params.enable_radial,
                                     test_params.reparametrize));
  }

  double observed_x;
  double observed_y;
  bool enable_radial;
  bool reparametrized;
};

//const string kBALVeniceSimple = "../data/venice/problem-89-110973-pre.txt";
//const string kBALVeniceMed = "../data/venice/problem-427-310384-pre.txt";

SummaryPtr SolveSimpleBA(const string &data_file_fpath, const ExperimentParams &experiment_params) {
  // Solves a problem from the BAL dataset.

  // Whether to also account for the cameras' radial distortion parameter.
  const bool enable_radial = experiment_params.enable_radial;
  const bool reparametrize = experiment_params.reparametrize;

  ceres::Problem problem;
  BALProblem bal_problem;
  if (!bal_problem.LoadFile(data_file_fpath.c_str(), reparametrize)) {
    LOG(ERROR) << "Could not load data from file: " << data_file_fpath << endl;
    return nullptr;
  }

  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each residual depends on a 3D point and 9-param camera (calibration not assumed).
    double obs_x = bal_problem.observations()[2 * i + 0];
    double obs_y = bal_problem.observations()[2 * i + 1];
    ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(obs_x, obs_y, experiment_params);

    // Regular squared loss (no robust estimators, for simplicity).
    ceres::LossFunction *loss_function = nullptr;

    problem.AddResidualBlock(
        cost_function,
        loss_function,
        bal_problem.mutable_camera_for_observation(i),
        bal_problem.mutable_point_for_observation(i));
  }

  LOG(INFO) << "Finished preparing problem (" << bal_problem.num_observations() << " observations)." << endl;

  auto summary = make_shared<ceres::Solver::Summary>();
  LOG(INFO) << "Starting to solve..." << endl;
  ceres::Solve(experiment_params.solver_options, &problem, summary.get());
  LOG(INFO) << "Finished!";
  return summary;
}

// TODO(andreib): Prolly the most flexible to just dump one file per one (config, dataset) pair. We'll end up with a
// lot of files, but it's not a big deal. We can aggregate in Python OK.
void SaveResults(
    const std::string &out_dir,
    const std::string &dataset_name,
    const std::string &problem_fname,
    const ExperimentParams &params,
    const ceres::Solver::Summary &summary
) {
  using namespace ceres;
  std::stringstream fname_ss;
  fname_ss << "results-" << dataset_name << "-" << problem_fname << "-" << params.get_label();

  const std::string fname = fname_ss.str() + ".csv";
  const std::string fname_meta = fname_ss.str() + ".meta.txt";
  const std::string fname_raw = fname_ss.str() + ".raw_summary.txt";

  const std::string fpath = out_dir + "/" + fname;
  const std::string fpath_meta = out_dir + "/" + fname;
  const std::string fpath_raw = out_dir + "/" + fname;

  if (FileExists(fpath)) {
    LOG(ERROR) << "Results file [" << fpath << "] already exists. Not re-dumping." << std::endl;
    return;
  }

  LOG(INFO) << "Writing data to files:" << std::endl;
  LOG(INFO) << "\t" << fname << std::endl;
  LOG(INFO) << "\t" << fname_meta << std::endl;
  LOG(INFO) << "\t" << fname_raw << std::endl;

  // Write all metadata you can to the meta file.
  // Then, write all iterations to the "main" file.
  // Then just dump the detailed summary to the raw file.

  // Note that all iterations have an index, and a dataset name, and N_cameras, N points, etc., so you
  // can easily distinguish different scenes. What about time? We have iter_time and total_time, so we can easily
  // look at the last iteration's total time to see the method's wall time.

  ofstream out(fpath);
  ofstream out_meta(fpath_meta);
  ofstream out_raw(fpath_raw);

  // Write the header
  out << "iteration, cost, cost_change, eta, is_successful, is_valid, gradient_norm, step_norm, trust_region_radius,"
      "line_search_iterations, linear_solver_iterations, step_solver_time_in_seconds, iteration_time_in_seconds, "
      "cumulative_time_in_seconds" << std::endl;

  // Dump the iterations (yep, there's no detailed code for this in Ceres...)
  for (auto &it_summary : summary.iterations) {
    out << it_summary.iteration << "," << it_summary.cost << "," << it_summary.cost_change << ","
        << it_summary.eta << "," << it_summary.step_is_successful << "," << it_summary.step_is_valid << ","
        << it_summary.gradient_norm << "," << it_summary.step_norm << "," << it_summary.trust_region_radius << ","
        << it_summary.line_search_iterations << "," << it_summary.linear_solver_iterations << ","
        << it_summary.step_solver_time_in_seconds << "," << it_summary.iteration_time_in_seconds << ", "
        << it_summary.cumulative_time_in_seconds
        << std::endl;
  }

  // Dump the configuration data (subset of the full report, but easier to parse)
  // TODO(andreib): More data, maybe.
  out_meta << params.get_details() << std::endl;

  // Dump the raw final report
  out_raw << summary.FullReport() << std::endl;

}

/**
 * Runs the basic experiments used in the report.
 *
 * The main design goals of this method:
 *    * Easy to re-do specific parts of the computation.
 *    * NO data analysis at all, but very rich dumping.
 *    * Resilient to, e.g., a few missing data files.
 */
void Experiments(const std::map<std::string, std::vector<int>> &problems, const std::string &dataset_root) {
//  vector<SummaryPtr> results;
  // TODO flagify
  std::string result_out_dir = "../experiments/00";

  ceres::Solver::Options base_options;
  base_options.max_num_iterations = 200;
  // Do not spend more than X minutes solving a problem.
  base_options.max_solver_time_in_seconds = 60 * 5;
  base_options.num_threads = 32;
  base_options.minimizer_progress_to_stdout = true;

  auto lm_configs = get_lm_configs(base_options);

  // TODO(andreib): Use C++17 paths.
  const std::string sequence = "trafalgar";
  const std::string sequence_root = dataset_root + "/" + sequence + "/";

  int i = 0;
  for (const std::string &fname : kProblemFiles.at(sequence)) {
    i++;
    const vector<int> &sequence_problems = problems.at(sequence);
    if (!sequence_problems.empty() &&
        std::find(sequence_problems.cbegin(), sequence_problems.cend(), i) == sequence_problems.cend()) {
      LOG(INFO) << "Skipping problem [" << i << "]...";
      continue;
    }

    const std::string fpath = sequence_root + fname;
    LOG(INFO) << "Experimenting on problem from file [" << fname << "].";

    try {
      auto config = lm_configs[0];
      auto result = SolveSimpleBA(fpath, config);

      if (result == nullptr) {
        LOG(ERROR) << "Error running experiment..." << endl;
        continue;
      }

      LOG(INFO) << result->FullReport() << std::endl << std::endl;
      SaveResults(result_out_dir, sequence, fname, config, *result);
    }
    catch (std::bad_alloc &bad_alloc_ex) {
      LOG(ERROR) << "Could not run experiment because of insufficient memory. Continuing..." << endl;
    }
  }
}


/// Splits a string using the given delimiter.
/// Source: https://stackoverflow.com/a/236803/1055295
template<typename Out>
void split(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

std::map<std::string, std::vector<int>> ParseProblemList(const std::string &problem_list) {
  using namespace std;
  map<string, vector<int>> res;

  vector<string> dataset_specs = split(problem_list, ';');
  for (const string &spec : dataset_specs) {
    LOG(INFO) << "Parsing spec: " << spec << endl;
    vector<string> spec_pair = split(spec, ':');
    CHECK_EQ(spec_pair.size(), 2) << "Sequence spec must follow the format NAME:a,b,c or NAME:ALL";

    string dataset = spec_pair[0];
    string entries = spec_pair[1];

    LOG(INFO) << "Dataset: " << dataset << endl;
    LOG(INFO) << "Entries: " << entries << endl;

    res[dataset] = vector<int>();
    if (entries == "ALL") {
      continue;
    }
    else {
      for (const string &nr : split(entries, ',')) {
        int entry = atoi(nr.c_str());
        if (0 == entry) {
          LOG(FATAL) << "Invalid entry [" << nr << "]. Must be a strictly positive integer.";
        }
        res[dataset].push_back(entry);
      }
    }
  }

  return res;
};

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();

  gflags::SetUsageMessage("Simple bundle adjustment solver benchmark.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto res = ParseProblemList(FLAGS_problem_list);
  for (const auto &pair : res) {
    std::cout << pair.first << " ";
    if (pair.second.empty()) {
      std::cout << "ALL";
    }
    else {
      for (int val : pair.second) {
        std::cout << val << " ";
      }
    }
    std::cout << endl;
  }

  Experiments(res, FLAGS_dataset_root);

//    int idx = 0;
//    for(auto result : results) {
//        cout << labels_outer[idx / 4] << ", " << labels_inner[idx % 4] << ": \t"
//             << result->initial_cost << " --> " << setprecision(12) << result->final_cost
//             << " in [" << result->num_successful_steps << "] succesful. steps "
//             << endl;
//        idx += 1;
//    }

  return 0;
}
