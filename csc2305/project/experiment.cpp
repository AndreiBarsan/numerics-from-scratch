//
// Main subroutines for running Bundle Adjustment experiments.
//

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include "bundle_adjustment.h"
#include "bal_problem.h"
#include "csv_writer.h"
#include "utils.h"

DEFINE_string(dataset_root, "../data",
              "The root folder where the BAL datasets are present. (See 'get_data.py' for more info.)");
DEFINE_string(output_dir, "../experiments/00", "Where to write experiment outputs.");
DEFINE_string(problem_list,
              "trafalgar:ALL",
              "Indices of the problems to solve. The format should be 'nameA:1,2,..,n;nameB:1,2,...,m;...'. For "
                  "instance, 'trafalgar:1,3,5' runs the first, third, and fifth problems from the trafalgar set, while "
                  "'venice:ALL; trafalgar:2,5' runs all the venice sequences and sequences 2 and 5 from the trafalgar "
                  "set.");
DEFINE_int64(max_seconds_per_problem, 180, "The maximum time to spend on one (problem, params) configuration.");

using SummaryPtr = std::shared_ptr<ceres::Solver::Summary>;

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

// 14 in total
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

// 16 in total
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
    if (solver_options.minimizer_type == TRUST_REGION) {
      out_ss << TrustRegionStrategyTypeToString(solver_options.trust_region_strategy_type) << "-";
      if (solver_options.trust_region_strategy_type == DOGLEG) {
        out_ss << DoglegTypeToString(solver_options.dogleg_type) << "-";
      }
      out_ss << LinearSolverTypeToString(solver_options.linear_solver_type) << "-";
    } else {
      out_ss << LineSearchDirectionTypeToString(solver_options.line_search_direction_type) << "-";
      if (solver_options.line_search_direction_type == NONLINEAR_CONJUGATE_GRADIENT) {
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

ceres::TrustRegionStrategyType kTrStrategies[] = {
    ceres::LEVENBERG_MARQUARDT,
    ceres::DOGLEG,
};

ceres::DoglegType kDoglegTypes[] = {
    ceres::TRADITIONAL_DOGLEG,
    ceres::SUBSPACE_DOGLEG,
};

ceres::LinearSolverType kTrSolvers[] = {
    // TODO(andreib): Describe this condition for elimination in your report, to explain why your
    // plots don't even include these approaches.
    // DENSE_NORMAL_CHOLESY takes 20-30 seconds even on a tiny problem with 21 images (trafalgar-1).
//    ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY,

    // Same for 'DENSE_QR'. Spends >30-40s on the first iteration of the simplest problem, so ignored.
    // Comparatively, sparse (or schur-complement) methods converge on this problem in 2-3 seconds at most.
//    ceres::LinearSolverType::DENSE_QR,
    ceres::SPARSE_NORMAL_CHOLESKY,
    ceres::DENSE_SCHUR,
    ceres::SPARSE_SCHUR,
    ceres::CGNR,
};

std::vector<ExperimentParams> get_lm_configs(const ceres::Solver::Options &base_options) {
  using namespace ceres;
  std::vector<ExperimentParams> out;
  for (LinearSolverType solver : kTrSolvers) {
    Solver::Options options = base_options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = LEVENBERG_MARQUARDT;
    options.linear_solver_type = solver;
    out.emplace_back(true, true, options);
  }
  return out;
}

std::vector<ExperimentParams> get_dogleg_configs(const ceres::Solver::Options &base_options) {
  using namespace ceres;
  std::vector<ExperimentParams> out;
  for (DoglegType dogleg_type : kDoglegTypes) {
    for (LinearSolverType solver : kTrSolvers) {
      Solver::Options options = base_options;
      options.minimizer_type = TRUST_REGION;
      options.trust_region_strategy_type = DOGLEG;
      options.linear_solver_type = solver;
      options.dogleg_type = dogleg_type;

      out.emplace_back(true, true, options);
    }
  }
  return out;
}

SummaryPtr SolveSimpleBA(const std::string &data_file_fpath, const ExperimentParams &experiment_params) {
  // Solves a problem from the BAL dataset.

  // Whether to also account for the cameras' radial distortion parameter.
  const bool enable_radial = experiment_params.enable_radial;
  const bool reparametrize = experiment_params.reparametrize;

  ceres::Problem problem;
  BALProblem bal_problem;
  if (!bal_problem.LoadFile(data_file_fpath.c_str(), reparametrize)) {
    LOG(ERROR) << "Could not load data from file: " << data_file_fpath << std::endl;
    return nullptr;
  }

  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each residual depends on a 3D point and 9-param camera (calibration not assumed).
    double obs_x = bal_problem.observations()[2 * i + 0];
    double obs_y = bal_problem.observations()[2 * i + 1];
    ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(
        obs_x,
        obs_y,
        experiment_params.enable_radial,
        experiment_params.reparametrize);

    // null loss function == regular squared loss (no robust estimators, for simplicity).
    ceres::LossFunction *loss_function = nullptr;

    problem.AddResidualBlock(
        cost_function,
        loss_function,
        bal_problem.mutable_camera_for_observation(i),
        bal_problem.mutable_point_for_observation(i));
  }

  LOG(INFO) << "Finished preparing problem (" << bal_problem.num_observations() << " observations)." << std::endl;

  auto summary = std::make_shared<ceres::Solver::Summary>();
  LOG(INFO) << "Starting to solve..." << std::endl;
  ceres::Solve(experiment_params.solver_options, &problem, summary.get());
  LOG(INFO) << "Finished!";
  return summary;
}
/// Dumps the output of a single experimental run into three files.
/// The files are the main CSV results (all iterations), a metadata file, and the Ceres Summary::FullReport output.
///
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
  const std::string fpath_meta = out_dir + "/" + fname_meta;
  const std::string fpath_raw = out_dir + "/" + fname_raw;

  if (FileExists(fpath)) {
    // TODO-LOW(andreib): Consider erroring out, or checking this in advance.
    LOG(WARNING) << "Results file [" << fpath << "] already exists. Will overwrite." << std::endl;
  }

  LOG(INFO) << "Writing data to files:" << std::endl;
  LOG(INFO) << "\t" << fname << std::endl;
  LOG(INFO) << "\t" << fname_meta << std::endl;
  LOG(INFO) << "\t" << fname_raw << std::endl;

  std::ofstream out(fpath);
  std::ofstream out_meta(fpath_meta);
  std::ofstream out_raw(fpath_raw);

  // Write the header (no spaces after commas because of performance reasons in Pandas).
  out << "iteration,cost,cost_change,eta,is_successful,is_valid,gradient_norm,step_norm,trust_region_radius,"
      "line_search_iterations,linear_solver_iterations,step_solver_time_in_seconds,iteration_time_in_seconds,"
      "cumulative_time_in_seconds" << std::endl;

  // Dump the iterations (yep, there's no detailed code for this in Ceres...)
  for (auto &it_summary : summary.iterations) {
    out << it_summary.iteration << "," << it_summary.cost << "," << it_summary.cost_change << ","
        << it_summary.eta << "," << it_summary.step_is_successful << "," << it_summary.step_is_valid << ","
        << it_summary.gradient_norm << "," << it_summary.step_norm << "," << it_summary.trust_region_radius << ","
        << it_summary.line_search_iterations << "," << it_summary.linear_solver_iterations << ","
        << it_summary.step_solver_time_in_seconds << "," << it_summary.iteration_time_in_seconds << ","
        << it_summary.cumulative_time_in_seconds
        << std::endl;
  }

  // Dump the configuration data (subset of the full report, but easier to parse)
  // TODO(andreib): More data, maybe.
  // TODO(andreib): Explicitly dump n_images and n_points from fname, + n_observations.
  out_meta << params.get_details() << std::endl;

  // Dump the raw final report
  out_raw << summary.FullReport() << std::endl;

}
/// Solves every problem in the list using the given optimizer config and dumps the results to the given directory.
void EvaluateOptimizerConfig(const std::string &dataset_root,
                             const std::string &result_out_dir,
                             const std::map<std::string, std::vector<int>> &problems,
                             const ExperimentParams &config
) {
  // TODO(andreib): Use C++17 paths.
  const std::string sequence = "trafalgar";
  const std::string sequence_root = dataset_root + "/" + sequence + "/";

  int i = 0;
  for (const std::string &fname : kProblemFiles.at(sequence)) {
    i++;
    const std::vector<int> &sequence_problems = problems.at(sequence);
    if (!sequence_problems.empty() &&
        find(sequence_problems.cbegin(), sequence_problems.cend(), i) == sequence_problems.cend()) {
      LOG(INFO) << "Skipping problem [" << i << "]...";
      continue;
    }

    const std::string fpath = sequence_root + fname;
    LOG(INFO) << "Experimenting on problem from file [" << fname << "].";

    try {
      auto result = SolveSimpleBA(fpath, config);

      if (result == nullptr) {
        LOG(ERROR) << "Error running experiment..." << std::endl;
        continue;
      }

      LOG(INFO) << result->BriefReport() << std::endl << std::endl;
      SaveResults(result_out_dir, sequence, fname, config, *result);

      // This assumes the problem files are ordered in increasing order of difficulty.
      if (fabs(result->total_time_in_seconds - static_cast<double>(FLAGS_max_seconds_per_problem)) <= 1.0) {
        LOG(ERROR) << "Experiment timed out, not continuing with even larger problems...";
        break;
      }
    }
    catch (std::bad_alloc &bad_alloc_ex) {
      // This can happen when attempting to use dense linear solvers for huge problems.
      LOG(ERROR) << "Could not run experiment because of insufficient memory. Continuing..." << std::endl;
    }
  }
}
/**
 * Runs the basic experiments used in the report.
 *
 * The main design goals of this method:
 *    * Easy to re-do specific parts of the computation.
 *    * NO data analysis at all, but very rich dumping.
 *    * Resilient to, e.g., a few missing data files.
 */
void Experiments(
    const std::string &dataset_root,
    const std::string &result_out_dir,
    const std::map<std::string, std::vector<int>> &problems
) {
  ceres::Solver::Options base_options;
  base_options.max_num_iterations = 200;
  base_options.max_solver_time_in_seconds = FLAGS_max_seconds_per_problem;
  base_options.num_threads = 24;
//  base_options.minimizer_progress_to_stdout = false;
  base_options.minimizer_progress_to_stdout = true;

  auto lm_configs = get_lm_configs(base_options);
  for (const auto &config : lm_configs) {
    LOG(INFO) << "";
    LOG(INFO) << "";
    LOG(INFO) << "Processing config [" << config.get_details() << "]";
    LOG(INFO) << "";
    LOG(INFO) << "";
    EvaluateOptimizerConfig(dataset_root, result_out_dir, problems, config);
  }
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
    } else {
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
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();

  google::SetUsageMessage("Simple bundle adjustment solver benchmark.");
  google::ParseCommandLineFlags(&argc, &argv, true);

  auto problem_list = ParseProblemList(FLAGS_problem_list);
  for (const auto &pair : problem_list) {
    std::cout << pair.first << " ";
    if (pair.second.empty()) {
      std::cout << "ALL";
    } else {
      for (int val : pair.second) {
        std::cout << val << " ";
      }
    }
    std::cout << std::endl;
  }

  Experiments(FLAGS_dataset_root, FLAGS_output_dir, problem_list);

  return 0;
}