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
#include "experiment_configs.h"

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

// TODO(andreib): Similar thing to problem_list, but for the optimizer configs.
//  --optimizer_list lm;t_dogleg;ss_dogleg;bfgs (or something like this)

using SummaryPtr = std::shared_ptr<ceres::Solver::Summary>;

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

  if (! is_dir(out_dir)) {
    std::stringstream err_ss;
    err_ss << "The output directory [" << out_dir << "] does not exist.";
    throw std::runtime_error(err_ss.str());
  }

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
  LOG(INFO) << "\t" << fpath << std::endl;
  LOG(INFO) << "\t" << fpath_meta << std::endl;
  LOG(INFO) << "\t" << fname_raw << std::endl;

  std::ofstream out(fpath);
  std::ofstream out_meta(fpath_meta);
  std::ofstream out_raw(fpath_raw);

  CHECK(out.is_open());
  CHECK(out_meta.is_open());
  CHECK(out_raw.is_open());

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
  for (const auto &el : problems) {
    const std::string sequence_name = el.first;
    const std::vector<int> sequence_problems = el.second;

    LOG(INFO) << "";
    LOG(INFO) << "Processing sequence category [" << sequence_name << "].";
    LOG(INFO) << "";
    // TODO(andreib): Use C++17 paths.
    const std::string sequence_root = dataset_root + "/" + sequence_name + "/";

    int i = 0;
    for (const std::string &fname : kProblemFiles.at(sequence_name)) {
      i++;
      if (!sequence_problems.empty() &&
          find(sequence_problems.cbegin(), sequence_problems.cend(), i) == sequence_problems.cend()) {
        LOG(INFO) << "Skipping problem [" << i << "]...";
        continue;
      }

      const std::string fpath = sequence_root + fname;
      LOG(INFO) << "Experimenting on problem from file [" << fname << "].";

      try {
        // TODO(andreib): The current tolerance seems inappropriate for very large problems. Perhaps it would make
        // sense to set it as a function of the number of residuals. IMPORTANT: look at mean final residual per pixel
        // before and after optimization! It's likely the residuals on average are super tiny even before BA!
        auto result = SolveSimpleBA(fpath, config);

        if (result == nullptr) {
          LOG(ERROR) << "Error running experiment..." << std::endl;
          continue;
        }

        LOG(INFO) << result->BriefReport() << std::endl << std::endl;
        SaveResults(result_out_dir, sequence_name, fname, config, *result);

        // This assumes the problem files are ordered in increasing order of difficulty.
        if (result->total_time_in_seconds > 1.0 + FLAGS_max_seconds_per_problem) {
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
}

/// Runs the basic experiments described in the report.
/// The main design goals of this method:
///  * Easy to re-do specific parts of the computation.
///  * NO data analysis at all, but very rich dumping.
///  * Resilient to, e.g., a few missing data files.
void Experiments(
    const std::string &dataset_root,
    const std::string &result_out_dir,
    const std::map<std::string, std::vector<int>> &problems
) {
  ceres::Solver::Options base_options;
  // We set the maximum number of iterations to a very large value since (for experimentation and practical purposes)
  // we care more about the maximum overall time. Ideally, we'd remove that cap as well, but we wish to run these
  // experiments in a finite amount of time.
  base_options.max_num_iterations = 20000;
  base_options.max_solver_time_in_seconds = FLAGS_max_seconds_per_problem;
  base_options.num_threads = 24;
//  base_options.minimizer_progress_to_stdout = false;
  base_options.minimizer_progress_to_stdout = true;

//  auto configs = get_line_search_configs(base_options);
//  auto configs = get_lm_configs(base_options);
  auto configs = get_dogleg_configs(base_options);
//  auto configs = get_it_schur_configs(base_options);
  // TODO(andreib): Experiment with varying final threshold.
  // TODO(andreib): Prof was very excited when I mentioned reconstructing Rome. Perhaps I could plot a resulting
  // pointcloud.
  // TODO(andreib): VERY important!! Plot residual value over time for selected solvers, to show how much the
  //                error goes down over time, what the initial residual is, etc. May need Y to be log scale.

  for (const auto &config : configs) {
    LOG(INFO) << "";
    LOG(INFO) << "";
    LOG(INFO) << "Processing config [" << config.get_details() << "]";
    LOG(INFO) << "";
    LOG(INFO) << "";
    EvaluateOptimizerConfig(dataset_root, result_out_dir, problems, config);
  }

  LOG(INFO) << "Experiments complete.";
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