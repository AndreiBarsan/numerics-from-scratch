//
// Manages the various Ceres experiment configurations.
//

#ifndef EXPERIMENT_CONFIGS_H
#define EXPERIMENT_CONFIGS_H

#include <fstream>
#include <iomanip>
#include <memory>
#include <vector>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include "bundle_adjustment.h"

// TODO(andreib): Solve inexactly with iterative Schur! The default max number of iterations is big, 500.
//    What if you set it to 100, 50, or even 10? It seems good steps take only a handful of iterations.

// TODO(andreib): Dedicated experiment with ls_iter limit.
// TODO(andreib): Dedicated experiment to look at max processors.

// Problem file name structure: problem-IMAGES-POINTS-OBSERVATIONS
std::vector<std::string> kVeniceFiles = {
    "problem-52-64053-pre.txt",
    "problem-89-110973-pre.txt",
    "problem-245-198739-pre.txt",
    "problem-427-310384-pre.txt",
    "problem-744-543562-pre.txt",
    "problem-951-708276-pre.txt",
    // Skip 1
    "problem-1158-802917-pre.txt",
    // Skip 2
    "problem-1288-866452-pre.txt",
    // Skip 1
    "problem-1408-912229-pre.txt",
    // Skip 2
    "problem-1490-935273-pre.txt",
    // Skip 13 very large problems due to time constraints.
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
    // DENSE_NORMAL_CHOLESY takes 2-3 HOURS even on a tiny problem with 21 images (trafalgar-1).
//    ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY,

    // Same for 'DENSE_QR'. Spends >8h on the first iteration of the simplest problem, so ignored.
    // Comparatively, sparse (or schur-complement) methods converge on this problem in 2-3 seconds at most.
//    ceres::LinearSolverType::DENSE_QR,
    ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY,
    ceres::LinearSolverType::DENSE_SCHUR,
    ceres::LinearSolverType::SPARSE_SCHUR,
    ceres::LinearSolverType::CGNR,
    ceres::LinearSolverType::ITERATIVE_SCHUR
};

struct ExperimentParams {
  bool enable_radial;
  bool reparametrize;

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

  Solver::Options solver_options;

  ExperimentParams(const bool enable_radial,
                   const bool reparametrize,
                   const Solver::Options &solver_options)
      : enable_radial(enable_radial),
        reparametrize(reparametrize),
        solver_options(solver_options) {}

  /// Returns a representation of this object suitable for including as part of a file name.
  std::string get_label() const {
    using namespace ceres;
    std::stringstream out_ss;
    out_ss << MinimizerTypeToString(solver_options.minimizer_type) << "-";
    if (solver_options.minimizer_type == TRUST_REGION) {
      //
      // Trust region method config
      //
      out_ss << TrustRegionStrategyTypeToString(solver_options.trust_region_strategy_type) << "-";
      if (solver_options.trust_region_strategy_type == DOGLEG) {
        out_ss << DoglegTypeToString(solver_options.dogleg_type) << "-";
      }
      out_ss << LinearSolverTypeToString(solver_options.linear_solver_type) << "-";
    } else {
      //
      // Line search method config
      //
      out_ss << LineSearchDirectionTypeToString(solver_options.line_search_direction_type) << "-";
      out_ss << LineSearchInterpolationTypeToString(solver_options.line_search_interpolation_type) << "-";
      if (solver_options.line_search_direction_type == NONLINEAR_CONJUGATE_GRADIENT) {
        out_ss << NonlinearConjugateGradientTypeToString(solver_options.nonlinear_conjugate_gradient_type) << "-";
      }

      // Probably unused, but you never know.
      out_ss << solver_options.max_lbfgs_rank << "-";
    }
    //
    // Common components
    //
    out_ss << solver_options.max_linear_solver_iterations << "-";
    if (solver_options.use_nonmonotonic_steps) {
      out_ss << "nonmonotonic_steps" << "-";
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

std::vector<ExperimentParams> get_lm_configs(const Solver::Options &base_options) {
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

std::vector<ExperimentParams> get_dogleg_configs_internal(
    const Solver::Options &base_options,
    ceres::DoglegType dogleg_type
) {
  using namespace ceres;
  std::vector<ExperimentParams> out;
    for (LinearSolverType solver : kTrSolvers) {
      Solver::Options options = base_options;
      options.minimizer_type = TRUST_REGION;
      options.trust_region_strategy_type = DOGLEG;
      options.dogleg_type = dogleg_type;
      options.linear_solver_type = solver;

      out.emplace_back(true, true, options);
  }
  return out;
}

/// Used for experimenting the max iteration count of the iterative Schur solver in LM.
/// Note: the iterative schur solver is not supported in dogleg.
std::vector<ExperimentParams> get_it_schur_configs(const Solver::Options &base_options) {
  using namespace ceres;
  // Note: the Ceres default is 500.
  int max_it_vals[] = { 10, 25, 50, 100, 250, 500, 750, 1000 };
  std::vector<ExperimentParams> out;

  for (const int &max_it : max_it_vals) {
    Solver::Options options = base_options;
    options.max_linear_solver_iterations = max_it;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = LEVENBERG_MARQUARDT;
    options.linear_solver_type = ITERATIVE_SCHUR;

    out.emplace_back(true, true, options);
  }
  return out;
}

std::vector<ExperimentParams> get_traditional_dogleg_configs(const Solver::Options &base_options) {
  return get_dogleg_configs_internal(base_options, ceres::DoglegType::TRADITIONAL_DOGLEG);
}

std::vector<ExperimentParams> get_subspace_dogleg_configs(const Solver::Options &base_options) {
  return get_dogleg_configs_internal(base_options, ceres::DoglegType::SUBSPACE_DOGLEG);
}

std::vector<ExperimentParams> get_line_search_configs(const Solver::Options &base_options) {
  // TODO(andreib): should really try gauss newton somehow. Consider asking mailing list. Basically GN is LM but
  // where you never add the regularizer. I think there's a param for that, max_lm_diagonal. I *think* that if I set
  // this to 0.0, then we degenerate into Gauss-Newton, but I'd have to dig a bit deeper.
  throw std::runtime_error("not yet implemented");
}

std::vector<ExperimentParams> get_dogleg_configs(const Solver::Options &base_options) {
  std::vector<ExperimentParams> joint;
  auto trad = get_traditional_dogleg_configs(base_options);
  auto ss = get_subspace_dogleg_configs(base_options);
  joint.insert(joint.end(), trad.begin(), trad.end());
  joint.insert(joint.end(), ss.begin(), ss.end());
  return joint;
}


#endif //EXPERIMENT_CONFIGS_H
