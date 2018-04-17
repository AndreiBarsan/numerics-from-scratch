/// An OpenMP hello world. Very simple exploration of OpenMP primitives.

#include <iostream>

#include <glog/logging.h>
// Should be brought in by the Ceres dependency. Just don't forget to add the '-fopenmp' compiler flag!
#include <omp.h>

#define BIG 100000000

void openmp_loop() {
  int tid;
  int thread_count;

  int n_cores = omp_get_max_threads();
  std::cout << "OpenMP will have a maximum of " << n_cores << " threads." << std::endl;
  omp_set_num_threads(n_cores);

#pragma omp parallel private(tid, thread_count)
  {
    tid = omp_get_thread_num();
    thread_count = omp_get_num_threads();
    LOG(INFO) << "Hello from thread " << tid << "/" << thread_count << "." << std::endl;

    if (tid == 0) {
      // Master-only code
      LOG(INFO) << "Bonus hello from master (" << tid << ")." << std::endl;
    }
  }

  LOG(INFO) << "Joined threads OK." << std::endl;

  long i;
  auto *a = new float[BIG];
  auto *b = new float[BIG];

#pragma omp parallel for
  for (i = 0; i < BIG; ++i) {
    b[i] = 1.23f * i;
    a[i] = 1.0;
    for (int j = 0; j < 100; ++j) {
      a[i] *= b[i] * b[i];
      a[i] /= i;
    }
  }

  for (i = 0; i < 1000; ++i) {
    LOG(INFO) << i << " " << i * i << " " << a[i * i] << std::endl;
  }

  delete a;
  delete b;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::INFO);
  openmp_loop();

  return 0;
}

