#ifndef CERES_BALPROBLEM_H
#define CERES_BALPROBLEM_H


#include <cstdio>
#include <glog/logging.h>

static const int kNumParamsPerCam = 9;
static const int kNumParamsPer3DPoint = 3;

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
public:
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_  + kNumParamsPerCam * num_cameras_; }
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * kNumParamsPerCam;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * kNumParamsPer3DPoint;
    }

    /// \brief Loads a standard format BAL structure from motion dataset.
    /// \param filename         Path to file to 'fopen'.
    /// \param reparametrize   Whether to reparameterize the camera pose.
    /// \return true on success.
    bool LoadFile(const char* filename, bool reparametrize) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = kNumParamsPerCam * num_cameras_ + kNumParamsPer3DPoint * num_points_;
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }

        if (reparametrize) {
            reparametrize_cameras();
        }

        return true;
    }

private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

    void reparametrize_cameras();

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    int* point_index_ = nullptr;
    int* camera_index_ = nullptr;
    double* observations_ = nullptr;
    double* parameters_ = nullptr;
};

#endif //CERES_BALPROBLEM_H
