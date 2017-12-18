#ifndef CERES_BALPROBLEM_H
#define CERES_BALPROBLEM_H

static const int kNumParamsPerCam = 9;

static const int kNumParamsPer3DPoint = 3;

#include <cstdio>
#include <glog/logging.h>

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

    void reparametrize_cameras() {
        std::cout << "Performing camera reparametrization..." << std::endl;
        for (int i = 0; i < num_cameras_; ++i) {
            int cam_idx = i * kNumParamsPerCam;

            double t_raw[3];
            t_raw[0] = parameters_[cam_idx + 3];
            t_raw[1] = parameters_[cam_idx + 4];
            t_raw[2] = parameters_[cam_idx + 5];
            double aa_rot_new[3];
            aa_rot_new[0] = -parameters_[cam_idx + 0];
            aa_rot_new[1] = -parameters_[cam_idx + 1];
            aa_rot_new[2] = -parameters_[cam_idx + 2];

//            printf("Camera %d translation: %.4lf %.4lf %.4lf.\n", i, t_raw[0], t_raw[1], t_raw[2]);
//            printf("Camera %d rotation: %.4lf %.4lf %.4lf.\n", i,
//                   -aa_rot_new[0], -aa_rot_new[1], -aa_rot_new[2]);

            double t_rotated[3];
            ceres::AngleAxisRotatePoint(aa_rot_new, t_raw, t_rotated);

            // Write back the new rotation (R')
            parameters_[cam_idx + 0 + 0] = aa_rot_new[0];
            parameters_[cam_idx + 0 + 1] = aa_rot_new[1];
            parameters_[cam_idx + 0 + 2] = aa_rot_new[2];

            // Write back the new translation -(R' t)
            parameters_[cam_idx + 3 + 0] = -t_rotated[0];
            parameters_[cam_idx + 3 + 1] = -t_rotated[1];
            parameters_[cam_idx + 3 + 2] = -t_rotated[2];

            // The remaining parameters (f, k1, k2) remain unchanged.
        }
    }

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};

#endif //CERES_BALPROBLEM_H
