// Originally from the DynSLAM project.

#ifndef DYNSLAM_CSVWRITER_H
#define DYNSLAM_CSVWRITER_H

#include <fstream>
#include <string>
#include <sys/stat.h>


inline bool FileExists(const std::string &fpath) {
  struct stat buffer;
  return stat(fpath.c_str(), &buffer) == 0;
}

/// \brief Interface for poor man's serialization.
/// In the long run, it would be nice to use protobufs or something for this...
class ICsvSerializable {
 public:
  virtual ~ICsvSerializable() = default;

  // TODO-LOW(andrei): The correct C++ way of doing this is by just making this writable to an ostream.
  /// \brief Should return the field names in the same order as GetData, without a newline.
  virtual std::string GetHeader() const = 0;
  virtual std::string GetData() const = 0;
};

class csv_writer {
 public:
  const std::string output_fpath_;

  explicit csv_writer(const std::string &output_fpath)
    : output_fpath_(output_fpath),
      wrote_header_(false),
      output_(new std::ofstream(output_fpath))
  {
    if(! FileExists(output_fpath)) {
      throw std::runtime_error("Could not open CSV file. Does the folder it should be in exist?");
    }
  }

  csv_writer(const csv_writer &) = delete;
  csv_writer(csv_writer &&) = delete;
  csv_writer& operator=(const csv_writer &) = delete;
  csv_writer& operator=(csv_writer &&) = delete;

  void Write(const ICsvSerializable &data);

  virtual ~csv_writer() {
    delete output_;
  }

 private:
  bool wrote_header_;
  std::ostream *output_;
};


#endif //DYNSLAM_CSVWRITER_H
