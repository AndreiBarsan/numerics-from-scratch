#include "utils.h"

#include <sstream>

template<typename Out>
void Split(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> Split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  Split(s, delim, back_inserter(elems));
  return elems;
}

bool PathExists(const std::string &path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

bool IsDir(const std::string &path) {
  struct stat info;

  if (stat(path.c_str(), &info) != 0) {
    return false;
  }

  return (info.st_mode & S_IFDIR) != 0;
}
