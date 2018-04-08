#include "Utils.h"

#include <sstream>

template<typename Out>
void split(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    *(result++) = item;
  }
}
std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, back_inserter(elems));
  return elems;
}
