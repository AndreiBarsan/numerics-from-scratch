#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <sys/stat.h>

/// Splits a string using the given delimiter.
/// Source: https://stackoverflow.com/a/236803/1055295
template<typename Out>
void split(const std::string &s, char delim, Out result);

std::vector<std::string> split(const std::string &s, char delim);

bool is_dir(const std::string &path);

#endif // UTILS_H
