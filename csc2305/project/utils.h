#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <sys/stat.h>

/// Splits a string using the given delimiter.
/// Source: https://stackoverflow.com/a/236803/1055295
template<typename Out>
void Split(const std::string &s, char delim, Out result);

std::vector<std::string> Split(const std::string &s, char delim);

/// Checks whether the path exists.
bool PathExists(const std::string &path);

/// Checks whether the path exists and is a directory.
bool IsDir(const std::string &path);


#endif // UTILS_H
