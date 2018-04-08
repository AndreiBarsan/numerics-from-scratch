
#include "CsvWriter.h"

#include <iostream>

using namespace std;

void CsvWriter::Write(const ICsvSerializable &data) {
  if (! wrote_header_) {
    *output_ << data.GetHeader() << endl;
    wrote_header_ = true;
  }

  *output_ << data.GetData() << endl;
}
