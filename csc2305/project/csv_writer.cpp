
#include "csv_writer.h"

#include <iostream>

using namespace std;

void csv_writer::Write(const ICsvSerializable &data) {
  if (! wrote_header_) {
    *output_ << data.GetHeader() << endl;
    wrote_header_ = true;
  }

  *output_ << data.GetData() << endl;
}
