#include "blackthorn/Tools.h"
#include <cmath>

namespace blackthorn::tools {

auto zero_or_subnormal(double x) -> bool {
  switch (std::fpclassify(x)) {
  case FP_ZERO:
  case FP_SUBNORMAL:
    return true;
  default:
    return false;
  }
}

} // namespace blackthorn::tools
