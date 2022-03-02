#include "blackthorn/Tools.h"
#include <fmt/format.h>

namespace blackthorn::tools {

static auto width_string(const std::string &in,
                         const std::vector<std::string> &fs) -> std::string {
  std::string s =
      fmt::format("{}({} {}", UNICODE_CAP_GAMMA, in, UNICODE_LARROW);

  for (const auto &f : fs) {
    s = fmt::format("{} + {}", s, f);
  }
  return fmt::format("{})", s);
}

static auto cs_string(const std::string &in1, const std::string &in2,
                      const std::vector<std::string> &fs) -> std::string {
  std::string s =
      fmt::format("{}({} + {} {}", UNICODE_SIGMA, in1, in2, UNICODE_LARROW);

  for (const auto &f : fs) {
    s = fmt::format("{} + {}", s, f);
  }
  return fmt::format("{})", s);
}

auto print_width(double mean, double std, const std::string &in,
                 const std::vector<std::string> &fs) -> void {
  std::string s = width_string(in, fs);
  fmt::print("{} = {} {} {}\n", s, mean, UNICODE_PM, std);
}

auto print_width(const std::string &head, double mean, double std,
                 const std::string &in, const std::vector<std::string> &fs)
    -> void {
  std::string s = width_string(in, fs);
  fmt::print("({}): {} = {} {} {}\n", head, s, mean, UNICODE_PM, std);
}

auto print_width(double val, const std::string &in,
                 const std::vector<std::string> &fs) -> void {
  std::string s = width_string(in, fs);
  fmt::print("{} = {}\n", s, val);
}

auto print_width(const std::string &head, double val, const std::string &in,
                 const std::vector<std::string> &fs) -> void {
  std::string s = width_string(in, fs);
  fmt::print("({}): {} = {}\n", head, s, val);
}

auto print_cross_section(double mean, double std, const std::string &in1,
                         const std::string &in2,
                         const std::vector<std::string> &fs) -> void {
  std::string s = cs_string(in1, in2, fs);
  fmt::print("{} = {} {} {}\n", s, mean, UNICODE_PM, std);
}

auto print_cross_section(const std::string &head, double mean, double std,
                         const std::string &in1, const std::string &in2,
                         const std::vector<std::string> &fs) -> void {
  std::string s = cs_string(in1, in2, fs);
  fmt::print("({}): {} = {} {} {}\n", head, s, mean, UNICODE_PM, std);
}

auto print_cross_section(double val, const std::string &in1,
                         const std::string &in2,
                         const std::vector<std::string> &fs) -> void {
  std::string s = cs_string(in1, in2, fs);
  fmt::print("{} = {}\n", s, val);
}

auto print_cross_section(const std::string &head, double val,
                         const std::string &in1, const std::string &in2,
                         const std::vector<std::string> &fs) -> void {
  std::string s = cs_string(in1, in2, fs);
  fmt::print("({}): {} = {}\n", head, s, val);
}

} // namespace blackthorn::tools
