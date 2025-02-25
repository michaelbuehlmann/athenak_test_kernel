#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename ViewType>
void load_data(const std::string &filename, ViewType d_view) {
  auto h_view = Kokkos::create_mirror_view(d_view);

  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  using ValueType = typename ViewType::non_const_value_type;
  std::string line;
  std::vector<ValueType> temp;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    ValueType value;
    while (iss >> value) {
      temp.push_back(value);
    }
  }

  if constexpr (ViewType::rank == 1) {
    if (temp.size() != h_view.extent(0)) {
      throw std::runtime_error("Data size mismatch for 1D view!");
    }
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      h_view(i) = temp[i];
    }
  } else if constexpr (ViewType::rank == 2) {
    if (temp.size() != h_view.extent(0) * h_view.extent(1)) {
      throw std::runtime_error("Data size mismatch for 2D view!");
    }
    size_t ncols = h_view.extent(1);
    for (size_t i = 0; i < h_view.extent(0); ++i) {
      for (size_t j = 0; j < h_view.extent(1); ++j) {
        h_view(i, j) = temp[i * ncols + j];
      }
    }
  } else {
    static_assert(ViewType::rank == 1 || ViewType::rank == 2, "Unsupported view rank");
  }

  Kokkos::deep_copy(d_view, h_view);
}