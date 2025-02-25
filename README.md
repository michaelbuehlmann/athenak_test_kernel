# athenak_test_kernel for Aurora

## Build and Run Instructions

1. **Preparation**
   - Clone the repo recursively or manually initialize the submodules (kokkos) with
     > git submodules update --init

2. **Compile with CMake**
   - Create a build directory:
     > mkdir build && cd build
   - Run CMake to configure the project:
     > cmake ..
   - Compile the code:
     > make

3. **Run the code**
   - Run the executable, providing the path to the data folder (/data in the repository) as an argument:
     > ./athenak_test_kernel /path/to/data_folder
