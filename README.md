# My Kokkos Application

## Overview
This project demonstrates the use of Kokkos for parallel computing in C++. It specifically implements a kernel for unpacking boundary values for cell-centered mesh variables, inspired by the functionality in the Athena++ astrophysical MHD code.

## Project Structure
```
my-kokkos-app
├── src
│   ├── main.cpp        # Entry point of the application
│   └── types.hpp      # Data structures and types used in the application
├── CMakeLists.txt     # CMake configuration file
└── README.md          # Project documentation
```

## Requirements
- CMake (version 3.10 or higher)
- Kokkos library (version 3.0 or higher)
- A compatible C++ compiler (e.g., GCC, Clang, or Intel)

## Building the Project
1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```bash
   cd my-kokkos-app
   ```
3. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```bash
   cmake ..
   ```
5. Build the project:
   ```bash
   make
   ```

## Running the Application
After building the project, you can run the application using:
```bash
./my-kokkos-app
```

## Functionality
The application initializes necessary data structures and invokes Kokkos parallel execution to unpack boundary values efficiently. It serves as a template for further development in astrophysical simulations or other applications requiring parallel processing of mesh data. 

## License
This project is licensed under the MIT License. See the LICENSE file for details.