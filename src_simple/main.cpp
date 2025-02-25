
#include <Kokkos_Core.hpp>
#include <iostream>

using DevExeSpace = Kokkos::DefaultExecutionSpace;
using TeamMember_t = Kokkos::TeamPolicy<>::member_type;

constexpr int nmb = 8;
constexpr int nnghbr = 56;
constexpr int nvar = 22;
constexpr int nmnv = nmb * nnghbr * nvar;

void kernel() {
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    int nkj = 32;
      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        tmember.team_barrier();
      }); }); // end par_for_outer
  Kokkos::fence();
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  {
    // Call the kernel
    std::cout << "Executing kernel..." << std::endl;
    kernel();
    std::cout << "Kernel executed successfully!" << std::endl;
  }
  return 0;
}