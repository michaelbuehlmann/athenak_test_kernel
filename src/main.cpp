#include "data.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

using DevExeSpace = Kokkos::DefaultExecutionSpace;
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using TeamMember_t = Kokkos::TeamPolicy<>::member_type;
using LayoutWrapper = Kokkos::LayoutRight;

constexpr int nmb = 8;
constexpr int nnghbr = 56;
constexpr int nvar = 22;
constexpr int nmnv = nmb * nnghbr * nvar;

void kernel(Kokkos::View<int **> nghbr_gid,
            Kokkos::View<int **> nghbr_lev,
            Kokkos::View<int **> sbuf_icoar,
            Kokkos::View<int **> sbuf_isame,
            Kokkos::View<int **> sbuf_ifine,
            Kokkos::View<int *> mblev) {
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank()) / (nnghbr * nvar);
    const int n = (tmember.league_rank() - m * (nnghbr * nvar)) / nvar;
    const int v = (tmember.league_rank() - m * (nnghbr * nvar) - n * nvar);

    // only load buffers when neighbor exists
    if (nghbr_gid(m, n) >= 0) {
      // if neighbor is at coarser level, use coar indices to pack buffer
      int il, iu, jl, ju, kl, ku;
      if (nghbr_lev(m, n) < mblev(m)) {
        il = sbuf_icoar(n, 0);
        iu = sbuf_icoar(n, 1);
        jl = sbuf_icoar(n, 2);
        ju = sbuf_icoar(n, 3);
        kl = sbuf_icoar(n, 4);
        ku = sbuf_icoar(n, 5);
        // if neighbor is at same level, use same indices to pack buffer
      } else if (nghbr_lev(m, n) == mblev(m)) {
        il = sbuf_isame(n, 0);
        iu = sbuf_isame(n, 1);
        jl = sbuf_isame(n, 2);
        ju = sbuf_isame(n, 3);
        kl = sbuf_isame(n, 4);
        ku = sbuf_isame(n, 5);
        // if neighbor is at finer level, use fine indices to pack buffer
      } else {
        il = sbuf_ifine(n, 0);
        iu = sbuf_ifine(n, 1);
        jl = sbuf_ifine(n, 2);
        ju = sbuf_ifine(n, 3);
        kl = sbuf_ifine(n, 4);
        ku = sbuf_ifine(n, 5);
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj = nk * nj;

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored
      // sequentially in MeshBlockPacks, so array index equals (target_id -
      // first_id)
      // int dm = nghbr.d_view(m, n).gid - mbgid.d_view(0);
      // int dn = nghbr.d_view(m, n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank
        // if (nghbr.d_view(m, n).rank == my_rank) {
        //   // if neighbor is at same or finer level, load data from u0
        //   if (nghbr.d_view(m, n).lev >= mblev.d_view(m)) {
        //     Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        //                          [&](const int i) {
        //                            rbuf[dn].vars(dm, (i - il + ni * (j - jl + nj * (k - kl + nk * v)))) = a(m, v, k, j, i);
        //                          });
        //     // if neighbor is at coarser level, load data from coarse_u0
        //   } else {
        //     Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        //                          [&](const int i) {
        //                            rbuf[dn].vars(dm, (i - il + ni * (j - jl + nj * (k - kl + nk * v)))) = ca(m, v, k, j, i);
        //                          });
        //   }

        //   // else copy into send buffer for MPI communication below

        // } else {
        //   // if neighbor is at same or finer level, load data from u0
        //   if (nghbr.d_view(m, n).lev >= mblev.d_view(m)) {
        //     Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        //                          [&](const int i) {
        //                            sbuf[n].vars(m, (i - il + ni * (j - jl + nj * (k - kl + nk * v)))) = a(m, v, k, j, i);
        //                          });
        //     // if neighbor is at coarser level, load data from coarse_u0
        //   } else {
        //     Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember, il, iu + 1),
        //                          [&](const int i) {
        //                            sbuf[n].vars(m, (i - il + ni * (j - jl + nj * (k - kl + nk * v)))) = ca(m, v, k, j, i);
        //                          });
        //   }
        // }
        tmember.team_barrier();
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer
  Kokkos::fence();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int **> nghbr_gid("nghbr_gid", nmb, nnghbr);
    Kokkos::View<int **> nghbr_lev("nghbr_lev", nmb, nnghbr);
    Kokkos::View<int **> sbuf_icoar("sbuf_icoar", nnghbr, 6);
    Kokkos::View<int **> sbuf_isame("sbuf_isame", nnghbr, 6);
    Kokkos::View<int **> sbuf_ifine("sbuf_ifine", nnghbr, 6);
    Kokkos::View<int *> mblev("mblev", nmb);

    // Load data from files
    std::string base_dir = "";
    if (argc == 2) {
      base_dir = std::string(argv[1]);
    } else {
      std::cerr << "Usage: ./main <path_to_data_folder>" << std::endl;
      return 1;
    }
    load_data(base_dir + "/nghbr_gid.txt", nghbr_gid);
    load_data(base_dir + "/nghbr_lev.txt", nghbr_lev);
    load_data(base_dir + "/sbuf_icoar.txt", sbuf_icoar);
    load_data(base_dir + "/sbuf_isame.txt", sbuf_isame);
    load_data(base_dir + "/sbuf_ifine.txt", sbuf_ifine);
    load_data(base_dir + "/mblev.txt", mblev);

    // Call the kernel
    kernel(nghbr_gid, nghbr_lev, sbuf_icoar, sbuf_isame, sbuf_ifine, mblev);
    std::cout << "Kernel executed successfully!" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}