// C ABI over the planner cores, loaded from Python via ctypes (shunt._native).
// Zero external dependencies: just compile this to a shared library.
#include "shunt_core.hpp"

extern "C" {

void shunt_lpt_schedule(const double* c, int n, int W, int* worker_of) {
    shunt::lpt_schedule(c, n, W, worker_of);
}

void shunt_balance_heads(const double* twk, const double* attn, int W,
                         double group_mean, int H, double theta,
                         int* head_counts, double* post) {
    shunt::balance_heads(twk, attn, W, group_mean, H, theta, head_counts, post);
}

void shunt_allocate_offload(const double* vol, int W, double B_be, double B_fe,
                            double bw_port, double bw_fe, const double* pcie,
                            double* own, double* fe, double* borrow) {
    shunt::allocate_offload(vol, W, B_be, B_fe, bw_port, bw_fe, pcie,
                            own, fe, borrow);
}

}  // extern "C"
