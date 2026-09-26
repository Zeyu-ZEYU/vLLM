// C ABI over the planner cores, loaded from Python via ctypes (shunt.native).
#include "shunt_core.hpp"

extern "C" {

void shunt_lpt_schedule(const double* c, int n, int W, int* worker_of,
                        const double* init_load) {
    shunt::lpt_schedule(c, n, W, worker_of, init_load);
}

int shunt_balance_heads(double* t, const double* a, int W, double group_mean,
                        int H, double theta, int* donors, int* recipients) {
    return shunt::balance_heads(t, a, W, group_mean, H, theta, donors,
                                recipients);
}

void shunt_allocate_offload(const double* vol, int W, double B_be, double B_fe,
                            double bw_port, double bw_fe, const double* pcie,
                            int allow_borrow, int allow_frontend, double* own,
                            double* fe, double* borrow) {
    shunt::allocate_offload(vol, W, B_be, B_fe, bw_port, bw_fe, pcie,
                            allow_borrow != 0, allow_frontend != 0, own, fe,
                            borrow);
}

}  // extern "C"
