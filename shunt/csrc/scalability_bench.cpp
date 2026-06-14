// Fig. 19: per-iteration decision cost of RS, EAP, and KVLB versus the DP worker
// count W (§4.5). EP degree is fixed in practice, so EAP and KVLB are node-local
// (8 workers) and flat; only the global RS grows with W. Under full load each DP
// worker holds ~32 requests, so RS schedules 32*W requests in one LPT pass.
//
// Pins to one core, times each algorithm, reports the median over trials.
// Build: make bench ; run: ./scalability_bench > scalability_results.txt
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include "shunt_core.hpp"

using clk = std::chrono::high_resolution_clock;
static double us(clk::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}
static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Median per-call time. Each trial runs the op `reps` times back to back and
// divides, so timer overhead is amortized for the sub-microsecond ops; `sink`
// consumes a result so the optimizer cannot elide the calls.
template <class F>
static double time_median(F&& f, int reps, int trials, volatile double& sink) {
    std::vector<double> t;
    t.reserve(trials);
    for (int k = 0; k < trials; ++k) {
        auto t0 = clk::now();
        double acc = 0.0;
        for (int r = 0; r < reps; ++r) acc += f();
        t.push_back(us(clk::now() - t0) / reps);
        sink += acc;
    }
    return median(t);
}

int main(int argc, char** argv) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(argc > 1 ? std::atoi(argv[1]) : 11, &set);  // pin to core 11 by default
    sched_setaffinity(0, sizeof(set), &set);
#endif
    const int REQ_PER_WORKER = 32;
    const int WPN = 8;            // workers per node (EAP/KVLB are node-local)
    const int H = 64;             // query heads
    const int TRIALS = 15;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(0.1, 3.0);
    volatile double sink = 0.0;

    // EAP and KVLB are node-local (8 workers), independent of W; measure once
    // with a fixed, representative skew (a straggler that actually triggers the
    // split) so the reported cost is stable and the same at every cluster size.
    std::vector<double> twk = {40, 6, 6, 5, 5, 5, 5, 5}, attn(WPN);
    for (int w = 0; w < WPN; ++w) attn[w] = twk[w] * 0.95;
    double gm = 0; for (double x : twk) gm += x; gm /= WPN;
    std::vector<int> hc(WPN); std::vector<double> post(WPN);
    double eap = time_median([&] {
        shunt::balance_heads(twk.data(), attn.data(), WPN, gm, H, 1.5,
                             hc.data(), post.data());
        return double(hc[0]);
    }, 50000, TRIALS, sink);

    // Common case: a hotspot overflows its own port and borrows spare backend
    // ports, but the KV still fits every budget (no full-overflow spread).
    std::vector<double> vol = {9e8, 3e7, 3e7, 3e7, 3e7, 3e7, 3e7, 3e7};
    std::vector<double> own(WPN), fe(WPN), borrow(WPN * WPN);
    double kv = time_median([&] {
        shunt::allocate_offload(vol.data(), WPN, 2e8, 2e8, 25e9, 25e9, nullptr,
                                own.data(), fe.data(), borrow.data());
        return own[0];
    }, 50000, TRIALS, sink);

    std::vector<int> Ws = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    printf("# W    RS_us       EAP_us    KVLB_us\n");
    for (int W : Ws) {
        int n = REQ_PER_WORKER * W;
        std::vector<double> c(n);
        for (auto& x : c) x = U(rng);
        std::vector<int> worker_of(n);
        int reps_rs = std::max(1, int(4'000'000 / n));   // amortize fixed cost
        double rs = time_median([&] {
            shunt::lpt_schedule(c.data(), n, W, worker_of.data());
            return worker_of[0];
        }, reps_rs, TRIALS, sink);
        printf("%-6d %-11.4f %-9.4f %-9.4f\n", W, rs, eap, kv);
        fflush(stdout);
    }
    return sink < 0 ? 1 : 0;  // keep `sink` live
}
