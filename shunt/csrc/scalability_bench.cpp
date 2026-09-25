// Per-iteration decision cost of LPT placement (RS), attention-head balancing
// (EAP), and KV offload allocation (KVLB) versus the number of DP workers.
//
// RS places `req_per_worker * W` requests on W workers in one LPT pass. EAP and
// KVLB are node-local: EAP balances one node's workers and KVLB allocates both
// KV directions of one node, so their inputs do not depend on W.
//
// Usage: scalability_bench [--cpu N] [--req-per-worker R] [--trials T]
//                          [--workers W1,W2,...]
// Prints CSV: workers,rs_us,eap_us,kvlb_us (median per-call time).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include "shunt_core.hpp"

using clk = std::chrono::steady_clock;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Median over trials of the per-call time; each trial repeats the call `reps`
// times so the timer cost is amortized for sub-microsecond calls.
template <class F>
static double time_us(F&& f, int reps, int trials, volatile double& sink) {
    std::vector<double> t;
    for (int k = 0; k < trials; ++k) {
        auto t0 = clk::now();
        double acc = 0.0;
        for (int r = 0; r < reps; ++r) acc += f();
        auto dt = std::chrono::duration<double, std::micro>(clk::now() - t0);
        t.push_back(dt.count() / reps);
        sink = sink + acc;
    }
    return median(t);
}

int main(int argc, char** argv) {
    int cpu = -1, req_per_worker = 32, trials = 15;
    std::vector<int> Ws = {16, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--cpu") cpu = std::atoi(next().c_str());
        else if (a == "--req-per-worker") req_per_worker = std::atoi(next().c_str());
        else if (a == "--trials") trials = std::atoi(next().c_str());
        else if (a == "--workers") {
            Ws.clear();
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ',')) Ws.push_back(std::atoi(tok.c_str()));
        }
    }
#ifdef __linux__
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    const int WPN = 8, H = 64;
    const double theta = 1.5;
    volatile double sink = 0.0;
    std::mt19937 rng(12345);

    // EAP: one node whose first worker is a straggler, the rest light.
    const std::vector<double> twk0 = {40e-3, 6e-3, 6e-3, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3};
    std::vector<double> attn(WPN), twk(WPN);
    for (int w = 0; w < WPN; ++w) attn[w] = 0.95 * twk0[w];
    double group_mean = 0.0;
    for (double x : twk0) group_mean += x;
    group_mean /= WPN;
    std::vector<int> donors(WPN * H), recips(WPN * H);
    double eap = time_us([&] {
        std::copy(twk0.begin(), twk0.end(), twk.begin());
        return double(shunt::balance_heads(twk.data(), attn.data(), WPN, group_mean,
                                           H, theta, donors.data(), recips.data()));
    }, 20000, trials, sink);

    // KVLB: both directions of one node; one hot port overflows and borrows.
    const std::vector<double> vin = {9e8, 3e7, 3e7, 3e7, 3e7, 3e7, 3e7, 3e7};
    const std::vector<double> vout = {6e8, 4e7, 4e7, 3e7, 3e7, 3e7, 3e7, 3e7};
    std::vector<double> own(WPN), fe(WPN), borrow(WPN * WPN);
    double kvlb = time_us([&] {
        shunt::allocate_offload(vin.data(), WPN, 2e8, 2e8, 25e9, 25e9, nullptr, true,
                                true, own.data(), fe.data(), borrow.data());
        double x = own[0];
        shunt::allocate_offload(vout.data(), WPN, 2e8, 2e8, 25e9, 25e9, nullptr,
                                true, true, own.data(), fe.data(), borrow.data());
        return x + own[0];
    }, 20000, trials, sink);

    std::uniform_real_distribution<double> U(0.1, 3.0);
    std::printf("workers,rs_us,eap_us,kvlb_us\n");
    for (int W : Ws) {
        int n = req_per_worker * W;
        std::vector<double> c(n);
        for (auto& x : c) x = U(rng);
        std::vector<int> worker_of(n);
        int reps = std::max(1, 2000000 / n);
        double rs = time_us([&] {
            shunt::lpt_schedule(c.data(), n, W, worker_of.data());
            return double(worker_of[0]);
        }, reps, trials, sink);
        std::printf("%d,%.4f,%.4f,%.4f\n", W, rs, eap, kvlb);
        std::fflush(stdout);
    }
    return sink < -1.0 ? 1 : 0;
}
