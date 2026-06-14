// Shunt planner cores in C++ (§3.2-§3.4): the microsecond-scale decision path
// the paper times (Figs 18-19). Header-only; identical in behaviour to the
// Python reference in shunt/algorithms.py (a parity test pins them together).
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace shunt {

// ---- Algorithm 1: compute-aware request scheduling (RS), §3.2 --------------
// LPT: heaviest request first onto the lightest worker so far. A min-heap of
// (load, worker) keeps it O(n log W) so it scales to thousands of workers
// (Fig. 19); ties break to the lowest worker index, matching the Python
// reference. worker_of has length n.
inline void lpt_schedule(const double* c, int n, int W, int* worker_of) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return c[a] > c[b]; });
    using PLW = std::pair<double, int>;          // (load, worker)
    auto cmp = [](const PLW& a, const PLW& b) {   // min by load, then by index
        return a.first != b.first ? a.first > b.first : a.second > b.second;
    };
    std::priority_queue<PLW, std::vector<PLW>, decltype(cmp)> pq(cmp);
    for (int w = 0; w < W; ++w) pq.push({0.0, w});
    for (int i : order) {
        auto [load, w] = pq.top();
        pq.pop();
        worker_of[i] = w;
        pq.push({load + c[i], w});
    }
}

// ---- Algorithm 2: elastic attention head balancing (EAP), §3.3 -------------
// One prefill node. Fires only when the node's worst worker-compute exceeds
// theta * group_mean; then hands one query head at a time from busiest to
// idlest until within one head of the node mean. head_counts and post (the
// post-split worker-compute times) have length W; post may be null.
inline void balance_heads(const double* twk, const double* attn, int W,
                          double group_mean, int H, double theta,
                          int* head_counts, double* post) {
    std::vector<double> t(twk, twk + W);
    for (int w = 0; w < W; ++w) head_counts[w] = H;
    double mx = *std::max_element(t.begin(), t.end());
    if (mx <= theta * group_mean) {
        if (post) std::copy(t.begin(), t.end(), post);
        return;
    }
    std::vector<double> e(W);
    for (int w = 0; w < W; ++w) e[w] = attn[w] / H;
    std::vector<int> h(W, H);
    while (true) {
        int s = int(std::max_element(t.begin(), t.end()) - t.begin());
        int d = int(std::min_element(t.begin(), t.end()) - t.begin());
        if (t[s] - t[d] <= e[s] || h[s] <= 0) break;
        t[s] -= e[s]; t[d] += e[s];
        h[s] -= 1;    h[d] += 1;
    }
    for (int w = 0; w < W; ++w) head_counts[w] = h[w];
    if (post) std::copy(t.begin(), t.end(), post);
}

// Water-fill `extra` total bytes across links to minimize the max finish time
// (load+added)/bw. Returns per-link added bytes in `added` (length n).
inline void waterfill(const double* load, const double* bw, int n, double extra,
                      double* added) {
    for (int i = 0; i < n; ++i) added[i] = 0.0;
    if (extra <= 0) return;
    double sum_bw = 0.0, hi0 = 0.0;
    for (int i = 0; i < n; ++i) { sum_bw += bw[i]; hi0 = std::max(hi0, load[i] / bw[i]); }
    double lo = 0.0, hi = hi0 + extra / sum_bw;
    for (int it = 0; it < 100; ++it) {
        double T = 0.5 * (lo + hi), need = 0.0;
        for (int i = 0; i < n; ++i) need += std::max(0.0, T * bw[i] - load[i]);
        if (need < extra) lo = T; else hi = T;
    }
    for (int i = 0; i < n; ++i) added[i] = std::max(0.0, hi * bw[i] - load[i]);
}

// ---- Algorithm 3: KV traffic load balancing (KVLB), one direction, §3.4 ----
// vol[W] worker KV bytes. Outputs: own[W] (bytes on own port), fe[W] (frontend
// bytes per worker), borrow[W*W] (borrow[w*W+lender] = w's bytes on lender's
// port, lender != w). pcie is a W*W distance matrix (row-major); null -> |i-j|.
inline void allocate_offload(const double* vol, int W, double B_be, double B_fe,
                             double bw_port, double bw_fe, const double* pcie,
                             double* own, double* fe, double* borrow) {
    std::vector<double> spare(W), residual(W);
    for (int i = 0; i < W * W; ++i) borrow[i] = 0.0;
    for (int w = 0; w < W; ++w) {
        own[w] = std::min(vol[w], B_be);
        fe[w] = 0.0;
        spare[w] = B_be - own[w];
        residual[w] = std::max(0.0, vol[w] - B_be);
    }
    auto dist = [&](int i, int j) { return pcie ? pcie[i * W + j] : double(std::abs(i - j)); };

    std::vector<int> byres(W);
    std::iota(byres.begin(), byres.end(), 0);
    std::sort(byres.begin(), byres.end(), [&](int a, int b) {
        return residual[a] != residual[b] ? residual[a] > residual[b] : a < b;
    });
    double fe_left = B_fe;
    for (int w : byres) {
        if (residual[w] <= 0) continue;
        std::vector<int> lenders(W);
        std::iota(lenders.begin(), lenders.end(), 0);
        std::sort(lenders.begin(), lenders.end(), [&](int a, int b) {
            double da = dist(w, a), db = dist(w, b);
            return da != db ? da < db : a < b;
        });
        for (int l : lenders) {
            if (residual[w] <= 1e-12) break;
            if (l == w || spare[l] <= 0) continue;
            double take = std::min(residual[w], spare[l]);
            borrow[w * W + l] += take;
            spare[l] -= take;
            residual[w] -= take;
        }
        if (residual[w] > 1e-12 && fe_left > 0) {
            double take = std::min(residual[w], fe_left);
            fe[w] += take; fe_left -= take; residual[w] -= take;
        }
    }
    // overflow past every budget: water-fill across all links by finish time
    double total_extra = 0.0;
    for (int w = 0; w < W; ++w) total_extra += residual[w];
    if (total_extra > 1e-9) {
        std::vector<double> committed(W + 1), bw(W + 1), added(W + 1);
        for (int p = 0; p < W; ++p) { committed[p] = B_be - spare[p]; bw[p] = bw_port; }
        committed[W] = B_fe - fe_left; bw[W] = bw_fe;
        waterfill(committed.data(), bw.data(), W + 1, total_extra, added.data());
        // donors ordered by the *post-borrow* residual (matches the reference)
        std::vector<int> donors;
        for (int w = 0; w < W; ++w)
            if (residual[w] > 1e-12) donors.push_back(w);
        std::sort(donors.begin(), donors.end(), [&](int a, int b) {
            return residual[a] != residual[b] ? residual[a] > residual[b] : a < b;
        });
        int di = 0;
        for (int link = 0; link <= W && di < (int)donors.size(); ++link) {
            double give = added[link];
            while (give > 1e-12 && di < (int)donors.size()) {
                int w = donors[di];
                if (residual[w] <= 1e-12) { ++di; continue; }
                double take = std::min(give, residual[w]);
                if (link < W) {
                    if (link == w) own[w] += take; else borrow[w * W + link] += take;
                } else {
                    fe[w] += take;
                }
                residual[w] -= take; give -= take;
                if (residual[w] <= 1e-12) ++di;
            }
        }
    }
}

}  // namespace shunt
