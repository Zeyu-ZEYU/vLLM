// Shunt's per-iteration algorithms in C++: LPT placement, attention-head
// balancing (Algorithm S1), and KV offload allocation (Algorithm 1).
//
// Header-only. The rules and tie-breaking match shunt/algorithms.py exactly;
// tests/test_native_parity.py checks both against each other.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace shunt {

constexpr double kEps = 1e-9;

// ---- LPT placement ---------------------------------------------------------
// Heaviest request first (ties: lower index); each goes to the least-loaded
// worker (ties: lower worker index). A min-heap keeps it O(n log W).
inline void lpt_schedule(const double* c, int n, int W, int* worker_of) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return c[a] > c[b]; });
    using LW = std::pair<double, int>;
    auto cmp = [](const LW& a, const LW& b) {
        return a.first != b.first ? a.first > b.first : a.second > b.second;
    };
    std::priority_queue<LW, std::vector<LW>, decltype(cmp)> pq(cmp);
    for (int w = 0; w < W; ++w) pq.push({0.0, w});
    for (int i : order) {
        LW top = pq.top();
        pq.pop();
        worker_of[i] = top.second;
        pq.push({top.first + c[i], top.second});
    }
}

// ---- Algorithm S1: attention-head balancing on one node ----------------------
// t[W]: worker-compute times (updated in place to the post-split times);
// a[W]: attention times. Writes the moves (donor, recipient) in order and
// returns their count; donors/recipients need room for W*H entries.
inline int balance_heads(double* t, const double* a, int W, double group_mean,
                         int H, double theta, int* donors, int* recipients) {
    if (W <= 0 || H <= 0) return 0;
    double mx = t[0];
    for (int w = 1; w < W; ++w) mx = std::max(mx, t[w]);
    if (mx <= theta * group_mean) return 0;
    std::vector<double> e(W);
    for (int w = 0; w < W; ++w) e[w] = a[w] / H;
    std::vector<int> own(W, H);
    int n = 0;
    while (true) {
        int s = 0, d = 0;
        for (int w = 1; w < W; ++w) {
            if (t[w] > t[s]) s = w;
            if (t[w] < t[d]) d = w;
        }
        if (own[s] == 0 || e[s] <= 0.0 || t[d] + e[s] >= t[s]) break;
        t[s] -= e[s];
        t[d] += e[s];
        own[s] -= 1;
        donors[n] = s;
        recipients[n] = d;
        ++n;
    }
    return n;
}

// ---- Algorithm 1: KV offload allocation, one node, one direction ------------
// Exact water-filling: add `extra` bytes over n links so the latest finish time
// load/bw is minimal. Writes the added bytes per link.
inline void waterfill(const double* load, const double* bw, int n, double extra,
                      double* added) {
    for (int i = 0; i < n; ++i) added[i] = 0.0;
    if (extra <= kEps || n == 0) return;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
        return load[x] / bw[x] < load[y] / bw[y];
    });
    double cum_b = 0.0, cum_l = 0.0, level = 0.0;
    int k = 0;
    for (k = 0; k < n; ++k) {
        int i = order[k];
        cum_b += bw[i];
        cum_l += load[i];
        level = (extra + cum_l) / cum_b;
        double nxt = (k + 1 < n)
                         ? load[order[k + 1]] / bw[order[k + 1]]
                         : std::numeric_limits<double>::infinity();
        if (level <= nxt) break;
    }
    if (k == n) k = n - 1;
    for (int j = 0; j <= k; ++j) {
        int i = order[j];
        added[i] = std::max(0.0, level * bw[i] - load[i]);
    }
}

// vol[W]: bytes per worker. Outputs own[W], fe[W], borrow[W*W]
// (borrow[w*W + p] = bytes worker w sends through port p != w).
// pcie: W*W distance matrix (row-major) or nullptr for |i-j|.
inline void allocate_offload(const double* vol, int W, double B_be, double B_fe,
                             double bw_port, double bw_fe, const double* pcie,
                             bool allow_borrow, bool allow_frontend,
                             double* own, double* fe, double* borrow) {
    std::vector<double> spare(W), residual(W);
    for (int i = 0; i < W * W; ++i) borrow[i] = 0.0;
    for (int w = 0; w < W; ++w) {
        own[w] = std::min(vol[w], B_be);
        spare[w] = B_be - own[w];
        residual[w] = std::max(0.0, vol[w] - B_be);
        fe[w] = 0.0;
    }
    auto dist = [&](int i, int j) {
        return pcie ? pcie[i * W + j] : double(std::abs(i - j));
    };
    auto by_residual = [&]() {
        std::vector<int> o(W);
        std::iota(o.begin(), o.end(), 0);
        std::stable_sort(o.begin(), o.end(),
                         [&](int x, int y) { return residual[x] > residual[y]; });
        return o;
    };
    double fe_left = allow_frontend ? B_fe : 0.0;
    for (int w : by_residual()) {
        if (residual[w] <= kEps) continue;
        if (allow_borrow) {
            std::vector<int> ports(W);
            std::iota(ports.begin(), ports.end(), 0);
            std::stable_sort(ports.begin(), ports.end(),
                             [&](int x, int y) { return dist(w, x) < dist(w, y); });
            for (int p : ports) {
                if (residual[w] <= kEps) break;
                if (p == w || spare[p] <= kEps) continue;
                double take = std::min(residual[w], spare[p]);
                borrow[w * W + p] += take;
                spare[p] -= take;
                residual[w] -= take;
            }
        }
        if (allow_frontend && residual[w] > kEps && fe_left > kEps) {
            double take = std::min(residual[w], fe_left);
            fe[w] += take;
            fe_left -= take;
            residual[w] -= take;
        }
    }
    // Beyond every budget: spread over the allowed links, balancing finish times.
    std::vector<double> port_load(W);
    for (int p = 0; p < W; ++p) {
        double s = 0.0;
        for (int w = 0; w < W; ++w) s += borrow[w * W + p];
        port_load[p] = own[p] + s;
    }
    double fe_load = 0.0;
    for (int w = 0; w < W; ++w) fe_load += fe[w];
    std::vector<int> links;
    std::vector<double> loads, bws, added;
    for (int w : by_residual()) {
        if (residual[w] <= kEps) continue;
        links.clear();
        loads.clear();
        bws.clear();
        for (int p = 0; p < W; ++p) {
            if (p == w || allow_borrow) {
                links.push_back(p);
                loads.push_back(port_load[p]);
                bws.push_back(bw_port);
            }
        }
        if (allow_frontend) {
            loads.push_back(fe_load);
            bws.push_back(bw_fe);
        }
        added.assign(loads.size(), 0.0);
        waterfill(loads.data(), bws.data(), int(loads.size()), residual[w],
                  added.data());
        for (size_t k = 0; k < links.size(); ++k) {
            if (added[k] <= 0.0) continue;
            int p = links[k];
            if (p == w)
                own[w] += added[k];
            else
                borrow[w * W + p] += added[k];
            port_load[p] += added[k];
        }
        if (allow_frontend && added.back() > 0.0) {
            fe[w] += added.back();
            fe_load += added.back();
        }
        residual[w] = 0.0;
    }
}

}  // namespace shunt
