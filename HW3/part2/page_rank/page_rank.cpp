#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    // int nnodes = num_nodes(g);
    // double equal_prob = 1.0 / nnodes;
    // for (int i = 0; i < nnodes; ++i)
    // {
    //     solution[i] = equal_prob;
    // }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */
    const int N = num_nodes(g);
    if (N == 0) return;

    std::vector<double> old_score(N), new_score(N);
    for (int i = 0; i < N; ++i) old_score[i] = 1/N; // 初始化為 1/N

    std::vector<int> outdeg(N);
    for (int v = 0; v < N; ++v)
        outdeg[v] = outgoing_size(g, v);

    while (true) {
        // 1️⃣ 計算 dangling mass：出度為 0 的節點分數總和
        double dangling_sum = 0.0;
        for (int v = 0; v < N; ++v)
            if (outdeg[v] == 0) dangling_sum += old_score[v];

        // 2️⃣ 計算每個節點的新分數
        for (int i = 0; i < N; ++i) {
            double inbound = 0.0;
            const Vertex *beg = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            for (const Vertex *p = beg; p != end; ++p) {
                int j = *p; // j → i
                int dj = outdeg[j];
                if (dj > 0) inbound += old_score[j] / (double)dj;
            }

            // 三個部分分開相加（順序固定）
            double val = 0.0;
            val += (1.0 - damping) / N;         // teleport
            val += damping * (dangling_sum / N); // dangling nodes 平均分配
            val += damping * inbound;            // 來自入邊的加權
            new_score[i] = val;
        }

        // 3️⃣ 計算差距以檢查收斂
        double diff = 0.0;
        for (int i = 0; i < N; ++i)
            diff += std::fabs(new_score[i] - old_score[i]);

        // 4️⃣ 收斂條件：使用平均差
        if ((diff / N) < convergence) break;

        std::swap(old_score, new_score);
    }

    // 5️⃣ 正規化結果（保險避免累積誤差）
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += new_score[i];
    double inv = (sum > 0.0) ? 1.0 / sum : 1.0 / N;

    for (int i = 0; i < N; ++i)
        solution[i] = new_score[i] * inv;
}
