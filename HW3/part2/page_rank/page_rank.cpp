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

    // solution[] 已在骨架中設為 1/N，這裡把它當成初始 old[]
    double *old_score = new double[N];
    double *new_score = new double[N];
    for (int i = 0; i < N; ++i) old_score[i] = solution[i];

    // 預先計算出度
    int *outdeg = new int[N];
    for (int v = 0; v < N; ++v) outdeg[v] = outgoing_size(g, v);

    while (true) {
        // 1) 計算 dangling mass（出度為 0 的節點分數總和）
        double dangling_sum = 0.0;
        for (int v = 0; v < N; ++v) {
            if (outdeg[v] == 0) dangling_sum += old_score[v];
        }

        // 2) 每個節點都會拿到的底座（teleport + 平均分配 dangling）
        const double base = (1.0 - damping) / N + damping * (dangling_sum / N);

        // 3) 逐點累加入邊貢獻，並累計本輪變化量（用於收斂判定）
        double diff = 0.0;
        for (int i = 0; i < N; ++i) {
            double acc = 0.0;

            const Vertex *beg = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            for (const Vertex *p = beg; p != end; ++p) {
                int j = *p;                 // j -> i
                int dj = outdeg[j];
                if (dj > 0) acc += old_score[j] / dj; // 非 dangling 的入邊才在這裡加
            }

            new_score[i] = base + damping * acc;
            diff += std::fabs(new_score[i] - old_score[i]);
        }

        // 4) 收斂判定
        if (diff < convergence) break;

        // 5) 下一輪
        std::swap(old_score, new_score);
    }

    // 輸出回 solution
    for (int i = 0; i < N; ++i) solution[i] = old_score[i];

    delete[] outdeg;
    delete[] old_score;
    delete[] new_score;
}
