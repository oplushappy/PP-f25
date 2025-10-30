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

    // 1. allocate
    double *old_score = new double[N];
    double *new_score = new double[N];
    int *outdeg = new int[N];
    double *inv_outdeg  = new double[N];


    bool done = false;
    #pragma omp parallel
    {
        // 2️. 初始化：所有節點初始分數均為 1/N
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
            old_score[i] = 1.0 / N;

        // 3. 預先計算每個節點的出度（out-degree）
        #pragma omp for schedule(static)
        for (int v = 0; v < N; ++v)
            outdeg[v] = outgoing_size(g, v);

        // 把除法換成乘法：出度為 0 → 倒數設成 0（自然等同於分母無貢獻）
        #pragma omp for schedule(static)
        for (int v = 0; v < N; ++v)
            inv_outdeg[v] = (outdeg[v] > 0) ? (1.0 / (double)outdeg[v]) : 0.0;

        while (true) {
            // 4️. 主要迴圈（直到收斂）
            // (a) 計算 dangling mass：出度為 0 的節點分數總和
            double dangling_sum = 0.0;
            #pragma omp for reduction(+:dangling_sum) schedule(static)
            for (int v = 0; v < N; ++v)
                if (outdeg[v] == 0) dangling_sum += old_score[v];

            const double base = (1.0 - damping) / (double)N + damping * (dangling_sum / (double)N);
            double diff = 0.0;

            // (b) 計算每個節點的新分數
            #pragma omp for reduction(+:diff) schedule(guided,1024)
            for (int i = 0; i < N; ++i) {
                double inbound = 0.0;

                // 對於所有指向 i 的節點 j（即 j → i）
                const Vertex *beg = incoming_begin(g, i);
                const Vertex *end = incoming_end(g, i);

                // 走訪所有入邊 j→i：用乘法 + SIMD + reduction
                #pragma omp simd reduction(+:inbound)
                for (const Vertex *p = beg; p != end; ++p) { // 14.9%
                    int j = *p;
                    // int dj = outdeg[j]; // 4.07%, 記憶體存取亂序
                    // if (dj > 0) // 13.56%, 分支 if (dj > 0) 容易 mispredict
                    //     inbound += old_score[j] / (double)dj; // + 45.98%, / 5.71%, (double) 12.09%, 加法累積 addsd 變熱
                    inbound += old_score[j] * inv_outdeg[j];
                }

                // PageRank 三部分公式：
                // 1. teleport term
                // 2. dangling node redistribution term
                // 3. inbound contribution term
                // double val = 0.0;
                // val += (1.0 - damping) / N;
                // val += damping * (dangling_sum / N);
                // val += damping * inbound;
                double val = base + damping * inbound;
                new_score[i] = val;
                diff += fabs(val - old_score[i]);
            }

            // (c) 檢查收斂：計算新舊分數的平均差距
            // double diff = 0.0;
            // for (int i = 0; i < N; ++i)
            //     diff += std::fabs(new_score[i] - old_score[i]);

            // if (diff < convergence)
            //     break;

            // (d) 準備下一輪迭代
            // for (int i = 0; i < N; ++i)
            //     old_score[i] = new_score[i];
            // std::swap(old_score, new_score);
            #pragma omp single
            {
                done = diff < convergence;
                if (!done) std::swap(old_score, new_score);
            }
            #pragma omp barrier

            if (done) break;
        } 
    }


    for (int i = 0; i < N; ++i)
        solution[i] = new_score[i];

    // 5. free
    delete[] old_score;
    delete[] new_score;
    delete[] outdeg;
    delete[] inv_outdeg;
}
