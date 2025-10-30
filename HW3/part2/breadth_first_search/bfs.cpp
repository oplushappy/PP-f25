#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <algorithm>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            // 54.9%
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

// 每個 thread 先暫存在自己獨立的 local buffer（locals[tid]）裡，
// 最後再把所有 thread 的 local 結果「合併」到共用的 new_frontier。
void top_down_step_parallel(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances) {
    // 每執行緒一個 local 暫存，避免 new_frontier->count++ 的競爭
    int P = omp_get_max_threads();
    std::vector<std::vector<int>> locals(P);

    #pragma omp parallel for schedule(dynamic,64)
    for (int i = 0; i < frontier->count; i++) {
        int v = frontier->vertices[i];

        int start_edge = g->outgoing_starts[v];
        int end_edge   = (v == g->num_nodes - 1) ? g->num_edges: g->outgoing_starts[v + 1];

        int tid = omp_get_thread_num();
        auto& local = locals[tid];

        for (int e = start_edge; e < end_edge; ++e) {
            int n = g->outgoing_edges[e];

            // 先做便宜的快速失敗：已訪問就略過，不做 CAS
            if (distances[n] != NOT_VISITED_MARKER) continue;

            // 原子「認領」：只有第一個成功的人能設定距離與入隊
            if (__sync_bool_compare_and_swap(&distances[n], NOT_VISITED_MARKER, distances[v] + 1)) {
                local.push_back(n);
            }
        }
    }

    // 2) 合併所有 threads 的 local 結果到 new_frontier
    for (int t = 0; t < P; t++) {
        for (int node : locals[t]) {
            int idx = new_frontier->count++;          // 這裡單執行緒，安全
            new_frontier->vertices[idx] = node;
        }
    }
}

// 單步：bottom-up（序列版）
// in_frontier  : 當前層 frontier（bitmap, 0/1）
// next_frontier: 下一層 frontier（bitmap, 0/1）
// distances    : 到 root 的距離；未訪問為 NOT_VISITED_MARKER
// curr_depth   : 目前層數（frontier 的距離）
static void bottom_up_step_serial(Graph g, const unsigned char* in_frontier, unsigned char* next_frontier, int* distances, int curr_depth) {
    const int N = g->num_nodes;

    for (int v = 0; v < N; ++v) {
        if (distances[v] != NOT_VISITED_MARKER) continue; // 只看還沒訪問的點

        // 走 v 的「入邊」：誰指向 v（孩子找爸爸）
        int beg = g->incoming_starts[v];
        int end = (v == N - 1) ? g->num_edges : g->incoming_starts[v + 1];

        for (int ei = beg; ei < end; ++ei) {
            int u = g->incoming_edges[ei];
            if (in_frontier[u]) {                // u 在當前 frontier → v 被發現
                distances[v] = curr_depth + 1;   // 設 v 的距離
                next_frontier[v] = 1;            // v 進下一層（bitmap 置位）
                break;                           // 早停：找到任一父就夠了
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        // top_down_step(graph, frontier, new_frontier, sol->distances);
        top_down_step_parallel(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    
    const int N = graph->num_nodes;

    // 1) 初始化距離
    for (int i = 0; i < N; ++i) sol->distances[i] = NOT_VISITED_MARKER;

    // 2) frontier 用 bitmap（0/1）
    std::vector<unsigned char> in_frontier(N, 0), next_frontier(N, 0);

    // 3) root 入 frontier，距離設 0
    in_frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    int curr_depth = 0;

    // 4) 逐層擴張：當前 frontier 非空就持續
    while (true) {
#ifdef VERBOSE
        double t0 = CycleTimer::current_seconds();
#endif
        // 清空下一層 bitmap
        std::fill(next_frontier.begin(), next_frontier.end(), 0);

        // 做一層 bottom-up
        bottom_up_step_serial(graph,
                              in_frontier.data(),
                              next_frontier.data(),
                              sol->distances,
                              curr_depth);

#ifdef VERBOSE
        double t1 = CycleTimer::current_seconds();
#endif
        // 計下一層大小（也可省略改用旗標）
        int next_count = 0;
        for (int v = 0; v < N; ++v) next_count += next_frontier[v];

#ifdef VERBOSE
        printf("frontier=%-10d %.4f sec\n", next_count, t1 - t0);
#endif
        if (next_count == 0) break;       // 無新節點 → 結束

        in_frontier.swap(next_frontier);   // 下一層變成新一層
        curr_depth++;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
