#include "bfs.h"

#include <cstdlib>
#include <omp.h>

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
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
