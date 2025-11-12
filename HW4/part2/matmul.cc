#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int rank,size; 
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto rows_of=[&](int r){ return (n * (r + 1)) / size - (n * r) / size; };
    auto row_disp=[&](int r){ return (n * r) / size; };

    int local_rows = rows_of(rank);

    *a_mat_ptr = local_rows ? (int*)malloc(sizeof(int) * local_rows * m) : nullptr;
    *b_mat_ptr = (m > 0 && l > 0) ? (int*)malloc(sizeof(int) * m * l) : nullptr;

    int *sendcounts=nullptr,*displs=nullptr;
    
    if(rank==0){
        sendcounts=(int*)malloc(sizeof(int) * size);
        displs=(int*)malloc(sizeof(int) * size);
        for(int r = 0; r < size; r++){ 
            sendcounts[r]= rows_of(r) * m; // sendcounts：紀錄每個 rank 要被分配的資料數量。 
            displs[r]= row_disp(r) * m; // displs：紀錄每個 rank 在原始矩陣 A（a_mat）裡的起始位置（offset）。
        }
    }
    // 由 rank 0 將整個 a_mat（完整矩陣 A）切分後，按照 sendcounts 和 displs 這兩張表，分別發送到各個 rank 的 *a_mat_ptr 裡。
    // MPI_Scatterv, MPI_Iscatterv - Scatters a buffer in parts to all tasks in a group.
    // int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
    MPI_Scatterv(a_mat, sendcounts, displs, MPI_INT, *a_mat_ptr, local_rows * m, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(rank == 0){ 
        free(sendcounts); 
        free(displs); 
    }

    // B 的第 i 列第 j 欄，搬到 BT 的第 j 欄第 i 列
    if(rank == 0 && *b_mat_ptr && b_mat) {
        // memcpy(*b_mat_ptr, b_mat, sizeof(int) * m * l);
        int* BT = *b_mat_ptr;               // 轉置後存放位置
        const int* B = b_mat;               // 原始 B
        for (int row = 0; row < m; row++) {
            const int* Bi = B + row * l;      // 原始 B 的第 i 列起點
            for (int j = 0; j < l; ++j) {
                // 把 B(i,j) 放到 BT 的 (j,i)
                // BT 是以「欄 j 連續」的布局：第 j 欄起點 = BT + j*m
                // 於是 (j,i) 的位移 = j*m + i
                BT[j * m + row] = Bi[j];
            }
        }
    }

    // MPI_Bcast, MPI_Ibcast - Broadcasts a message from the process with rank root to all other processes of the group.
    // int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int rank,size; 
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto rows_of=[&](int r){ return (n * (r + 1)) / size - (n * r) / size; };
    auto row_disp=[&](int r){ return (n * r) / size; };
    int local_rows = rows_of(rank);

    int local_elems = local_rows * l;
    int* c_local = local_elems ? (int*)malloc(sizeof(int) * local_elems) : nullptr;

    // 乘法：A 為 row-major、B 為「欄連續」(每欄長度 m)
    for (int i = 0; i < local_rows; i++) {
        const int *Ai = a_mat + i * m;   // 本地第 i 列
        int *Ci       = c_local + i * l; // 對應輸出第 i 列
        for (int j = 0; j < l; j++) {
            const int *Bj = b_mat + j * m;   // 第 j 欄（長度 m）連續
            long long acc = 0;               // 防止中途乘加溢位
            for (int k = 0; k < m; ++k) {
                acc += (long long)Ai[k] * Bj[k];
            }
            Ci[j] = (int)acc;
        }
    }

    // 收回到 rank 0
    // rank 0 先準備好「每個人要傳多少」的表格
    int *recvcounts = nullptr, *displs = nullptr;
    if (rank == 0) {
        recvcounts = (int*)malloc(sizeof(int) * size);
        displs     = (int*)malloc(sizeof(int) * size);
        for (int r = 0; r < size; ++r) {
            recvcounts[r] = rows_of(r) * l;
            displs[r]     = row_disp(r) * l;
        }
    }

    // MPI_Gatherv, MPI_Igatherv - Gathers varying amounts of data from all processes to the root process
    // int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm)
    // MPI_Gatherv(
    //     c_local,                 // [每個 rank] 自己的發送緩衝區
    //     local_rows * l,          // [每個 rank] 自己要送幾個元素
    //     MPI_INT,                 // [每個 rank] 資料型別
    //     out_mat,                 // [只有 rank 0 有效] 接收完整結果的緩衝區
    //     recvcounts,              // [只有 rank 0 有效] 各 rank 傳來的元素數量表
    //     displs,                  // [只有 rank 0 有效] 各 rank 結果在 out_mat 中的偏移
    //     MPI_INT,                 // [只有 rank 0 有效] 接收資料型別
    //     0,                       // root（誰是接收者）= rank 0
    //     MPI_COMM_WORLD
    // );
    MPI_Gatherv(c_local, local_rows * l, MPI_INT, out_mat, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0){ 
        free(recvcounts); 
        free(displs); 
    }

    if(c_local) free(c_local);
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* TODO */
    if(a_mat) free(a_mat);
    if(b_mat) free(b_mat);
}
