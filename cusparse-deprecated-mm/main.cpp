#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <cusparse.h>
#include <library_types.h>
#include <cuda_runtime.h>

#define N 2
#define K 3
#define M ((N)*(K)-1)
#define NNZ ((N)*(K)-1)

#define frand() ((float)rand()/(RAND_MAX+1.0))

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
      throw; 
  }
}

template<typename T> 
void csrmm2(
  char transa, char transb, 
  int64_t m, int64_t n, int64_t k, int64_t nnz, 
  T alpha, T *csrvala, int *csrrowptra, int *csrcolinda, 
  T *b, int64_t ldb, T beta, T *c, int64_t ldc)
{
  static_assert(std::is_same<float, T>::value); 
  constexpr auto cusparse_value_type = CUDA_R_32F; 

  if (csrvala == nullptr || b == nullptr || c == nullptr) return; 

  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  printf("\n%d %d %d %d %d\n", m, n, k, ldb, ldc);

  int* h1 = (int*)malloc(sizeof(int)*(m+1)); 
  cudaMemcpy(h1, csrrowptra, sizeof(int)*(m+1), cudaMemcpyDeviceToHost); 
  for(int i=0; i<m+1; i++) printf("%d ", h1[i]);
  printf("\n");
  free(h1); 

  int* h2 = (int*)malloc(sizeof(int)*(nnz)); 
  cudaMemcpy(h2, csrcolinda, sizeof(int)*(nnz), cudaMemcpyDeviceToHost); 
  for(int i=0; i<nnz; i++) printf("%d ", h2[i]);
  printf("\n");
  free(h2); 

  T* h3 = (T*)malloc(sizeof(T)*(nnz)); 
  cudaMemcpy(h3, csrvala, sizeof(T)*(nnz), cudaMemcpyDeviceToHost); 
  for(int i=0; i<nnz; i++) printf("%f ", h3[i]);
  printf("\n");
  free(h3); 

  T* h4 = (T*)malloc(sizeof(T)*(k*n)); 
  cudaMemcpy(h4, b, sizeof(T)*(k*n), cudaMemcpyDeviceToHost); 
  for(int i=0; i<k*n; i++) printf("%f ", h4[i]);
  printf("\n");
  free(h4); 

  int64_t ma = m, ka = k; 
  if (transa != 'n') std::swap(ma, ka); 

  cusparseSpMatDescr_t descA; 
  cusparseCreateCsr(
    &descA,                     /* output */
    ma, ka, nnz,                /* rows, cols, number of non zero elements */
    csrrowptra,                 /* row offsets of the sparse matrix, size = rows +1 */
    csrcolinda,                 /* column indices of the sparse matrix, size = nnz */
    csrvala,                    /* values of the sparse matrix, size = nnz */
    CUSPARSE_INDEX_32I,         /* data type of row offsets index */
    CUSPARSE_INDEX_32I,         /* data type of col indices */
    CUSPARSE_INDEX_BASE_ZERO,   /* base index of row offset and col indes */
    cusparse_value_type         /* data type of values */
  ); 

  int64_t kb = k, nb = n;
  if (transb != 'n') std::swap(kb, nb); 

  cusparseDnMatDescr_t descB; 
  cusparseCreateDnMat(
    &descB,               /* output */
    kb, nb, ldb,          /* rows, cols, leading dimension */
    b,                    /* values */
    cusparse_value_type,  /* data type of values */
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ); 

  cusparseDnMatDescr_t descC; 
  cusparseCreateDnMat(
    &descC,               /* output */
    m, n, ldc,            /* rows, cols, leading dimension */
    c,                    /* values */ 
    cusparse_value_type,  /* data type of values */ 
    CUSPARSE_ORDER_COL    /* memory layout, ONLY column-major is supported now */
  ); 


  cusparseHandle_t handle; 
  cusparseCreate(&handle); 

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize; 
  cusparseSpMM_bufferSize(
    handle, opa, opb,     
    &alpha,               
    descA, descB, 
    &beta, 
    descC, 
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    &bufferSize           /* output */
  ); 

  void* externalBuffer; // device pointer
  cudaMalloc(&externalBuffer, bufferSize); 

  cusparseSpMM(
    handle, opa, opb, 
    &alpha, 
    descA, descB, 
    &beta, 
    descC, 
    cusparse_value_type,  /* data type in which the computation is executed */
    CUSPARSE_CSRMM_ALG1,  /* default computing algorithm for CSR sparse matrix format */
    externalBuffer        /* external buffer */
  ); 

  cudaFree(externalBuffer); 
  cusparseDestroySpMat(descA); 
  cusparseDestroyDnMat(descB); 
  cusparseDestroyDnMat(descC); 

  cudaDeviceSynchronize();

  T* h5 = (T*)malloc(sizeof(T)*(m*n)); 
  cudaMemcpy(h5, c, sizeof(T)*(m*n), cudaMemcpyDeviceToHost); 
  for(int i=0; i<m*n; i++) printf("%f ", h5[i]);
  printf("\n");
  free(h5); 

  cusparseDestroy(handle); 
}

int main(){
    int *d_row, *d_col; 
    float *d_val, *d_b, *d_c; 
    
    std::vector<int> row; 
    for(int i=0; i<=M; i++) row.push_back(i); 

    std::vector<int> col; 
    for(int i=0; i<NNZ; i++) col.push_back(i % K); 

    std::vector<float> val;
    for(int i=0; i<NNZ; i++) val.push_back(frand()); 

    std::vector<float> b;
    for(int i=0; i<K*N; i++) b.push_back(frand()); 

    std::vector<float> c(M*N, 0.0); 
    
    cudaMalloc(&d_row, sizeof(int)*(M+1)); 
    cudaMemcpy(d_row, row.data(), sizeof(int)*(M+1), cudaMemcpyHostToDevice); 

    cudaMalloc(&d_col, sizeof(int)*(NNZ)); 
    cudaMemcpy(d_col, col.data(), sizeof(int)*(NNZ), cudaMemcpyHostToDevice); 

    cudaMalloc(&d_val, sizeof(float)*(NNZ)); 
    cudaMemcpy(d_val, val.data(), sizeof(float)*(NNZ), cudaMemcpyHostToDevice); 

    cudaMalloc(&d_b, sizeof(float)*(K*N)); 
    cudaMemcpy(d_b, b.data(), sizeof(float)*(K*N), cudaMemcpyHostToDevice); 

    cudaMalloc(&d_c, sizeof(float)*(M*N)); 
    cudaMemcpy(d_c, c.data(), sizeof(float)*(M*N), cudaMemcpyHostToDevice); 

    int64_t ldb = std::min(N, K);
    int64_t ldc = std::max(M, N); 

    csrmm2<float>('n', 't', M, N, K, NNZ, 1.0, d_val, d_row, d_col, d_b, ldb, 1.0, d_c, ldc); 

    cudaFree(d_row); 
    cudaFree(d_col); 
    cudaFree(d_val); 
    cudaFree(d_b); 
    cudaFree(d_c); 
    return 0; 
}