#include <iostream>
#include <iomanip>
#include <exception>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <magma_v2.h>
#include <magma_types.h>

#define B 4
#define N 32

#define LDA N

#define NITER 200
#define NITER_WARMUP 50

#ifdef REUSE_MAGMA_QUEUE
#pragma message("reuse magma queue enabled")
#endif

#define CUDA_CHECK(EXPR)                                                                        \
    do {                                                                                        \
        cudaError_t __expr = EXPR;                                                              \
        if (__expr != cudaSuccess) {                                                            \
            printf("CUDA failure at line: %d, function: %s\n", __LINE__, __PRETTY_FUNCTION__);  \
            throw std::runtime_error(std::string("cuda error number: ")                         \
                + std::string(cudaGetErrorString(__expr)));                                     \
        }                                                                                       \
    } while(0)

#define CUSOLVER_CHECK(EXPR)                                                                        \
    do {                                                                                            \
        cusolverStatus_t __expr = EXPR;                                                             \
        if (__expr != CUSOLVER_STATUS_SUCCESS) {                                                    \
            printf("cusolver failure at line: %d, function: %s\n", __LINE__, __PRETTY_FUNCTION__);  \
            throw std::runtime_error(std::string("cusolver error number: ")                         \
                + std::to_string(__expr));                                                          \
        }                                                                                           \
    } while(0)

// print the batch
template<class T>
void print_array(T* hx, size_t nelem, const std::vector<int>& dividers) {
    std::cout << std::setw(8);

    for (int i=0; i<nelem; i++) {
        std::cout << hx[i] << "  ";
        for (int div : dividers) if ((i + 1) % div == 0) printf("\n");
    }
    std::cout << "\n";
}

template<class T>
void convert_column_row_major(T* in, T* out, size_t b, size_t n) {
    const size_t n2 = n * n;

    for (int bi = 0; bi < b; bi++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                out[bi * n2 + i * n + j] = in[bi * n2 + i + j * n];
            }
        }
    }
}

// Device array
template<class T>
struct Tensor {
    T* ptr;
    const size_t nelem_;
    Tensor() = delete;
    Tensor(const Tensor&) = delete;

    Tensor(size_t n) : nelem_(n) {
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * n));
    }

    ~Tensor() {
        cudaFree(ptr);
    }

    void copy_from_device(T* that_ptr, size_t nelem) {
        CUDA_CHECK(cudaMemcpy(ptr, that_ptr, sizeof(T) * nelem, cudaMemcpyDeviceToDevice));
    }
    void copy_from_host(T* that_ptr, size_t nelem) {
        CUDA_CHECK(cudaMemcpy(ptr, that_ptr, sizeof(T) * nelem, cudaMemcpyHostToDevice));
    }
    void copy_to_cpu(T* host_ptr, size_t nelem) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, ptr, sizeof(T) * nelem, cudaMemcpyDeviceToHost));
    }
};

template<class T>
struct TensorHost {
    T* ptr;
    const size_t nelem_;
    TensorHost() = delete;
    TensorHost(const TensorHost&) = delete;

    TensorHost(size_t n): nelem_(n) {
        ptr = (T*) malloc(sizeof(T) * n);
    }

    ~TensorHost() {
        free(ptr);
    }

    T& operator[](size_t idx) {
        return ptr[idx];
    }

    const T operator[](size_t idx) const {
        return ptr[idx];
    }

    const void print(size_t nelemt_shift, size_t nelem, const std::vector<int>& dividers = {}) const {
        print_array<T>(ptr + nelemt_shift, nelem, dividers);
    }

    template<class THAT>
    void copy_from_cpu(THAT* that_ptr, size_t nelem) {
        for (int i=0; i<nelem; i++) {
            ptr[i] = (T) that_ptr[i];
        }
    }

    void check_info(int line_number, const char* function_name) const {
        bool contains_non_zero = false;
        for (int i=0; i<nelem_; i++) {
            if (ptr[i] != 0) {
                printf("%d-th batch returned non-zero info, which is %d\n", i, ptr[i]);
                contains_non_zero = true;
            }
        }
        if (!contains_non_zero) {
            printf("info check passed, all batches returned zero info,\n"
                   "at line number %d,\n"
                   "function name %s\n",
                line_number, function_name);
        }
    }
};


void magma_inverse(const Tensor<float>& dx_orig) {
    magma_int_t b = B;
    magma_int_t n = N;

    Tensor<float> dx_copy(B*N*N);
    dx_copy.copy_from_device(dx_orig.ptr, B*N*N);

    TensorHost<float*> h_dA_array(B);
    for (int i=0; i<B; i++) { h_dA_array[i] = dx_copy.ptr + i * N * N; }
    Tensor<float*> d_dA_array(B);
    d_dA_array.copy_from_host(h_dA_array.ptr, B);

    Tensor<magma_int_t> d_ipiv(B * N);
    TensorHost<magma_int_t*> h_ipiv_array(B);
    for (int i=0; i<B; i++) { h_ipiv_array[i] = d_ipiv.ptr + i * N; }
    Tensor<magma_int_t*> d_ipiv_array(B);
    d_ipiv_array.copy_from_host(h_ipiv_array.ptr, B);

    Tensor<magma_int_t> d_info(B);

    Tensor<float> d_dinvA(B*N*N);
    TensorHost<float*> h_dinvA_array(B);
    for (int i=0; i<B; i++) { h_dinvA_array[i] = d_dinvA.ptr + i * N * N; }
    Tensor<float*> d_dinvA_array(B);
    d_dinvA_array.copy_from_host(h_dinvA_array.ptr, B);

    cudaEvent_t event_start, event_end;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublas handle create failed");
    }

    cusparseHandle_t cusparse_handle;
    if (cusparseCreate(&cusparse_handle) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparse handle create failed");
    }

#ifdef REUSE_MAGMA_QUEUE
    magma_queue_t queue;
    magma_queue_create_from_cuda(0, stream, cublas_handle, cusparse_handle, &queue);
#endif

    for (int i=0; i<NITER_WARMUP; i++) {
#ifndef REUSE_MAGMA_QUEUE
        magma_queue_t queue;
        magma_queue_create_from_cuda(0, stream, cublas_handle, cusparse_handle, &queue);
#endif
        magma_sgetrf_batched(n, n, d_dA_array.ptr, n, d_ipiv_array.ptr, d_info.ptr, b, queue);
        magma_sgetri_outofplace_batched(
            n, d_dA_array.ptr, n, d_ipiv_array.ptr, d_dinvA_array.ptr, n, d_info.ptr, b, queue);
#ifndef REUSE_MAGMA_QUEUE
        magma_queue_destroy(queue);
#endif
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(event_start, stream));
    for (int i=0; i<NITER; i++) {
#ifndef REUSE_MAGMA_QUEUE
        magma_queue_t queue;
        magma_queue_create_from_cuda(0, stream, cublas_handle, cusparse_handle, &queue);
#endif
        magma_sgetrf_batched(n, n, d_dA_array.ptr, n, d_ipiv_array.ptr, d_info.ptr, b, queue);
        magma_sgetri_outofplace_batched(
            n, d_dA_array.ptr, n, d_ipiv_array.ptr, d_dinvA_array.ptr, n, d_info.ptr, b, queue);
#ifndef REUSE_MAGMA_QUEUE
        magma_queue_destroy(queue);
#endif
    }
    CUDA_CHECK(cudaEventRecord(event_end, stream));
    CUDA_CHECK(cudaEventSynchronize(event_end));

#ifdef REUSE_MAGMA_QUEUE
    magma_queue_destroy(queue);
#endif

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_end));
    printf("magma time elapsed %f ms\n", ms / NITER);

    CUDA_CHECK(cudaStreamDestroy(stream));

    if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublas handle destroy failed");
    }

    if (cusparseDestroy(cusparse_handle) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparse handle destroy failed");
    }

    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_end));

    TensorHost<float> h_dinvA(B*N*N);
    d_dinvA.copy_to_cpu(h_dinvA.ptr, B*N*N);
    h_dinvA.print(64, 16);
}


int main()
{
    TensorHost<float> ha_f(B * N * N);

    for (int i=0; i<B; i++) {
        for (int j=0; j<N; j++) {
            // fill some random data
            for (int k=0; k<N; k++) ha_f[i * N * N + j * N + k] = (i + 1 + 2*j + 3*k);
        }
    }

    Tensor<float> da_f(B * N * N);

    da_f.copy_from_host(ha_f.ptr, B * N * N);

    if (magma_init() != MAGMA_SUCCESS) {
        throw std::runtime_error("magma init failed");
    }

    magma_inverse(da_f);

    // cusolverDnHandle_t handle;
    // CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // we may replace cusolver handle create/destroy to simply cudaMalloc/cudaFree
    // the performance regression still exists
    // so it's not cusolver handle's fault, it's probably any cudaMalloc
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, (size_t)1e6));
    CUDA_CHECK(cudaDeviceSynchronize());

    magma_inverse(da_f);

    // CUSOLVER_CHECK(cusolverDnDestroy(handle));

    CUDA_CHECK(cudaFree(ptr));

    magma_inverse(da_f);

    if (magma_finalize() != MAGMA_SUCCESS) {
        throw std::runtime_error("magma finalize failed");
    }

    return 0;
}
