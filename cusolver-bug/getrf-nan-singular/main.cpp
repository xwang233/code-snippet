#include <iostream>
#include <iomanip>
#include <exception>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define B 1
#define N 3
#define NRHS 3

#define LDA N
#define LDB N

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
    // std::cout << std::setw(8);

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

void cusolver_getrf(const Tensor<float>& da_orig, cusolverDnHandle_t handle) {
    const int b = B;
    const int n = N;

    Tensor<float> da_copy(B*N*N);
    da_copy.copy_from_device(da_orig.ptr, B*N*N);

    int lwork;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, da_copy.ptr, n, &lwork));

    Tensor<float> dwork(lwork);
    Tensor<int> dinfo(1);
    CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, da_copy.ptr, n, dwork.ptr, nullptr, dinfo.ptr));

    TensorHost<int> hinfo(1);
    dinfo.copy_to_cpu(hinfo.ptr, 1);

    hinfo.check_info(__LINE__, "cusolver_getrf");

    TensorHost<float> h_res(B*N*N);
    da_copy.copy_to_cpu(h_res.ptr, B*N*N);

    h_res.print(0, B*N*N, {N, N*N, B*N*N});
}

extern "C" void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);

void lapack_getrf(const TensorHost<float>& ha_orig){
    int b = B;
    int n = N;

    TensorHost<float> ha_copy(B*N*N);
    ha_copy.copy_from_cpu(ha_orig.ptr, B*N*N);

    TensorHost<int> ipiv(B*N);
    int hinfo;
    sgetrf_(&n, &n, ha_copy.ptr, &n, ipiv.ptr, &hinfo);

    if (hinfo != 0) {
        printf("lapack_getrf info check failed, info = %d\n", hinfo);
    }

    ha_copy.print(0, B*N*N, {N, N*N, B*N*N});
}

int main()
{
    TensorHost<float> ha_f(B * N * N);

    for (int bi=0; bi<B; bi++) {
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                ha_f[bi * N * N + i * N + j] = 1;
            }
        }
    }

    ha_f.print(0, B * N * N, {N, N*N, B*N*N});

    Tensor<float> da_f(B * N * N);

    da_f.copy_from_host(ha_f.ptr, B * N * N);

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cusolver_getrf(da_f, handle);
    lapack_getrf(ha_f);

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return 0;
}
