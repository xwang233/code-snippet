#include <iostream>
#include <iomanip>
#include <exception>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define B 2
#define N 4
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

void potrs_batched_float(
        const Tensor<float>& da_orig, const Tensor<float>& db_orig,
        cusolverDnHandle_t handle) {

    const int b = B;
    const int n = N;
    const int nrhs = NRHS;
    const int lda = LDA;
    const int ldb = LDB;

    Tensor<float> da_copy(B * N * N);
    da_copy.copy_from_device(da_orig.ptr, B * N * N);
    Tensor<float> db_copy(B * N * NRHS);
    db_copy.copy_from_device(db_orig.ptr, B * N * NRHS);

    TensorHost<float*> ha_array_ptr(B);
    for (int i=0; i<b; i++) ha_array_ptr[i] = da_copy.ptr + i * N * N;
    Tensor<float*> da_array_ptr(B);
    da_array_ptr.copy_from_host(ha_array_ptr.ptr, B);

    TensorHost<float*> hb_array_ptr(B);
    for (int i=0; i<b; i++) hb_array_ptr[i] = db_copy.ptr + i * N * NRHS;
    Tensor<float*> db_array_ptr(B);
    db_array_ptr.copy_from_host(hb_array_ptr.ptr, B);

    Tensor<int> d_info(B);

    CUSOLVER_CHECK(cusolverDnSpotrsBatched(
        handle, CUBLAS_FILL_MODE_LOWER, n, nrhs,
        da_array_ptr.ptr,
        lda,
        db_array_ptr.ptr,
        ldb,
        d_info.ptr,
        b
    ));

    TensorHost<int> h_info(B);
    d_info.copy_to_cpu(h_info.ptr, B);
    h_info.check_info(__LINE__, __PRETTY_FUNCTION__);

    TensorHost<float> hb_res(B * N * NRHS);
    db_copy.copy_to_cpu(hb_res.ptr, B * N * NRHS);
    hb_res.print(0, B * N * NRHS, {NRHS, N * NRHS});
}


void potrs_single_loop_float(
        const Tensor<float>& da_orig, const Tensor<float>& db_orig,
        cusolverDnHandle_t handle) {
    
    const int b = B;
    const int n = N;
    const int nrhs = NRHS;
    const int lda = LDA;
    const int ldb = LDB;

    Tensor<float> da_copy(B * N * N);
    da_copy.copy_from_device(da_orig.ptr, B * N * N);
    Tensor<float> db_copy(B * N * NRHS);
    db_copy.copy_from_device(db_orig.ptr, B * N * NRHS);

    Tensor<int> d_info(B);

    for (int bi=0; bi<b; bi++) {
        CUSOLVER_CHECK(cusolverDnSpotrs(
            handle, CUBLAS_FILL_MODE_LOWER, n, nrhs,
            da_copy.ptr + bi * N * N,
            lda,
            db_copy.ptr + bi * N * NRHS,
            ldb, 
            d_info.ptr + bi
        ));
    }
    TensorHost<int> h_info(B);
    d_info.copy_to_cpu(h_info.ptr, B);
    h_info.check_info(__LINE__, __PRETTY_FUNCTION__);

    TensorHost<float> hb_res(B * N * NRHS);
    db_copy.copy_to_cpu(hb_res.ptr, B * N * NRHS);
    hb_res.print(0, B * N * NRHS, {NRHS, N * NRHS});
}

extern "C" void spotrs_(char* uplo, int* n, int* nrhs, float* a, int* lda, float* b,
    int* ldb, int* info);

void spotrs_lapack(const TensorHost<float>& ha_orig, const TensorHost<float>& hb_orig) {
    int b = B;
    int n = N;
    int nrhs = NRHS;
    int lda = LDA;
    int ldb = LDB;

    char uplo = 'L';

    TensorHost<float> ha_copy(B * N * N);
    ha_copy.copy_from_cpu(ha_orig.ptr, B * N * N);
    TensorHost<float> hb_copy(B * N * NRHS);
    hb_copy.copy_from_cpu(hb_orig.ptr, B * N * NRHS);

    TensorHost<int> h_info(B);

    for (int bi=0; bi<b; bi++) {
        spotrs_(
            &uplo, &n, &nrhs,
            ha_copy.ptr + bi * N * N,
            &lda,
            hb_copy.ptr + bi * N * NRHS,
            &ldb,
            h_info.ptr + bi
        );
    }

    h_info.check_info(__LINE__, __PRETTY_FUNCTION__);

    hb_copy.print(0, B * N * NRHS, {NRHS, N * NRHS});
}

int main()
{
    TensorHost<float> ha_f(B * N * N);
    TensorHost<float> hb_f(B * N * NRHS);

    for (int i=0; i<B; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<N; k++) ha_f[i * N * N + j * N + k] = (i + 1 + 2*j + 3*k);
            for (int k=0; k<NRHS; k++) hb_f[i * N * NRHS + j * NRHS + k] = (3*i + 2*j + k + 2);
        }
    }

    Tensor<float> da_f(B * N * N);
    Tensor<float> db_f(B * N * NRHS);

    da_f.copy_from_host(ha_f.ptr, B * N * N);
    db_f.copy_from_host(hb_f.ptr, B * N * NRHS);

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    spotrs_lapack(ha_f, hb_f);
    potrs_single_loop_float(da_f, db_f, handle);
    potrs_batched_float(da_f, db_f, handle);

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return 0;
}
