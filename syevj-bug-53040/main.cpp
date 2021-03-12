#include <iostream>
#include <iomanip>
#include <exception>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define B 2
#define N 3

#define N2 (N * N)
#define NELEM (B * N2)

#define CUDA_CHECK(EXPR)                                            \
    do {                                                            \
        cudaError_t __expr = EXPR;                                  \
        if (__expr != cudaSuccess) {                                \
            printf("CUDA failure at line: %d\n", __LINE__);         \
            throw std::runtime_error(cudaGetErrorString(__expr));   \
        }                                                           \
    } while(0)

#define CUSOLVER_CHECK(EXPR)                                    \
    do {                                                        \
        cusolverStatus_t __expr = EXPR;                         \
        if (__expr != CUSOLVER_STATUS_SUCCESS) {                \
            printf("cusolver failure at line: %d\n", __LINE__); \
            throw std::runtime_error(std::to_string(__expr));   \
        }                                                       \
    } while(0)

// print the batch
template<class T>
void print_array(T* hx, size_t nelem = N2) {
    std::cout << std::setw(8);

    for (int i=0; i<nelem; i++) {
        std::cout << hx[i] << "  ";
        if ((i + 1) % N == 0) printf("\n");
        if ((i + 1) % N2 == 0) printf("\n");
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

    const void print(size_t nelemt_shift = 0, size_t nelem = N2) const {
        print_array<T>(ptr + nelemt_shift, nelem);
    }

    template<class THAT>
    void copy_from_cpu(THAT* that_ptr, size_t nelem) {
        for (int i=0; i<nelem; i++) {
            ptr[i] = (T) that_ptr[i];
        }
    }

    void check_info() const {
        bool contains_non_zero = false;
        for (int i=0; i<nelem_; i++) {
            if (ptr[i] != 0) {
                printf("%d-th batch returned non-zero info, which is %d\n", i, ptr[i]);
                contains_non_zero = true;
            }
        }
        if (!contains_non_zero) {
            printf("info check passed, all batches returned zero info\n");
        }
    }
};

// 1. syevj batched, float
void first__syevj_batched_float(const Tensor<float>& dx_orig, cusolverDnHandle_t handle) {
    const int n = N;
    const int b = B;
    const int n2 = N2;

    Tensor<float> dx_copy(NELEM);
    dx_copy.copy_from_device(dx_orig.ptr, NELEM);

    Tensor<float> dxev(N*B);
    int lwork;

    syevjInfo_t params;
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&params));

    CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(
        handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, nullptr,
        n, nullptr, &lwork, params, b));

    Tensor<float> work(lwork);
    Tensor<int> info(B);

    CUSOLVER_CHECK(cusolverDnSsyevjBatched(
        handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dx_copy.ptr,
        n, dxev.ptr, work.ptr, lwork, info.ptr, params, b));

    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(params));

    TensorHost<float> hres(NELEM);
    TensorHost<float> hres_ev(N*B);
    TensorHost<int> hinfo(B);
    dx_copy.copy_to_cpu(hres.ptr, NELEM);
    dxev.copy_to_cpu(hres_ev.ptr, N*B);
    info.copy_to_cpu(hinfo.ptr, B);

    printf("1. syevj batched, float\n");
    hinfo.check_info();
    printf("eigenvalues\n");
    hres_ev.print(0, N*B);
    printf("eigenvectors\n");
    hres.print(0, NELEM);
}


// 2. syevj single matrix, float
void second__syevj_single_float(const Tensor<float>& dx_orig, cusolverDnHandle_t handle) {
    const int n = N;
    const int b = B;
    const int n2 = N2;

    Tensor<float> dx_copy(NELEM);
    dx_copy.copy_from_device(dx_orig.ptr, NELEM);

    Tensor<float> dxev(N*B);
    int lwork;

    syevjInfo_t params;
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&params));

    CUSOLVER_CHECK(cusolverDnSsyevj_bufferSize(
        handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, nullptr,
        n, nullptr, &lwork, params));
    
    Tensor<float> work(lwork);
    Tensor<int> info(B);

    for (int bi = 0; bi < B; bi++) {
        CUSOLVER_CHECK(cusolverDnSsyevj(
            handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
            dx_copy.ptr + bi * N2,
            n,
            dxev.ptr + bi * N,
            work.ptr, lwork,
            info.ptr + bi,
            params
        ));
        cudaDeviceSynchronize();
    }

    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(params));

    TensorHost<float> hres(NELEM);
    TensorHost<float> hres_ev(N*B);
    TensorHost<int> hinfo(B);
    dx_copy.copy_to_cpu(hres.ptr, NELEM);
    dxev.copy_to_cpu(hres_ev.ptr, N*B);
    info.copy_to_cpu(hinfo.ptr, B);

    printf("2. syevj single matrix, float\n");
    hinfo.check_info();
    printf("eigenvalues\n");
    hres_ev.print(0, N*B);
    printf("eigenvectors\n");
    hres.print(0, NELEM);
}

extern "C" void ssyev_(
    char* jobz, char* uplo, int* n, float* a, int* lda, float* w,
    float* work, int* lwork, int* info);

// 3. lapack syev single matrix, float
void third__syev_single_lapack(const TensorHost<float>& hx_orig) {
    int n = N;
    int b = B;
    int n2 = N2;
    char jobz = 'V';
    char uplo = 'L';

    TensorHost<float> hres(NELEM);
    hres.copy_from_cpu(hx_orig.ptr, NELEM);
    TensorHost<float> hres_ev(N*B);
    int lwork = 3 * N;
    TensorHost<float> work(lwork);
    TensorHost<int> hinfo(B);

    for (int bi = 0; bi < B; bi++) {
        ssyev_(
            &jobz, &uplo, &n, 
            hres.ptr + bi * N2,
            &n, 
            hres_ev.ptr + bi * N,
            work.ptr,
            &lwork,
            hinfo.ptr + bi
        );
    }

    printf("3. lapack syev single matrix, float\n");
    hinfo.check_info();
    printf("eigenvalues\n");
    hres_ev.print(0, N*B);
    printf("eigenvectors\n");
    hres.print(0, NELEM);
}


int main()
{
    float data[] = {
        -1.0163544467154912, -0.3432738726341262, -0.29756117704720914,
        -0.4600532061539261, -1.1599746387824497, 0.7295944866840084,
        -1.8334102097679381, -0.037177034906145726, 0.16004028033268325,

        -0.8431944385213139, -0.16356860913864624, 1.3159331737746953,
        -0.14841961280594543, 1.574649849716095, 1.313211035889485,
        0.14347214752391496, -0.30118687384058973, -0.2472602710856551
    };

    TensorHost<float> hx_f_row_major(NELEM);
    TensorHost<float> hx_f(NELEM);

    hx_f_row_major.copy_from_cpu(data, NELEM);
    convert_column_row_major(hx_f_row_major.ptr, hx_f.ptr, B, N);

    printf("input matrix\n");
    hx_f.print(0, 2*N2);

    Tensor<float> dx_f(NELEM);

    dx_f.copy_from_host(hx_f.ptr, NELEM);

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));


    first__syevj_batched_float(dx_f, handle);
    second__syevj_single_float(dx_f, handle);
    third__syev_single_lapack(hx_f);

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return 0;
}
