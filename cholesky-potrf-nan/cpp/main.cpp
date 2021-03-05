#include <iostream>
#include <iomanip>
#include <exception>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define B 16
#define N 256
#define NELEM B *N *N

#define CUDA_CHECK(EXPR)                                            \
    do {                                                            \
        cudaError_t __expr = EXPR;                                  \
        if (__expr != cudaSuccess) {                                \
            printf("CUDA failure at line: %s\n", __LINE__);         \
            throw std::runtime_error(cudaGetErrorString(__expr));   \
        }                                                           \
    } while(0)

#define CUSOLVER_CHECK(EXPR)                                    \
    do {                                                        \
        cusolverStatus_t __expr = EXPR;                         \
        if (__expr != CUSOLVER_STATUS_SUCCESS) {                \
            printf("cusolver failure at line: %s\n", __LINE__); \
            throw std::runtime_error(std::to_string(__expr));   \
        }                                                       \
    } while(0)

// print the first 10 and last 10 elements of a batch
template<class T>
void print_array(T* hx) {
    std::cout << std::setw(8);

    for (int i=0; i<10; i++) {
        std::cout << hx[i] << "  ";
    }
    std::cout << "\n";
    for (int i=N*N-10; i<N*N; i++) {
        std::cout << hx[i] << "  ";
    }
    std::cout << "\n" << "\n";
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
    void copy_to_cpu(T* host_ptr, size_t nelem) {
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

    const void print(size_t nelemt_shift = 0) const {
        print_array<T>(ptr + nelemt_shift);
    }

    template<class THAT>
    void copy_from_cpu(THAT* that_ptr, size_t nelem) {
        for (int i=0; i<nelem; i++) {
            ptr[i] = (T) that_ptr[i];
        }
    }
};

// 1. potrf batched, float
void first__potrf_batched_float(const Tensor<float>& dx_orig, cusolverDnHandle_t handle) {
    const int n = N;
    const int b = B;
    const int n2 = N * N;

    Tensor<float> dx_copy(NELEM);
    dx_copy.copy_from_device(dx_orig.ptr, NELEM);

    TensorHost<float*> hA(B);
    for (int i=0; i<B; i++) {
        hA[i] = dx_copy.ptr + i * n2;
    }
    Tensor<float*> dA(B);
    dA.copy_from_host(hA.ptr, B);

    Tensor<int> info(B);

    CUSOLVER_CHECK(cusolverDnSpotrfBatched(
        handle, CUBLAS_FILL_MODE_LOWER, n, dA.ptr, n, info.ptr, b));
    
    TensorHost<float> hres(NELEM);
    dx_copy.copy_to_cpu(hres.ptr, NELEM);

    printf("1. potrf batched, float, 2nd batch\n");
    hres.print(2 * n2);
}


// 2. potrf batched, double
void second__potrf_batched_double(const Tensor<double>& dx_orig, cusolverDnHandle_t handle) {
    const int n = N;
    const int b = B;
    const int n2 = N * N;

    Tensor<double> dx_copy(NELEM);
    dx_copy.copy_from_device(dx_orig.ptr, NELEM);

    TensorHost<double*> hA(B);
    for (int i=0; i<B; i++) {
        hA[i] = dx_copy.ptr + i * n2;
    }
    Tensor<double*> dA(B);
    dA.copy_from_host(hA.ptr, B);

    Tensor<int> info(B);

    CUSOLVER_CHECK(cusolverDnDpotrfBatched(
        handle, CUBLAS_FILL_MODE_LOWER, n, dA.ptr, n, info.ptr, b));
    
    TensorHost<double> hres(NELEM);
    dx_copy.copy_to_cpu(hres.ptr, NELEM);

    printf("2. potrf batched, double, 2nd batch\n");
    hres.print(2 * n2);
}


// 3. potrf single, float
void third__potrf_single_float(const Tensor<float>& dx_orig, cusolverDnHandle_t handle, size_t kth_batch) {
    const int n = N;
    const int b = B;
    const int n2 = N * N;

    Tensor<float> dx_copy(n2);
    dx_copy.copy_from_device(dx_orig.ptr + kth_batch * n2, n2);

    int lwork;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, dx_copy.ptr, n, &lwork));

    Tensor<int> info(1);
    Tensor<float> d_workspace(lwork);
    CUSOLVER_CHECK(cusolverDnSpotrf(
        handle, CUBLAS_FILL_MODE_LOWER, n, dx_copy.ptr, n, d_workspace.ptr, lwork, info.ptr));
    
    TensorHost<float> hres(n2);
    dx_copy.copy_to_cpu(hres.ptr, n2);

    printf("3. potrf single, float, 2nd batch\n");
    hres.print();
}

#ifdef USE_LAPACK

extern "C" void spotrf_(char*, int*, float*, int*, int*);

// 4. lapack potrf single, float
void fourth__potrf_single_lapack(const TensorHost<float>& dx_orig, size_t kth_batch) {
    int n = N;
    int b = B;
    int n2 = N * N;
    char uplo = 'L';

    TensorHost<float> dx_copy(n2);
    dx_copy.copy_from_cpu(dx_orig.ptr + kth_batch * n2, n2);

    int info;
    spotrf_(&uplo, &n, dx_copy.ptr, &n, &info);

    printf("4. lapack potrf single, float, 2nd batch\n");
    dx_copy.print();
}

#endif


int main()
{
    TensorHost<float> hx_f(NELEM);
    TensorHost<double> hx_d(NELEM);

    FILE *fp = fopen("data.bin", "rb");
    fread(hx_f.ptr, sizeof(float), NELEM, fp);
    fclose(fp);

    hx_d.copy_from_cpu(hx_f.ptr, NELEM);

    // hx_f.print();

    Tensor<float> dx_f(NELEM);
    Tensor<double> dx_d(NELEM);

    dx_f.copy_from_host(hx_f.ptr, NELEM);
    dx_d.copy_from_host(hx_d.ptr, NELEM);

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));


    first__potrf_batched_float(dx_f, handle);
    second__potrf_batched_double(dx_d, handle);
    third__potrf_single_float(dx_f, handle, 2);

#ifdef USE_LAPACK
    fourth__potrf_single_lapack(hx_f, 2);
#endif

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return 0;
}