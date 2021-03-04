template<typename scalar_t>
inline static void _apply_svd_lib_gesvdj(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, Tensor& infos, bool compute_uv) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = matrixStride(VT);
  auto batchsize = batchCount(self);

  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");   

  CUDA_PARALLEL_STREAM_LAUNCH(i, batchsize, [&] {
    gesvdjInfo_t gesvdj_params;
    TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
    // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1.0e-7));
    // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 15));

    auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    at::cuda::solver::gesvdj<scalar_t>(
      handle, jobz, /*econ=*/ 1, m, n,
      self_data + i * self_stride,
      m,
      S_data + i * S_stride,
      U_data + i * U_stride,
      m,
      VT_data + i * VT_stride,
      n,
      infos.data_ptr<int>() + i,
      gesvdj_params
    );

    TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
  });
}

inline static void apply_svd_lib_gesvdj(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, Tensor& infos, bool compute_uv) {
  const int64_t m = self.size(-2), n = self.size(-1);
  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  VT = VT.transpose(-2, -1);  // gesvdj returns V instead of V^H

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda_gesvdj", [&] {
    _apply_svd_lib_gesvdj<scalar_t>(self_working_copy, U, S, VT, infos, compute_uv);
  });

  VT = VT.conj();
}