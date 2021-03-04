template<typename scalar_t>
inline static void apply_svd_lib_gesvd(Tensor& self, Tensor& U, Tensor& S, Tensor& VT, char jobchar, Tensor& infos) {
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
  TORCH_INTERNAL_ASSERT(m >= n, "apply_svd_lib requires m >= n, but got m = ", m, ", n = ", n);

  CUDA_PARALLEL_STREAM_LAUNCH(i, batchsize, [&] {
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    at::cuda::solver::gesvd<scalar_t>(handle, jobchar, jobchar, m, n, self_data + i * self_stride,
      m, S_data + i * S_stride, U_data + i * U_stride, m, VT_data + i * VT_stride, n, infos.data_ptr<int>() + i);
  });
}

// cusolver gesvd with CUDA_PARALLEL_STREAM_LAUNCH
inline static void apply_svd_lib_gesvd_wrapper(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, char jobchar, Tensor& infos) {
  const int64_t m = self.size(-2), n = self.size(-1);
  if (m >= n) {
    Tensor self_working_copy = cloneBatchedColumnMajor(self);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda", [&] {
      apply_svd_lib_gesvd<scalar_t>(self_working_copy, U, S, VT, jobchar, infos);
    });
  } else {
    // cusolver gesvd requires the matrix dimensions m, n satisfy m >= n
    Tensor self_working_copy = cloneBatchedColumnMajor(self.transpose(-2, -1));

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda", [&] {
      apply_svd_lib_gesvd<scalar_t>(self_working_copy, VT, S, U, jobchar, infos);
    });

    VT = VT.transpose(-2, -1);
    U = U.transpose(-2, -1);
  }
}

std::tuple<Tensor, Tensor, Tensor> _svd_helper_cuda_lib(const Tensor& self, bool some, bool compute_uv) {
  at::Tensor infos = at::zeros({batchCount(self)}, self.options().dtype(at::kInt));
  const int64_t m = self.size(-2), n = self.size(-1);
  const int64_t k = std::min(m, n);

  char jobchar = compute_uv ? (some ? 'S' : 'A') : 'N';

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) = \
    _create_U_S_VT(self, some, compute_uv, /* svd_use_cusolver = */ true);
  // U, S, V working copies are already column majored now

  if (self.numel() > 0) {

    apply_svd_lib_gesvd_wrapper(self, U_working_copy, S_working_copy, VT_working_copy, jobchar, infos);

    batchCheckErrors(infos, "svd_cuda");

    if (compute_uv) {
      if (some) {
        VT_working_copy = VT_working_copy.narrow(-1, 0, k);
      }
    } else {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  }

  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}