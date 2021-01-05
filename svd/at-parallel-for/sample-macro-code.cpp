/*
  Sample code for aten/src/ATen/native/cuda/BatchLinearAlgebraLib.h
  Macro CUDA_PARALLEL_STREAM_LAUNCH
*/


// inner div 4

#define CUDA_PARALLEL_STREAM_LAUNCH(_i, _batch_size, _stmt) \
do {                                                        \
  auto _main_stream = at::cuda::getCurrentCUDAStream();     \
  at::cuda::CUDAEvent _main_event;                          \
  _main_event.record(_main_stream);                         \
  at::parallel_for(0, _batch_size, _batch_size / 4 + 1, [&](int _start, int _end){ \
    auto _stream = at::cuda::getStreamFromPool();           \
    at::cuda::CUDAStreamGuard _guard(_stream);              \
    _main_event.block(_stream);                             \
    for (int _i = _start; _i < _end; _i++) {                \
      _stmt();                                              \
    }                                                       \
    at::cuda::CUDAEvent _finished;                          \
    _finished.record(_stream);                              \
    _finished.block(_main_stream);                          \
  });                                                       \
} while(0)


// outer div 4

#define CUDA_PARALLEL_STREAM_LAUNCH(_i, _batch_size, _stmt) \
do {                                                        \
  auto _main_stream = at::cuda::getCurrentCUDAStream();     \
  at::cuda::CUDAEvent _main_event;                          \
  _main_event.record(_main_stream);                         \
  at::parallel_for(0, _batch_size, _batch_size / 4 + 1, [&](int _start, int _end){ \
    for (int _i = _start; _i < _end; _i++) {                \
      auto _stream = at::cuda::getStreamFromPool();         \
      at::cuda::CUDAStreamGuard _guard(_stream);            \
      _main_event.block(_stream);                           \
      _stmt();                                              \
      at::cuda::CUDAEvent _finished;                        \
      _finished.record(_stream);                            \
      _finished.block(_main_stream);                        \
    }                                                       \
  });                                                       \
} while(0)


// inner 1

#define CUDA_PARALLEL_STREAM_LAUNCH(_i, _batch_size, _stmt) \
do {                                                        \
  auto _main_stream = at::cuda::getCurrentCUDAStream();     \
  at::cuda::CUDAEvent _main_event;                          \
  _main_event.record(_main_stream);                         \
  at::parallel_for(0, _batch_size, 1, [&](int _start, int _end){ \
    auto _stream = at::cuda::getStreamFromPool();           \
    at::cuda::CUDAStreamGuard _guard(_stream);              \
    _main_event.block(_stream);                             \
    for (int _i = _start; _i < _end; _i++) {                \
      _stmt();                                              \
    }                                                       \
    at::cuda::CUDAEvent _finished;                          \
    _finished.record(_stream);                              \
    _finished.block(_main_stream);                          \
  });                                                       \
} while(0)


// outer 1

#define CUDA_PARALLEL_STREAM_LAUNCH(_i, _batch_size, _stmt) \
do {                                                        \
  auto _main_stream = at::cuda::getCurrentCUDAStream();     \
  at::cuda::CUDAEvent _main_event;                          \
  _main_event.record(_main_stream);                         \
  at::parallel_for(0, _batch_size, 1, [&](int _start, int _end){ \
    for (int _i = _start; _i < _end; _i++) {                \
      auto _stream = at::cuda::getStreamFromPool();         \
      at::cuda::CUDAStreamGuard _guard(_stream);            \
      _main_event.block(_stream);                           \
      _stmt();                                              \
      at::cuda::CUDAEvent _finished;                        \
      _finished.record(_stream);                            \
      _finished.block(_main_stream);                        \
    }                                                       \
  });                                                       \
} while(0)

