# UMAP
This repo provides a JAX implementation of UMAP and a color space (CIECAM16-UCS)
that points can be embedded into along with spacial dimensions. It also features
an updated data structure that should be more cache friendly.

For an example of a model that gives pixelwise embeddings to recolor, see the
`unet` model in
[kentslaney/efficientnet](https://github.com/kentslaney/efficientnet).

## Project Status
I thought I had gotten this repo working, but after testing with real-world size
data, it looks like JAX, XLA, or CUDA has a memory leak. The core dump was
triggered by a manual profile request, but the test just hangs otherwise.

```
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Failed to complete all kernels launched on stream 0x61d9d17f4410: CUDA error: Could not synchronize CUDA stream: CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1136] failed to unload module 0x61d9d2bf1b90; leaking: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
[...]
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1136] failed to unload module 0x61d9d3468850; leaking: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1209] failed to free device memory at 0x71a385403e00; result: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
[...]
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1209] failed to free device memory at 0x71a385407000; result: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1262] error deallocating host memory at 0x71a385600200: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
[...]
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1262] error deallocating host memory at 0x71a385608000: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
E external/xla/xla/stream_executor/cuda/cuda_driver.cc:1136] failed to unload module 0x61d9f19322b0; leaking: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
F1011 [datetime]   26817 pjrt_stream_executor_client.cc:1602] Non-OK-status: Release( false).status()
Status: INTERNAL: CUDA error: : CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
*** Check failure stack trace: ***
    @     0x71a53c567db4  absl::lts_20230802::log_internal::LogMessage::SendToLog()
    @     0x71a53c567cb4  absl::lts_20230802::log_internal::LogMessage::Flush()
    @     0x71a53c568159  absl::lts_20230802::log_internal::LogMessageFatal::~LogMessageFatal()
    @     0x71a534b8cb7a  xla::PjRtStreamExecutorBuffer::Delete()
    @     0x71a534b8c4fd  xla::PjRtStreamExecutorBuffer::~PjRtStreamExecutorBuffer()
    @     0x71a534b8c6de  xla::PjRtStreamExecutorBuffer::~PjRtStreamExecutorBuffer()
    @     0x71a534af0a3a  PJRT_Buffer::~PJRT_Buffer()
    @     0x71a534af0944  pjrt::PJRT_Buffer_Destroy()
    @     0x71a545531a70  std::_Function_handler<>::_M_invoke()
    @     0x71a545526a82  xla::PjRtCApiBuffer::~PjRtCApiBuffer()
    @     0x71a545526aee  xla::PjRtCApiBuffer::~PjRtCApiBuffer()
    @     0x71a54b1d7839  absl::lts_20230802::inlined_vector_internal::Storage<>::DestroyContents()
    @     0x71a54b1d5a24  xla::ifrt::PjRtArray::~PjRtArray()
    @     0x71a54b1d5aee  xla::ifrt::PjRtArray::~PjRtArray()
    @     0x71a54a96fe2a  xla::PyArray_Storage::~PyArray_Storage()
    @     0x71a54a963ab3  PyArray_tp_dealloc
    @     0x71a5a4b7ecf4  (unknown)
    @     0x71a5a4be5815  (unknown)
    @     0x71a5a4b7f697  (unknown)
    @     0x71a5a4c8e29d  _PyObject_ClearManagedDict
    @     0x71a5a4c886cb  (unknown)
    @     0x71a5a4b9b024  (unknown)
    @     0x71a5a4c87941  (unknown)
    @     0x71a5a4c86d75  (unknown)
    @     0x71a5a4c70230  Py_FinalizeEx
    @     0x71a5a4ac50a4  (unknown)
    @     0x71a5a4c3c5ec  Py_BytesMain
    @     0x71a5a4834e08  (unknown)
    @     0x71a5a4834ecc  __libc_start_main
    @     0x61d9ab9f2045  _start
zsh: IOT instruction (core dumped)  python src/umap-cam/test.py --manual-profile -rd digits_avl_aknn
```
That being said, I suspect it starts as a bug in this repo.
