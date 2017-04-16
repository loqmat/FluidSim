#include "fluid/cuda_buffer.hpp"

namespace Fluids {

	CUDABuffer::CUDABuffer(std::size_t size) :
		_byte_size(size),
		_internal(NULL) {
		cudaMalloc(&_internal, size);
	}
	CUDABuffer::~CUDABuffer() {
		cudaFree(&_internal);
	}

	std::size_t CUDABuffer::byteSize() const {
		return _byte_size;
	}

	void CUDABuffer::upload(const void* data) {
		cudaMemcpy(_internal, data, _byte_size, cudaMemcpyHostToDevice);
	}

	void CUDABuffer::download(void* data) const {
		cudaMemcpy(data, _internal, _byte_size, cudaMemcpyDeviceToHost);
	}

}