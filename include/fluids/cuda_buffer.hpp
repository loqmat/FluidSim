#pragma once

#ifndef __CUDA_BUFFER_HPP__
#define __CUDA_BUFFER_HPP__

#include "common.hpp"
#include "cuda_setup.hpp"

namespace Fluids {
	template<typename T>
	struct CUDABuffer {
	private:
		std::size_t _byte_size;
		T* _internal;
		
	public:
		CUDABuffer(std::size_t size ) :
			_byte_size(size * sizeof(T)),
			_internal(NULL) {
			cudaMalloc((void**)&_internal, size);
		}
		~CUDABuffer() {
			cudaFree((void**)&_internal);
		}

		std::size_t byteSize() const {
			return _byte_size;
		}

		void upload(const T* data) {
			cudaMemcpy((void*)_internal, data, _byte_size, cudaMemcpyHostToDevice);
		}
		void download(T* data) const {
			cudaMemcpy(data, (void*)_internal, _byte_size, cudaMemcpyDeviceToHost);
		}

		operator T*() {
			return _internal;
		}
		operator const T*() const {
			return _internal;
		}

		T& operator[] (int i) {
			return _internal[i];
		}
		const T& operator[] (int i) const {
			return _internal[i];
		}
	};

}

#endif//__CUDA_BUFFER_HPP__