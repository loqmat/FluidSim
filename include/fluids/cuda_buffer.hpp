#pragma once

#ifndef __CUDA_BUFFER_HPP__
#define __CUDA_BUFFER_HPP__

#include "common.hpp"
#include "cuda_setup.hpp"

#include <iostream>

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
			std::cerr << "MOO" << std::endl;
			checkCUDAReturn( cudaMalloc((void**)&_internal, _byte_size) );
			std::cerr << "MOO" << std::endl;
		}
		~CUDABuffer() {
			checkCUDAReturn( cudaFree((void**)&_internal) );
		}

		std::size_t byteSize() const {
			return _byte_size;
		}

		void upload(const T* data) {
			checkCUDAReturn( cudaMemcpy((void*)_internal, data, _byte_size, cudaMemcpyHostToDevice) );
		}
		void download(T* data) const {
			checkCUDAReturn( cudaMemcpy(data, (void*)_internal, _byte_size, cudaMemcpyDeviceToHost) );
		}

		CUDA_SHARED_FUNCTION operator T*() {
			return _internal;
		}
		CUDA_SHARED_FUNCTION operator const T*() const {
			return _internal;
		}

		CUDA_DEVICE_FUNCTION T& operator[] (int i) {
			return _internal[i];
		}
		CUDA_DEVICE_FUNCTION const T& operator[] (int i) const {
			return _internal[i];
		}
	};

}

#endif//__CUDA_BUFFER_HPP__