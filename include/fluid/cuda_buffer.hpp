#pragma once

#ifndef __CUDA_BUFFER_HPP__
#define __CUDA_BUFFER_HPP__

#include "common.hpp"
#include "cuda_setup.hpp"

namespace Fluids {
	struct CUDABuffer {
	private:
		std::size_t _byte_size;
		char* _internal;
	public:
		CUDABuffer(std::size_t size);
		~CUDABuffer();

		std::size_t byteSize() const;

		void CUDABuffer::upload(const void* data);
		void CUDABuffer::download(void* data) const;

		template<typename T>
		operator T*() {
			return (T*) _internal;
		}
	};
}

#endif//__CUDA_BUFFER_HPP__