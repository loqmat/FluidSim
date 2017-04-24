#pragma once

#ifndef __CUDA_UNIFORM_BUFFER_HPP__
#define __CUDA_UNIFORM_BUFFER_HPP__

#include "common.hpp"

#include "gl/glew.h"
#include "glfw/glfw3.h"

#include "cuda_setup.hpp"
#include "cuda_gl_interop.h"

#include <iostream>

namespace Fluids {

	template<typename T>
	struct UniformBuffer {
	private:
		std::size_t _byte_size;
		GLuint _gl_buffer;

		cudaGraphicsResource_t _cuda_resource;
		cudaStream_t _cuda_stream;

		T* _bound_data;

	public:

		UniformBuffer(std::size_t size) :
			_byte_size(size) {

			std::cerr << "MOOoooOOO" << std::endl;

			glGenBuffers(1, &_gl_buffer);
			glBindBuffer(GL_UNIFORM_BUFFER, _gl_buffer);
			glBufferData(GL_UNIFORM_BUFFER, _byte_size * sizeof(T), NULL, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);

			checkCUDAReturn( cudaStreamCreateWithFlags(&_cuda_stream, cudaStreamDefault) );
			checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource, _gl_buffer, cudaGraphicsRegisterFlagsNone) );
			checkCUDAReturn( cudaDeviceSynchronize() );

			std::cerr << "MOOoooOOO" << std::endl;
		}
		~UniformBuffer() {
			glDeleteBuffers(1, &_gl_buffer);
		}

		std::size_t byteSize() const {
			return _byte_size;
		}
		GLuint handleGL() const {
			return _gl_buffer;
		}

		T* bindCUDA() {
			T* mapped_data;
			std::size_t mapped_size;
			checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource, _cuda_stream ) );
			checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&mapped_data, &mapped_size, _cuda_resource ) );
			checkCUDAReturn( cudaDeviceSynchronize() );
			return mapped_data;
		}
		void unbindCUDA() {
			checkCUDAReturn( cudaGraphicsUnmapResources( 1, &_cuda_resource, _cuda_stream ) );
			checkCUDAReturn( cudaDeviceSynchronize() );
			_bound_data = NULL;
		}

		void bindGL(GLuint bindpt, std::size_t offset, std::size_t range) {
			glBindBufferRange(GL_UNIFORM_BUFFER, bindpt, _gl_buffer, offset, range);
		}
		void unbindGL(GLuint bindpt) {
			glBindBufferBase(GL_UNIFORM_BUFFER, bindpt, 0);
		}
	};

}

#endif

