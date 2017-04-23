#pragma once

#ifndef __CUDA_UNIFORM_BUFFER_HPP__
#define __CUDA_UNIFORM_BUFFER_HPP__

#include "common.hpp"

#include "gl/glew.h"
#include "glfw/glfw3.h"

#include "cuda_setup.hpp"
#include "cuda_gl_interop.h"

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

		UniformBuffer(std::size_t bytes) :
			_byte_size(bytes) {

			glGenBuffers(1, &_gl_buffer);
			glBindBuffer(GL_UNIFORM_BUFFER, _gl_buffer);
			glBufferData(GL_UNIFORM_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_UNIFORM_BUFFER, 0);

			checkCUDAReturn( cudaStreamCreateWithFlags(&_cuda_stream, cudaStreamDefault) );
			checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource, _gl_buffer, cudaGraphicsRegisterFlagsNone) );
			checkCUDAReturn( cudaDeviceSynchronize() );
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

		operator T* () {
			return _bound_data;
		}
		operator const T* () const {
			return _bound_data;
		}

		T& operator[] (int i) {
			if ( _bound_data == NULL )
				throw "Cannot access uniform buffer, data is not bound to CUDA!";
			return _bound_data[i];
		}
		const T& operator[] (int i) const {
			if ( _bound_data == NULL )
				throw "Cannot access uniform buffer, data is not bound to CUDA!";
			return _bound_data[i];
		}

		void bindCUDA() {
			std::size_t mapped_size;
			checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource, _cuda_stream ) );
			checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&_bound_data, &mapped_size, _cuda_resource ) );
			checkCUDAReturn( cudaDeviceSynchronize() );
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

