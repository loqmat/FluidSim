#pragma once

#ifndef __CUDA_UNIFORM_BUFFER_HPP__
#define __CUDA_UNIFORM_BUFFER_HPP__

#include "common.hpp"

#include "gl/glew.h"
#include "glfw/glfw3.h"

#include "cuda_setup.hpp"
#include "cuda_gl_interop.h"

namespace Fluids {
	struct UniformBuffer {
	private:
		std::size_t _byte_size;
		GLuint _gl_buffer;

		cudaGraphicsResource_t _cuda_resource;
		cudaStream_t _cuda_stream;
	public:
		UniformBuffer(std::size_t bytes);
		~UniformBuffer();

		std::size_t byteSize() const;
		void subData(float* data, std::size_t offset, std::size_t size);

		void* bindCUDA();
		void unbindCUDA();

		void bindGL(GLuint bindpt, std::size_t offset, std::size_t range);
		void unbindGL(GLuint bindpt);
	};
}

#endif

