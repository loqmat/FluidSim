#include "fluids/cuda_uniform_buffer.hpp"
#include "cuda_gl_interop.h"

namespace Fluids {

	UniformBuffer::UniformBuffer(std::size_t bytes) :
		_byte_size(bytes) {

		glGenBuffers(1, &_gl_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, _gl_buffer);
		glBufferData(GL_UNIFORM_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		checkCUDAReturn( cudaStreamCreateWithFlags(&_cuda_stream, cudaStreamDefault) );
		checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource, _gl_buffer, cudaGraphicsRegisterFlagsNone) );
		checkCUDAReturn( cudaDeviceSynchronize() );
	}
	UniformBuffer::~UniformBuffer() {
		glDeleteBuffers(1, &_gl_buffer);
	}

	std::size_t UniformBuffer::byteSize() const {
		return _byte_size;
	}
	GLuint UniformBuffer::handleGL() const {
		return _gl_buffer;
	}

	void* UniformBuffer::bindCUDA() {
		float* mapped_data;
		std::size_t mapped_size;
		checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource, _cuda_stream ) );
		checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&mapped_data, &mapped_size, _cuda_resource ) );
		checkCUDAReturn( cudaDeviceSynchronize() );
		return mapped_data;
	}
	void UniformBuffer::unbindCUDA() {
		checkCUDAReturn( cudaGraphicsUnmapResources( 1, &_cuda_resource, _cuda_stream ) );
		checkCUDAReturn( cudaDeviceSynchronize() );
	}	

	void UniformBuffer::bindGL(GLuint id, std::size_t offset, std::size_t range) {
		glBindBufferRange(GL_UNIFORM_BUFFER, id, _gl_buffer, offset, range);
	}
	void UniformBuffer::unbindGL(GLuint id) {
		glBindBufferBase(GL_UNIFORM_BUFFER, id, 0);
	}

}