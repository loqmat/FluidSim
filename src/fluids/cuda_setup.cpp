#include "cuda.h"
#include "fluids/cuda_setup.hpp"
#include "fluids/setup.hpp"
#include "cuda_gl_interop.h"
#include <iostream>

namespace Fluids {

	void CUDA::Setup() {
		cudaGetDeviceProperties(&properties, 0);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);

		std::cout << "Device name: " << properties.name << std::endl;
		std::cout << "CUDA Capability version: " << properties.major << "." <<  properties.minor << std::endl;
		std::cout << "CUDA Driver / Runtime Version: " << driverVersion << "/" << runtimeVersion << ", CUDA_VERSION: " << CUDA_VERSION << std::endl;
		std::cout << "Max Global Memory: " << properties.totalGlobalMem << std::endl;
		std::cout << "Max Threads Per Block: " << properties.maxThreadsPerBlock << std::endl;

		if (!properties.unifiedAddressing)
			throw "Unified addressing not available.";

		if (!properties.canMapHostMemory)
			throw "Can't map host memory.";

		cudaSetDeviceFlags(cudaDeviceMapHost);
	}

	void CUDA::GLSetup() {
		if (cudaGLGetDevices(&deviceCount, devices, 8, cudaGLDeviceListAll) != cudaSuccess)
			throw "cannot initialize CUDA devices";

		if (cudaGLSetGLDevice(devices[0]) != cudaSuccess)
			throw "cannot set intial CUDA devices";
	}

	void checkCUDAReturn(cudaError err) {
		if ( err != cudaSuccess ) {
			throw cudaGetErrorString(err);
		}
	}
	void checkCUDAResult() {
		checkCUDAReturn(cudaGetLastError());
	}
	void checkGLResult() {
		GLuint err;
		if ( (err = glGetError()) != GL_NO_ERROR ) {
			switch( err ) {
				case GL_INVALID_ENUM:
					throw "OpenGL: Invalid Enum";
				case GL_INVALID_VALUE:
					throw "OpenGL: Invalid Value";
				case GL_INVALID_OPERATION:
					throw "OpenGL: Invalid Operation";
				case GL_INVALID_FRAMEBUFFER_OPERATION:
					throw "OpenGL: Invalid Framebuffer Operation";
				case GL_OUT_OF_MEMORY:
					throw "OpenGL: Out of Memory";
				case GL_STACK_UNDERFLOW:
					throw "OpenGL: Stack Underflow";
				case GL_STACK_OVERFLOW:
					throw "OpenGL: Stack Overflow";
				default:
					throw "OpenGL: Unknown Error";
			}
		}
	}
}