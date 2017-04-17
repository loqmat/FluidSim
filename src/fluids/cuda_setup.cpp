#include "cuda.h"
#include "fluid/cuda_setup.hpp"
#include "fluid/setup.hpp"
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

}