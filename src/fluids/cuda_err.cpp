#include "cuda.h"
#include "fluid/cuda_err.hpp"
#include "fluid/setup.hpp"
#include "cuda_gl_interop.h"
#include <iostream>

namespace Fluids {

	void cudaSetup() {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		int driverVersion = 0,
			runtimeVersion = 0;

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);

		std::cout << "Device name: " << prop.name << std::endl;
		std::cout << "CUDA Capability version: " << prop.major << "." <<  prop.minor << std::endl;
		std::cout << "CUDA Driver / Runtime Version: " << driverVersion << "/" << runtimeVersion << ", CUDA_VERSION: " << CUDA_VERSION << std::endl;

		if (!prop.unifiedAddressing)
			throw "Unified addressing not available.";
		cudaGetDeviceProperties(&prop, 0);
		if (!prop.canMapHostMemory)
			throw "Can't map host memory.";

		cudaSetDeviceFlags(cudaDeviceMapHost);
	}

	void cudaGLSetup() {
		unsigned int deviceCount;
		int devices[8];

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