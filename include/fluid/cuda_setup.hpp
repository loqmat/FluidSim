#pragma once

#ifndef __CUDA_ERROR_HANDLING_HPP__
#define __CUDA_ERROR_HANDLING_HPP__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Fluids {
	struct CUDA {
		cudaDeviceProp properties;
		int driverVersion;
		int runtimeVersion;
		unsigned int deviceCount;
		int devices[8];

		void Setup();
		void GLSetup();
	};
	void checkCUDAReturn(cudaError err);
	void checkCUDAResult();
}

#endif//__CUDA_ERROR_HANDLING_HPP__