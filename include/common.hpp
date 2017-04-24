#pragma once

#ifndef __FLUID_COMMON_HPP__
#define __FLUID_COMMON_HPP__

#include <cstdlib>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned char byte;

#define CUDA_GLOBAL_FUNCTION __global__
#define CUDA_DEVICE_FUNCTION __device__
#define CUDA_SHARED_FUNCTION __device__ __host__

#define THREAD_COUNT 512

#endif//__FLUID_COMMON_HPP__