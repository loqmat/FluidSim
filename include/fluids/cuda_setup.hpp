#pragma once

#ifndef __CUDA_ERROR_HANDLING_HPP__
#define __CUDA_ERROR_HANDLING_HPP__

#include "common.hpp"

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
	void checkGLResult();

	CUDA_DEVICE_FUNCTION inline void getXYZFromIndex(int i, int dimx, int dimy, int* x, int* y, int* z) {
		int dimxy = dimx * dimy;

		*x = 1;
		*x = (i % dimxy) % dimx;
		*y = (i % dimxy) / dimx;
		*z = (i / dimxy);
	}
	CUDA_DEVICE_FUNCTION inline void getIndexFromXYZ(int x, int y, int z, int dimx, int dimy, int* i) {
		*i = x  + y * dimx + z * dimx * dimy;
	}

	CUDA_DEVICE_FUNCTION inline float cuDot4 (const float4& vA, const float4& vB) {
		return (vA.x * vB.x + vA.y * vB.y + vA.z * vB.z);
	}
	CUDA_DEVICE_FUNCTION inline float4 cuCross4 (const float4& a, const float4& b) {
		return make_float4( a.y*b.z - a.z*b.y,
							a.z*b.x - a.x*b.z,
							a.x*b.y - a.y*b.x, 0.0f );
	}

	CUDA_DEVICE_FUNCTION inline float4 operator* (const float4& v, float s) {
		return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
	}
	CUDA_DEVICE_FUNCTION inline float4 operator* (float s, const float4& v) {
		return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
	}

	CUDA_DEVICE_FUNCTION inline float4 operator/ (const float4& v, float s) {
		return make_float4(v.x / s, v.y / s, v.z / s, v.w / s);
	}
	CUDA_DEVICE_FUNCTION inline float4 operator/ (float s, const float4& v) {
		return make_float4(v.x / s, v.y / s, v.z / s, v.w / s);
	}

	CUDA_DEVICE_FUNCTION inline float4 operator+ (const float4& vA, const float4& vB) {
		return make_float4(vA.x + vB.x, vA.y + vB.y, vA.z + vB.z, vA.w + vB.w);
	}
	CUDA_DEVICE_FUNCTION inline float4 operator- (const float4& vA, const float4& vB) {
		return make_float4(vA.x - vB.x, vA.y - vB.y, vA.z - vB.z, vA.w - vB.w);
	}


	CUDA_DEVICE_FUNCTION inline float3 operator* (const float3& v, float s) {
		return make_float3(v.x * s, v.y * s, v.z * s);
	}
	CUDA_DEVICE_FUNCTION inline float3 operator* (float s, const float3& v) {
		return make_float3(v.x * s, v.y * s, v.z * s);
	}

	CUDA_DEVICE_FUNCTION inline float3 operator/ (const float3& v, float s) {
		return make_float3(v.x / s, v.y / s, v.z / s);
	}
	CUDA_DEVICE_FUNCTION inline float3 operator/ (float s, const float3& v) {
		return make_float3(v.x / s, v.y / s, v.z / s);
	}

	CUDA_DEVICE_FUNCTION inline float3 operator+ (const float3& vA, const float3& vB) {
		return make_float3(vA.x + vB.x, vA.y + vB.y, vA.z + vB.z);
	}
	CUDA_DEVICE_FUNCTION inline float3 operator- (const float3& vA, const float3& vB) {
		return make_float3(vA.x - vB.x, vA.y - vB.y, vA.z - vB.z);
	}

	CUDA_DEVICE_FUNCTION inline float3 operator+ (const float3& vA, const float4& vB) {
		return make_float3(vA.x + vB.x, vA.y + vB.y, vA.z + vB.z);
	}
	CUDA_DEVICE_FUNCTION inline float3 operator- (const float3& vA, const float4& vB) {
		return make_float3(vA.x - vB.x, vA.y - vB.y, vA.z - vB.z);
	}
	CUDA_DEVICE_FUNCTION inline float3 operator+ (const float4& vA, const float3& vB) {
		return make_float3(vA.x + vB.x, vA.y + vB.y, vA.z + vB.z);
	}
	CUDA_DEVICE_FUNCTION inline float3 operator- (const float4& vA, const float3& vB) {
		return make_float3(vA.x - vB.x, vA.y - vB.y, vA.z - vB.z);
	}

}

#endif//__CUDA_ERROR_HANDLING_HPP__