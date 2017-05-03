#pragma once

#ifndef __FLUIDS_MARCHING_CUBES_HPP__
#define __FLUIDS_MARCHING_CUBES_HPP__

#include "core/vector.hpp"
#include "grid.hpp"

#include <thrust/device_vector.h>

namespace Fluids {

	class MarchingCubes {
	public:
		struct device_data {
			int face_count;
			int volume,
				point_volume,
				length,
				width,
				depth,
				length_x_width;
			float3 cube_size;
			Uint* cube_values;
			Uint* neighbor_values;
			float3* normal_values;
			float3* vertex_buffer;
			float3* normal_buffer;
			int* index_buffer;

			device_data() { ; }
			device_data(MarchingCubes&, float3*, float3*, int*);
		};

		MarchingCubes(int L, int W, int D, const core::vec3i&);

		void uploadData(device_data& data) { _device_data.upload(&data); }
		device_data& getUploadedData() { return *((device_data*)_device_data); }

		int getFaceCount() { return _face_count; }
		void synchronizeWithDevice() {
			device_data fetch;
			checkCUDAReturn( cudaMemcpy((void*)&fetch, (device_data*)_device_data, sizeof(device_data), cudaMemcpyDeviceToHost) );
			checkCUDAReturn( cudaDeviceSynchronize() );
			_face_count = fetch.face_count;
		}

		CUDA_DEVICE_FUNCTION static void clearVoxelData(device_data& data, int index);
		CUDA_DEVICE_FUNCTION static void computeVoxelData(grid::device_data& g, device_data& data, int index);
		CUDA_DEVICE_FUNCTION static void computeSurfaceNodes(grid::device_data& g, device_data& data, int index);
		CUDA_DEVICE_FUNCTION static void computeIsoSurface(device_data& data, int index);

		void bindGL();

		int* bindIndices();
		void unbindIndices();

		float* bindVertices();
		void unbindVertices();

		float* bindNormals();
		void unbindNormals();

	private:
		CUDABuffer<device_data> _device_data;

		int _length,
			_width,
			_depth,
			_volume,
			_point_volume;

		float3 _cube_dimensions;

		thrust::device_vector<Uint> _cube_values; // each byte is 8 boolean values, fetch: array[i/8] & (1<<(i%8))
		thrust::device_vector<float3> _normal_values;
		thrust::device_vector<Uint> _neighbor_values; // 

		GLuint _vertex_buffer;
		GLuint _normal_buffer;
		GLuint _index_buffer;

		cudaGraphicsResource_t _cuda_resource_vb;
		cudaGraphicsResource_t _cuda_resource_nb;
		cudaGraphicsResource_t _cuda_resource_ib;

		int _face_count;
	};

	__global__ void clearVoxelData(MarchingCubes::device_data&);
	__global__ void computeVoxelData(grid::device_data& g, MarchingCubes::device_data&);
	__global__ void computeSurfaceNodes(grid::device_data& g, MarchingCubes::device_data& data);
	__global__ void computeIsoSurface(MarchingCubes::device_data& data);
	void runMarchingCubes(grid::device_data&, MarchingCubes&);
}

#endif//__FLUIDS_MARCHING_CUBES_HPP__