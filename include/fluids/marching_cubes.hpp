#pragma once

#ifndef __FLUIDS_MARCHING_CUBES_HPP__
#define __FLUIDS_MARCHING_CUBES_HPP__

#include "core/vector.hpp"
#include "grid.hpp"

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
			core::vec3 cube_size;
			byte* cube_values;
			byte* neighbor_values;
			int* index_buffer;

			device_data() { ; }
			device_data(MarchingCubes&, int*);
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
		CUDA_DEVICE_FUNCTION static void computeSurfaceNodes(device_data& data, int index);
		CUDA_DEVICE_FUNCTION static void computerNodeNeighbors(device_data& data, int index);

		void bindGL();

		int* bindIndices();
		void unbindIndices();

	private:
		CUDABuffer<device_data> _device_data;
		int _length,
			_width,
			_depth,
			_volume,
			_point_volume;
		core::vec3 _cube_dimensions;
		CUDABuffer<byte> _cube_values; // each byte is 8 boolean values, fetch: array[i/8] & (1<<(i%8))
		CUDABuffer<byte> _neighbor_values; // 
		GLuint _array_buffer;
		GLuint _index_buffer;
		cudaGraphicsResource_t _cuda_resource;
		int _face_count;
	};

	__global__ void clearVoxelData(MarchingCubes::device_data&);
	__global__ void computeVoxelData(grid::device_data& g, MarchingCubes::device_data&);
	__global__ void computeSurfaceNodes(MarchingCubes::device_data& data);
	__global__ void computerNodeNeighbors(MarchingCubes::device_data& data);
	void runMarchingCubes(grid::device_data&, MarchingCubes&);
}

#endif//__FLUIDS_MARCHING_CUBES_HPP__