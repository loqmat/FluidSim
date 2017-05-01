#include "fluids/marching_cubes.hpp"
#include "fluids/grid.hpp"

namespace Fluids {

	__global__ void clearVoxelData(MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.point_volume / 8 + 1 )
			MarchingCubes::clearVoxelData(data, i);
	}
	__global__ void computeVoxelData(grid::device_data& g, MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.point_volume )
			MarchingCubes::computeVoxelData(g, data, i);
	}
	__global__ void computeSurfaceNodes(MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.volume )
			MarchingCubes::computeSurfaceNodes(data, i);
	}
	void runMarchingCubes(grid::device_data& g, MarchingCubes& mc) {
		int* indices = mc.bindIndices();

		MarchingCubes::device_data data(mc, indices);
		data.face_count = 0;
		mc.uploadData(data);

		int block_count, thread_count;

		block_count = data.point_volume/8/THREAD_COUNT + (((data.point_volume/8)%THREAD_COUNT > 0)?1:0);
		thread_count = std::min(data.point_volume/8, THREAD_COUNT);

		clearVoxelData<<<block_count,thread_count>>>( mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		block_count = data.point_volume/THREAD_COUNT + ((data.point_volume%THREAD_COUNT > 0)?1:0);
		thread_count = std::min(data.point_volume, THREAD_COUNT);

		computeVoxelData<<<block_count,thread_count>>>( g, mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		block_count = data.volume/THREAD_COUNT + ((data.volume%THREAD_COUNT > 0)?1:0);
		thread_count = std::min(data.volume, THREAD_COUNT);

		computeSurfaceNodes<<<block_count,thread_count>>>( mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		mc.unbindIndices();
		mc.synchronizeWithDevice();
	}

	MarchingCubes::device_data::device_data(MarchingCubes& mc, int* indbuf) :
		face_count(0),
		volume(mc._volume),
		point_volume(mc._point_volume),
		length(mc._length),
		width(mc._width),
		depth(mc._depth),
		length_x_width(length * width),
		cube_size(mc._cube_dimensions),
		cube_values(mc._cube_values),
		neighbor_values(mc._neighbor_values),
		index_buffer(indbuf) { ; }

	MarchingCubes::MarchingCubes(int L, int W, int D, const core::vec3i& dim) :
		_device_data(1),
		_length(L),
		_width(W),
		_depth(D),
		_volume(L*W*D),
		_point_volume((L+1)*(W+1)*(D+1)),
		_cube_values(_point_volume/8 + 1),
		_neighbor_values(_volume) {

		_cube_dimensions = core::vec3((float)dim.x / L, 
									  (float)dim.y / W,
									  (float)dim.z / D);

		{
			glGenBuffers(1, &_array_buffer);
			glBindBuffer(GL_ARRAY_BUFFER, _array_buffer);
			
			//core::vec3* temp = new core::vec3[_point_volume];
			//for ( int z=0;z<D+1;z++ ) for ( int y=0;y<W+1;y++ ) for ( int x=0;x<L+1;x++ )
				//temp[x+y*(L+1)+z*(L+1)*(W+1)] = core::vec3(_cube_dimensions.x * x / 2, _cube_dimensions.y * y / 2, _cube_dimensions.z * z / 2);

			core::vec3* temp = new core::vec3[_volume];
			for ( int z=0;z<D;z++ ) for ( int y=0;y<W;y++ ) for ( int x=0;x<L;x++ )
				temp[x + y*L + z*L*W] = core::vec3( _cube_dimensions.x * x + _cube_dimensions.x / 2.0f,
													_cube_dimensions.y * y + _cube_dimensions.y / 2.0f,
													_cube_dimensions.z * z + _cube_dimensions.z / 2.0f );

			std::cerr << "create buffer of size " << _volume * sizeof(core::vec3) << std::endl;
			glBufferData(GL_ARRAY_BUFFER, _volume * sizeof(core::vec3), temp, GL_STATIC_DRAW);

			delete[] temp;

			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		{
			int indices_x = (W+1) * (D+1) * 2;
			int indices_y = (L+1) * (D+1) * 2;
			int indices_z = (L+1) * (W+1) * 2;

			glGenBuffers(1, &_index_buffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index_buffer);

			std::cerr << "create buffer of size " << sizeof(int) * (indices_x + indices_y + indices_z) * 3 << std::endl;
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * (indices_x + indices_y + indices_z) * 3, NULL, GL_DYNAMIC_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}

		checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource, _index_buffer, cudaGraphicsRegisterFlagsNone) );
		checkCUDAReturn( cudaDeviceSynchronize() );

	}

	void MarchingCubes::bindGL() {
		glBindBuffer(GL_ARRAY_BUFFER, _array_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index_buffer);
	}
	int* MarchingCubes::bindIndices() {
		int* mapped_data;
		std::size_t mapped_size;
		checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource, 0 ) );
		checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&mapped_data, &mapped_size, _cuda_resource ) );
		checkCUDAReturn( cudaDeviceSynchronize() );
		return mapped_data;
	}
	void MarchingCubes::unbindIndices() {
		checkCUDAReturn( cudaGraphicsUnmapResources( 1, &_cuda_resource, 0 ) );
		checkCUDAReturn( cudaDeviceSynchronize() );
	}

	CUDA_DEVICE_FUNCTION void MarchingCubes::clearVoxelData(device_data& data, int index) {
		data.cube_values[index] = 0;
	}

	CUDA_DEVICE_FUNCTION void MarchingCubes::computeVoxelData(grid::device_data& g, device_data& data, int index) {
		int x, y, z;
		getXYZFromIndex(index, data.length + 1, data.width + 1, &x, &y, &z);

		core::vec3 position(data.cube_size.x*x, data.cube_size.y*y, data.cube_size.z*z);
		for (int i=0;i<g.particle_count;i++)
			if ( core::Distance(position, g.positions[i].xyz()) <= 1.0f ) {
				data.cube_values[index/8] |= (1<<(index%8));
				//data.index_buffer[atomicAdd(&data.face_count, 1)] = index;
				break;
			}
	}

	CUDA_DEVICE_FUNCTION void MarchingCubes::computeSurfaceNodes(device_data& data, int index) {
		int x, y, z;

		getXYZFromIndex(index, data.length, data.width, &x, &y, &z);

		int i[8];

		getIndexFromXYZ(x  , y,   z  , data.length + 1, data.width + 1, i + 0);
		getIndexFromXYZ(x+1, y,   z  , data.length + 1, data.width + 1, i + 1);
		getIndexFromXYZ(x  , y+1, z  , data.length + 1, data.width + 1, i + 2);
		getIndexFromXYZ(x+1, y+1, z  , data.length + 1, data.width + 1, i + 3);
		
		getIndexFromXYZ(x  , y  , z+1, data.length + 1, data.width + 1, i + 4);
		getIndexFromXYZ(x+1, y  , z+1, data.length + 1, data.width + 1, i + 5);
		getIndexFromXYZ(x  , y+1, z+1, data.length + 1, data.width + 1, i + 6);
		getIndexFromXYZ(x+1, y+1, z+1, data.length + 1, data.width + 1, i + 7);

		int count = 0;
		for ( int j=0;j<8;j++ )
			count += ((data.cube_values[i[j]/8] & (1<<(i[j]%8))) > 0);

		printf("--------------------\r\n%4d %4d %4d %4d\r\n%4d %4d %4d %4d\r\n",
			i[0], i[1], i[2], i[3],
			i[4], i[5], i[6], i[7]);
		printf("%d => %d%d%d%d%d%d%d%d\r\n",
			index,
			((data.cube_values[i[0]/8] & (1<<(i[0]%8))) > 0),
			((data.cube_values[i[1]/8] & (1<<(i[1]%8))) > 0),
			((data.cube_values[i[2]/8] & (1<<(i[2]%8))) > 0),
			((data.cube_values[i[3]/8] & (1<<(i[3]%8))) > 0),
			((data.cube_values[i[4]/8] & (1<<(i[4]%8))) > 0),
			((data.cube_values[i[5]/8] & (1<<(i[5]%8))) > 0),
			((data.cube_values[i[6]/8] & (1<<(i[6]%8))) > 0),
			((data.cube_values[i[7]/8] & (1<<(i[7]%8))) > 0) 
		);

		//printf("%d %d %d = %d\r\n", x,y,z, count);

		data.neighbor_values[index] = ( count != 0 );
		if ( data.neighbor_values[index] ) {
			int new_index = atomicAdd(&data.face_count, 1);
			//printf("%d %d %d = %d\r\n", x, y, z, new_index);
			data.index_buffer[new_index] = index;
		}
	}
	CUDA_DEVICE_FUNCTION void MarchingCubes::computerNodeNeighbors(device_data& data, int index) {

	}

}