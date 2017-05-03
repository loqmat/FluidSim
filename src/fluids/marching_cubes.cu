#include "fluids/marching_cubes.hpp"
#include "fluids/grid.hpp"

namespace Fluids {

	__global__ void clearVoxelData(MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.point_volume / 32 + 1 )
			MarchingCubes::clearVoxelData(data, i);
	}
	__global__ void computeVoxelData(grid::device_data& g, MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.point_volume )
			MarchingCubes::computeVoxelData(g, data, i);
	}
	__global__ void computeSurfaceNodes(grid::device_data& g, MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.volume )
			MarchingCubes::computeSurfaceNodes(g, data, i);
	}
	__global__ void computeIsoSurface(MarchingCubes::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < 2 * data.volume + 2 )
			MarchingCubes::computeIsoSurface(data, i);
	}
	void runMarchingCubes(grid::device_data& g, MarchingCubes& mc) {
		core::vec3* v = (core::vec3*)mc.bindVertices();
		core::vec3* n = (core::vec3*)mc.bindNormals();
		int* indices = mc.bindIndices();
		checkCUDAReturn( cudaDeviceSynchronize() );

		MarchingCubes::device_data data(mc, v, n, indices);
		data.face_count = 0;
		mc.uploadData(data);

		int block_count, thread_count;

		int needed = data.point_volume / 32 + 1;
		block_count = needed/THREAD_COUNT + (((needed%THREAD_COUNT) > 0)?1:0);
		thread_count = std::min(needed, THREAD_COUNT);

		clearVoxelData<<<block_count,thread_count>>>( mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		block_count = data.point_volume/THREAD_COUNT + (((data.point_volume%THREAD_COUNT) > 0)?1:0);
		thread_count = std::min(data.point_volume, THREAD_COUNT);

		computeVoxelData<<<block_count,thread_count>>>( g, mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		block_count = data.volume/THREAD_COUNT + (((data.volume%THREAD_COUNT) > 0)?1:0);
		thread_count = std::min(data.volume, THREAD_COUNT);

		computeSurfaceNodes<<<block_count,thread_count>>>( g, mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		needed = 2 * data.volume + 2;
		block_count = needed/THREAD_COUNT + (((needed%THREAD_COUNT) > 0)?1:0);
		thread_count = std::min(needed, THREAD_COUNT);

		computeIsoSurface<<<block_count,thread_count>>>( mc.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		mc.unbindVertices();
		mc.unbindNormals();
		mc.unbindIndices();
		checkCUDAReturn( cudaDeviceSynchronize() );
		mc.synchronizeWithDevice();
	}

	MarchingCubes::device_data::device_data(MarchingCubes& mc, core::vec3* v, core::vec3* n, int* indbuf) :
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
		normal_values(mc._normal_values),
		vertex_buffer(v),
		normal_buffer(n),
		index_buffer(indbuf) { ; }

	MarchingCubes::MarchingCubes(int L, int W, int D, const core::vec3i& dim) :
		_device_data(1),
		_length(L),
		_width(W),
		_depth(D),
		_volume(L*W*D),
		_point_volume((L+1)*(W+1)*(D+1)),
		_cube_values(_point_volume/32 + 1),
		_normal_values(_point_volume),
		_neighbor_values(_volume/32 + 1) {

		_cube_dimensions = core::vec3((float)dim.x / L, 
									  (float)dim.y / W,
									  (float)dim.z / D);

		{
			glGenBuffers(1, &_vertex_buffer);
			glGenBuffers(1, &_normal_buffer);

			glBindBuffer(GL_ARRAY_BUFFER, _vertex_buffer);

			core::vec3* temp = new core::vec3[_volume];
			for ( int z=0;z<D;z++ ) for ( int y=0;y<W;y++ ) for ( int x=0;x<L;x++ )
				temp[x + y*L + z*L*W] = core::vec3( _cube_dimensions.x * x / 2.0f + _cube_dimensions.x / 4.0f,
													_cube_dimensions.y * y / 2.0f + _cube_dimensions.y / 4.0f,
													_cube_dimensions.z * z / 2.0f + _cube_dimensions.z / 4.0f );

			std::cerr << "create buffer of size " << _volume * sizeof(core::vec3) << std::endl;
			glBufferData(GL_ARRAY_BUFFER, _volume * sizeof(core::vec3), temp, GL_DYNAMIC_DRAW);

			glBindBuffer(GL_ARRAY_BUFFER, _normal_buffer);

			for ( int z=0;z<D;z++ ) for ( int y=0;y<W;y++ ) for ( int x=0;x<L;x++ )
				temp[x + y*L + z*L*W] = core::vec3( 1,0,0 );
			glBufferData(GL_ARRAY_BUFFER, _volume * sizeof(core::vec3), temp, GL_DYNAMIC_DRAW);

			delete[] temp;

			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		{
			int indices_x = (W+1) * (D+1) * 4;
			int indices_y = (L+1) * (D+1) * 4;
			int indices_z = (L+1) * (W+1) * 4;

			glGenBuffers(1, &_index_buffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index_buffer);

			std::cerr << "create buffer of size " << sizeof(int) * (indices_x + indices_y + indices_z) * 3 << std::endl;
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * (indices_x + indices_y + indices_z) * 3, NULL, GL_DYNAMIC_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}

		checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource_ib, _index_buffer, cudaGraphicsRegisterFlagsNone) );
		checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource_vb, _vertex_buffer, cudaGraphicsRegisterFlagsNone) );
		checkCUDAReturn( cudaGraphicsGLRegisterBuffer(&_cuda_resource_nb, _normal_buffer, cudaGraphicsRegisterFlagsNone) );
		checkCUDAReturn( cudaDeviceSynchronize() );

	}

	void MarchingCubes::bindGL() {
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, _vertex_buffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, _normal_buffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _index_buffer);
	}
	int* MarchingCubes::bindIndices() {
		int* mapped_data;
		std::size_t mapped_size;
		checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource_ib, 0 ) );
		checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&mapped_data, &mapped_size, _cuda_resource_ib ) );
		return mapped_data;
	}
	void MarchingCubes::unbindIndices() {
		checkCUDAReturn( cudaGraphicsUnmapResources( 1, &_cuda_resource_ib, 0 ) );
	}

	float* MarchingCubes::bindVertices() {
		float* mapped_data;
		std::size_t mapped_size;
		checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource_vb, 0 ) );
		checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&mapped_data, &mapped_size, _cuda_resource_vb ) );
		return mapped_data;
	}
	void MarchingCubes::unbindVertices() {
		checkCUDAReturn( cudaGraphicsUnmapResources( 1, &_cuda_resource_vb, 0 ) );
	}

	float* MarchingCubes::bindNormals() {
		float* mapped_data;
		std::size_t mapped_size;
		checkCUDAReturn( cudaGraphicsMapResources( 1, &_cuda_resource_nb, 0 ) );
		checkCUDAReturn( cudaGraphicsResourceGetMappedPointer( (void**)&mapped_data, &mapped_size, _cuda_resource_nb ) );
		return mapped_data;
	}
	void MarchingCubes::unbindNormals() {
		checkCUDAReturn( cudaGraphicsUnmapResources( 1, &_cuda_resource_nb, 0 ) );
	}

	CUDA_DEVICE_FUNCTION void MarchingCubes::clearVoxelData(device_data& data, int index) {
		data.cube_values[index] = 0;
		if ( index < data.volume / 32 + 1)
			data.neighbor_values[index] = 0;
		//printf("clear %d(%d)\r\n", index, index * 32);
		//printf("%d - N/A = %x\r\n", index, data.cube_values[index]);
	}

	CUDA_DEVICE_FUNCTION void MarchingCubes::computeVoxelData(grid::device_data& g, device_data& data, int index) {
		int x, y, z;
		getXYZFromIndex(index, data.length + 1, data.width + 1, &x, &y, &z);

		core::vec3 position(data.cube_size.x*x, data.cube_size.y*y, data.cube_size.z*z);
		//float density = 0.0f;
		
		int counted = 0;
		data.normal_values[index] = core::vec3(0,0,0);
		for (int i=0;i<g.particle_count;i++) {
			if ( core::Distance(position, g.positions[i].xyz()) <= CONST_H ) {
				atomicOr(&data.cube_values[index/32], (1<<(index%32)));
				data.normal_values[index] += g.color_field[i];
				counted ++ ;
				//data.index_buffer[atomicAdd(&data.face_count, 1)] = index;
			}
		}
		if ( counted != 0 )
			data.normal_values[index] /= counted;
	}

	CUDA_DEVICE_FUNCTION inline bool check_node(Uint* data, int index, int offset) {
		return data[index] & (Uint)(1<<offset);
	}
	CUDA_DEVICE_FUNCTION void MarchingCubes::computeSurfaceNodes(grid::device_data& g, device_data& data, int index) {
		int index_div = index / 32;
		int index_mod = index % 32;
		int x, y, z;

		getXYZFromIndex(index, data.length, data.width, &x, &y, &z);

		data.vertex_buffer[index] = core::vec3( data.cube_size.x * x / 2.0f + data.cube_size.x / 4.0f,
												data.cube_size.y * y / 2.0f + data.cube_size.y / 4.0f,
												data.cube_size.z * z / 2.0f + data.cube_size.z / 4.0f );

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
			count += ((data.cube_values[i[j]/32] & (1<<(i[j]%32))) > 0);
		
		atomicOr(&data.neighbor_values[index_div], ((count!=0&&count!=8)<<(index_mod)));

		__syncthreads();

		getIndexFromXYZ(max(0, x-1), y, z, data.length, data.width, i + 0);
		getIndexFromXYZ(min(x+1, data.length-1), y, z, data.length, data.width, i + 1);

		getIndexFromXYZ(x, max(0, y-1), z, data.length, data.width, i + 2);
		getIndexFromXYZ(x, min(y+1, data.width-1), z, data.length, data.width, i + 3);

		getIndexFromXYZ(x, y, max(0, z-1), data.length, data.width, i + 4);
		getIndexFromXYZ(x, y, min(z+1, data.depth-1), data.length, data.width, i + 5);

		bool left 	= check_node(data.neighbor_values, i[0]/32, i[0]%32);
		bool right 	= check_node(data.neighbor_values, i[1]/32, i[1]%32);
		bool front 	= check_node(data.neighbor_values, i[2]/32, i[2]%32);
		bool back 	= check_node(data.neighbor_values, i[3]/32, i[3]%32);
		bool up 	= check_node(data.neighbor_values, i[4]/32, i[4]%32);
		bool down	= check_node(data.neighbor_values, i[5]/32, i[5]%32);

		float total = left + right + front + back + up + down;

		if ( total != 0 ) {
			core::vec3 highest( data.cube_size.x * (x+1) / 2.0f,
								data.cube_size.y * (y+1) / 2.0f,
								data.cube_size.z * (z+1) / 2.0f );
			core::vec3 lowest ( data.cube_size.x * x / 2.0f,
								data.cube_size.y * y / 2.0f,
								data.cube_size.z * z / 2.0f );

			for ( int k=0;k<30;k++ ) {
				data.vertex_buffer[index] = ( left * data.vertex_buffer[i[0]] +
											  right * data.vertex_buffer[i[1]] +
											  front * data.vertex_buffer[i[2]] +
											  back * data.vertex_buffer[i[3]] +
											  up * data.vertex_buffer[i[4]] +
											  down * data.vertex_buffer[i[5]] ) / total;

				data.vertex_buffer[index].x = max(min(data.vertex_buffer[index].x, highest.x), lowest.x);
				data.vertex_buffer[index].y = max(min(data.vertex_buffer[index].y, highest.y), lowest.y);
				data.vertex_buffer[index].z = max(min(data.vertex_buffer[index].z, highest.z), lowest.z);
			}
		}

		/*if ( self && left && front ) {
			int face_index = atomicAdd(&data.face_count, 3);
			printf("%d = %d\r\n", index, face_index);
			data.index_buffer[face_index+0] = index;
			data.index_buffer[face_index+1] = i[0];
			data.index_buffer[face_index+2] = i[2];
		}*/

		//int count = 8 * ((data.cube_values[i[0]/32] & (1<<(i[0]%32)))>0);

		/*printf("--------------------\r\n%4d %4d %4d %4d\r\n%4d %4d %4d %4d\r\n",
			i[0], i[1], i[2], i[3],
			i[4], i[5], i[6], i[7]);
		printf("%d => %d%d%d%d%d%d%d%d\r\n",
			index,
			((data.cube_values[i[0]/32] & (1<<(i[0]%32))) > 0),
			((data.cube_values[i[1]/32] & (1<<(i[1]%32))) > 0),
			((data.cube_values[i[2]/32] & (1<<(i[2]%32))) > 0),
			((data.cube_values[i[3]/32] & (1<<(i[3]%32))) > 0),
			((data.cube_values[i[4]/32] & (1<<(i[4]%32))) > 0),
			((data.cube_values[i[5]/32] & (1<<(i[5]%32))) > 0),
			((data.cube_values[i[6]/32] & (1<<(i[6]%32))) > 0),
			((data.cube_values[i[7]/32] & (1<<(i[7]%32))) > 0) 
		);*/

		//printf("%d %d %d = %d\r\n", x,y,z, count);

		//printf("-- %d = %x (%u)\r\n", index/32, data.neighbor_values[index/32], ((data.neighbor_values[(index)/32] & ((count!=0&&count!=8)<<((index)%32)))>0) );

		if ( data.neighbor_values[(index)/32] & ((count!=0&&count!=8)<<((index)%32)) ) {
			//int new_index = atomicAdd(&data.face_count, 1);
			//data.index_buffer[new_index] = index;
			data.normal_buffer[index]= (data.normal_values[i[0]] +
										data.normal_values[i[1]] +
										data.normal_values[i[2]] +
										data.normal_values[i[3]] +
										data.normal_values[i[4]] +
										data.normal_values[i[5]] +
										data.normal_values[i[6]] +
										data.normal_values[i[7]])/8.0f;
		}
	}
	CUDA_DEVICE_FUNCTION void MarchingCubes::computeIsoSurface(device_data& data, int index) {
		int real_index = index / 2;

		int index_div = real_index / 32;
		int index_mod = real_index % 32;

		int x, y, z;
		getXYZFromIndex(real_index, data.length, data.width, &x, &y, &z);

		//printf("%d = %d %d %d\r\n", index, x, y, z);

		bool self = check_node(data.neighbor_values, index_div, index_mod);

		int i[6];

		int x2 = ((x+1) % data.length) ? x+1 : -1,
			y2 = ((y+1) % data.width)  ? y+1 : -1,
			z2 = ((z+1) % data.depth)  ? z+1 : -1;

		getIndexFromXYZ(x2, y,  z , data.length, data.width, i + 0);
		getIndexFromXYZ(x2, y2, z , data.length, data.width, i + 1);

		getIndexFromXYZ(x , y2, z , data.length, data.width, i + 2);
		getIndexFromXYZ(x , y2, z2, data.length, data.width, i + 3);

		getIndexFromXYZ(x , y , z2, data.length, data.width, i + 4);
		getIndexFromXYZ(x2, y , z2, data.length, data.width, i + 5);

		bool left  = x2 != -1 && check_node(data.neighbor_values, i[0]/32, i[0]%32);
		bool right = x2 != -1 && check_node(data.neighbor_values, i[1]/32, i[1]%32);

		bool front = y2 != -1 && check_node(data.neighbor_values, i[2]/32, i[2]%32);
		bool back  = y2 != -1 && check_node(data.neighbor_values, i[3]/32, i[3]%32);

		bool up    = z2 != -1 && check_node(data.neighbor_values, i[4]/32, i[4]%32);
		bool down  = z2 != -1 && check_node(data.neighbor_values, i[5]/32, i[5]%32);

		if ( !(index % 2) ) {
			if ( self && left && front ) {
				int face_index = atomicAdd(&data.face_count, 3);
					
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = real_index;
				data.index_buffer[face_index+1] = i[0];
				data.index_buffer[face_index+2] = i[2];
			}
			if ( self && left && up ) {
				int face_index = atomicAdd(&data.face_count, 3);
					
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = real_index;
				data.index_buffer[face_index+1] = i[0];
				data.index_buffer[face_index+2] = i[4];
			}
			if ( self && front && up ) {
				int face_index = atomicAdd(&data.face_count, 3);
					
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = real_index;
				data.index_buffer[face_index+1] = i[2];
				data.index_buffer[face_index+2] = i[4];
			}
		} else {
			if ( right && left && front ) {
				int face_index = atomicAdd(&data.face_count, 3);
					
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = i[0];
				data.index_buffer[face_index+1] = i[1];
				data.index_buffer[face_index+2] = i[2];
			}
			if ( down && left && up ) {
				int face_index = atomicAdd(&data.face_count, 3);
					
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = i[0];
				data.index_buffer[face_index+1] = i[4];
				data.index_buffer[face_index+2] = i[5];
			}
			if ( back && front && up ) {
				int face_index = atomicAdd(&data.face_count, 3);
					
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = i[2];
				data.index_buffer[face_index+1] = i[3];
				data.index_buffer[face_index+2] = i[4];
			}
		}


		//printf("++ %d = %x (%u)\r\n", index/32, data.neighbor_values[index/32], self);

		/*if ( self ) {
			int x, y, z;
			getXYZFromIndex(index, data.length, data.width, &x, &y, &z);

			int i[6];

			int x2 = ((x+1) % data.length) ? x+1 : -1,
				y2 = ((y+1) % data.width)  ? y+1 : -1,
				z2 = ((z+1) % data.depth)  ? z+1 : -1;

			getIndexFromXYZ(x2, y,  z , data.length, data.width, i + 0);
			getIndexFromXYZ(x2, y2, z , data.length, data.width, i + 1);

			getIndexFromXYZ(x , y2, z , data.length, data.width, i + 2);
			getIndexFromXYZ(x , y2, z2, data.length, data.width, i + 3);

			getIndexFromXYZ(x , y , z2, data.length, data.width, i + 4);
			getIndexFromXYZ(x2, y , z2, data.length, data.width, i + 5);

			bool left  = x2 != -1 && check_node(data.neighbor_values, i[0]/32, i[0]%32);
			bool right = x2 != -1 && check_node(data.neighbor_values, i[1]/32, i[1]%32);

			bool front = y2 != -1 && check_node(data.neighbor_values, i[2]/32, i[2]%32);
			bool back  = y2 != -1 && check_node(data.neighbor_values, i[3]/32, i[3]%32);

			bool up    = z2 != -1 && check_node(data.neighbor_values, i[4]/32, i[4]%32);
			bool down  = z2 != -1 && check_node(data.neighbor_values, i[5]/32, i[5]%32);

			if ( left && front) {
				int face_index = atomicAdd(&data.face_count, 3);
				
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = index;
				data.index_buffer[face_index+1] = i[0];
				data.index_buffer[face_index+2] = i[2];
			}

			if ( left && up ) {
				int face_index = atomicAdd(&data.face_count, 3);
				
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = index;
				data.index_buffer[face_index+1] = i[0];
				data.index_buffer[face_index+2] = i[4];
			}

			if ( front && up ) {
				int face_index = atomicAdd(&data.face_count, 3);
				
				//printf("%d = %d\r\n", index, face_index);

				data.index_buffer[face_index+0] = index;
				data.index_buffer[face_index+1] = i[2];
				data.index_buffer[face_index+2] = i[4];
			}
		}*/
	}

}