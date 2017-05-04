#include "fluids/grid.hpp"
#include "fluids/marching_cubes.hpp"

#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "curand.h"
#include "curand_kernel.h"
#include <iostream>

namespace Fluids {

	// once per particle
	__global__ void configureInitialSettings(grid::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int x = 0, y = 0, z = 0;
		getXYZFromIndex(i, data.dimensions.x, data.dimensions.y, &x, &y, &z);

		//printf("%d %d %d\r\n", x, y, z);

		curandState_t state;
		curand_init(i, 0, 0, &state);

		if ( i < data.particle_count ) {
			data.positions[i] = make_float4(x + curand_normal(&state) / 10.0f,
											y + curand_normal(&state) / 10.0f,
											z + curand_normal(&state) / 10.0f, 0.0);
			skipahead(3, &state);

			grid::reassignParticle(data, i);

			data.velocity[i] = make_float4(0.0,0.0,0.0,0.0);
			data.force[i] = make_float4(0.0,GRAVITATIONAL_ACCELERATION,0.0,0.0);
			data.density[i] = 0.0;
			data.pressure[i] = 0.0;
			data.sorted_particle_id[i] = i;
			//printf("%d = %d\r\n", i, data.cell_index[i]);
		}
		if ( i < data.dimensions.x*data.dimensions.y*data.dimensions.z ) {
			data.cell_count[i] = 1;
			data.cell_offset[i] = i;
			data.sorted_cell_id[i] = i;
		}
	}

	// once per particle
	__global__ void calculatePressure(grid::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.particle_count )
			grid::calculatePressure(data, i);
	}
	// once per particle
	__global__ void calculateForces(grid::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.particle_count )
			grid::calculateForces(data, i);
	}
	// once per particle
	__global__ void integrate(grid::device_data& data, double dt) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.particle_count )
			grid::integrate(data, i, dt);
	}

	void runCUDASimulation(bool visualize, grid& sim, MarchingCubes& mc, double dt, unsigned int frame) {
		std::cerr << "--- FRAME " << frame << " ---------------------------------------------" << std::endl;

		float millis = 0.0f;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		int numParticles = sim.getParticleCount();

		thrust::device_vector<int>& cell_offset = sim.cellOffset();
		thrust::device_vector<int>& cell_count = sim.cellCount();
		thrust::device_vector<int>& sorted_cell_id = sim.sortedCellID();
		thrust::device_vector<int>& cell_index = sim.cellID();
		thrust::device_vector<int>& sorted_particle_id = sim.sortedParticleID();

		int vol = sim.getVolume();

		cudaMemcpy( thrust::raw_pointer_cast(sorted_cell_id.data()),
					thrust::raw_pointer_cast(cell_index.data()),
					numParticles * sizeof(int),
					cudaMemcpyDeviceToDevice);

		thrust::sequence(thrust::device, sorted_particle_id.begin(), sorted_particle_id.begin() + numParticles);
		thrust::fill(thrust::device, sorted_cell_id.begin() + numParticles, sorted_cell_id.end(), vol + 1 );
		thrust::sort_by_key(thrust::device, sorted_cell_id.begin(), sorted_cell_id.begin() + numParticles, sorted_particle_id.begin());

		thrust::device_vector<int> cell_end(vol);
		thrust::counting_iterator<int> search_begin(0);

		thrust::lower_bound(thrust::device, sorted_cell_id.begin(), sorted_cell_id.begin() + vol, search_begin, search_begin + vol, cell_offset.begin());
		thrust::upper_bound(thrust::device, sorted_cell_id.begin(), sorted_cell_id.begin() + vol, search_begin, search_begin + vol, cell_end.begin());
		thrust::transform(thrust::device, cell_end.begin(), cell_end.begin() + vol, cell_offset.begin(), cell_count.begin(), thrust::minus<int>());
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millis, start, stop);
		sim.perf_sort_time = (sim.perf_sort_time * sim.perf_count + millis) / (sim.perf_count + 1);
		std::cerr << "cell sorting took " << sim.perf_sort_time << std::endl;

		// debug print
		/*int* a = new int[vol];
		int* b = new int[vol];
		int* c = new int[vol];
		int* d = new int[vol];
		int* e = new int[vol];
		int* f = new int[vol];

		cudaMemcpy (a, thrust::raw_pointer_cast(cell_index.data()), vol*sizeof(int), cudaMemcpyDeviceToHost );
		cudaMemcpy (b, thrust::raw_pointer_cast(sorted_cell_id.data()), vol*sizeof(int), cudaMemcpyDeviceToHost );
		cudaMemcpy (c, thrust::raw_pointer_cast(sorted_particle_id.data()), vol*sizeof(int), cudaMemcpyDeviceToHost );
		cudaMemcpy (d, thrust::raw_pointer_cast(cell_offset.data()), vol*sizeof(int), cudaMemcpyDeviceToHost );
		cudaMemcpy (e, thrust::raw_pointer_cast(cell_end.data()), vol*sizeof(int), cudaMemcpyDeviceToHost );
		cudaMemcpy (f, thrust::raw_pointer_cast(cell_count.data()), vol*sizeof(int), cudaMemcpyDeviceToHost );

		for ( int i=0;i<vol;i++ ) {
			std::cerr << "(" << b[i] << ", " << c[i] << ") ";
		} std::cerr << std::endl;

		for ( int i=0;i<vol;i++ ) {
			std::cerr << "(offset " << d[i] << ", end " << e[i] << ", count " << f[i]<< ")    ";
		} std::cerr << std::endl;

		delete[] a;
		delete[] b;
		delete[] c;
		delete[] d;
		delete[] e;*/

		//std::cerr << "------------------------------------------------------------" << std::endl;

		grid::device_data data(sim, sim.bindPositions());
		sim.uploadData(data);

		int block_count = data.particle_count/THREAD_COUNT + ((data.particle_count%THREAD_COUNT > 0)?1:0);
		int thread_count = std::min(data.particle_count, THREAD_COUNT);

		//std::cerr << "BLOCK COUNT " << block_count << std::endl;
		//std::cerr << "THREAD COUNT " << thread_count << std::endl;
		//std::cerr << "PARTICLE COUNT " << data.particle_count << std::endl;

		cudaEventRecord(start);
		calculatePressure<<<block_count,thread_count>>>( sim.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millis, start, stop);
		sim.perf_calc_pressure = (sim.perf_calc_pressure * sim.perf_count + millis) / (sim.perf_count + 1);
		std::cerr << "calculate pressure took " << sim.perf_calc_pressure << std::endl;

		//std::cerr << "FINISHED CALCULATING PRESSURE" << std::endl;

		cudaEventRecord(start);
		calculateForces<<<block_count,thread_count>>>( sim.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millis, start, stop);
		sim.perf_calc_forces = (sim.perf_calc_forces * sim.perf_count + millis) / (sim.perf_count + 1);
		std::cerr << "calculate forces took " << sim.perf_calc_forces << std::endl;

		//std::cerr << "FINISHED CALCULATING FORCES" << std::endl;

		cudaEventRecord(start);
		integrate<<<block_count,thread_count>>>( sim.getUploadedData(), dt );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millis, start, stop);
		sim.perf_integrate = (sim.perf_integrate * sim.perf_count + millis) / (sim.perf_count + 1);
		std::cerr << "integrate took " << sim.perf_integrate << std::endl;

		//if ( visualize ) {
		cudaEventRecord(start);
		runMarchingCubes(sim.getUploadedData(), mc);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&millis, start, stop);
		sim.perf_visualize = (sim.perf_visualize * sim.perf_count + millis) / (sim.perf_count + 1);
		std::cerr << "visualization took " << sim.perf_visualize << std::endl;
		//}


		//std::cerr << "FINISHED INTEGRATING POSITIONS" << std::endl;

		sim.unbindPositions();

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		++ sim.perf_count;
	}

	grid::grid(int length, int width, int depth, int count) :
		perf_count(0),
		perf_sort_time(0.0f),
		perf_calc_pressure(0.0f),
		perf_calc_forces(0.0f),
		perf_integrate(0.0f),
		perf_visualize(0.0f),
		_device_data(1),
		_dimensions(make_int4(length, width, depth, 0)),
		_cell_count(length*width*depth),
		_cell_offset(length*width*depth),
		_sorted_cell_id(length*width*depth),
		_cell_index(length*width*depth),
		_velocity(length*width*depth),
		_force(length*width*depth),
		_density(length*width*depth),
		_pressure(length*width*depth),
		_sorted_particle_id(length*width*depth),
		_color_field(length*width*depth),
		_positions(length*width*depth) {

		int volume = length*width*depth;

		_particle_count = count;

		glBindBuffer(GL_COPY_WRITE_BUFFER, _positions.handleGL());
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(float4) * volume, NULL);
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

		grid::device_data data(*this, bindPositions());
		uploadData(data);

		int block_count = data.particle_count/THREAD_COUNT + ((data.particle_count%THREAD_COUNT > 0)?1:0);
		int thread_count = std::min(data.particle_count, THREAD_COUNT);

		configureInitialSettings<<<block_count,thread_count>>>( getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		unbindPositions();
	}

	CUDA_DEVICE_FUNCTION void grid::wallCollision( device_data& data, int i ) {
		float4& pos = data.positions[i];
		float4& vel = data.velocity[i];

		if (pos.x < 0) {
			pos.x = -pos.x;
			vel.x = -vel.x * 0.9f;
		} else if (pos.x > data.dimensions.x) {
			//float oldx = pos.x;
			pos.x = 2*data.dimensions.x - pos.x - 0.1f;
			vel.x = -vel.x * 0.9f;
			//printf("x = %f, becomes %f\r\n", oldx, pos.x);
		}

		if (pos.y < 0) {
			pos.y = -pos.y;
			vel.y = -vel.y * 0.9f;
		} else if (pos.y > data.dimensions.y) {
			//float oldy = pos.y;
			pos.y = 2*data.dimensions.y - pos.y;
			vel.y = -vel.y * 0.9f;
			//printf("y = %f, becomes %f\r\n", oldy, pos.y);
		}

		if (pos.z < 0) {
			pos.z = -pos.z;
			vel.z = -vel.z * 0.9f;
		} else if (pos.z > data.dimensions.z) {
			pos.z = 2*data.dimensions.z - pos.z;
			vel.z = -vel.z * 0.9f;
		}
	}

	CUDA_DEVICE_FUNCTION void grid::reassignParticle( device_data& data, int i ) {
		getIndexFromXYZ(data.positions[i].x/CONST_H,
						data.positions[i].y/CONST_H,
						data.positions[i].z/CONST_H, 
						data.dimensions.x, 
						data.dimensions.y, 
						&data.cell_index[i]);
		/*printf("particle %d new cell is %d at position (%f %f %f) with velocity (%f %f %f) and force (%f %f %f), pressure at (%f) and density at (%f)\r\n",
			i, data.cell_index[i],
			data.positions[i].x, data.positions[i].y, data.positions[i].z,
			data.velocity[i].x, data.velocity[i].y, data.velocity[i].z,
			data.force[i].x, data.force[i].y, data.force[i].z,
			data.pressure[i], data.density[i]
			);*/
	}



	CUDA_DEVICE_FUNCTION float grid::WPoly6 ( float r2 ) {
		return 315.0/(64.0*MATH_PI*pow(CONST_H,9)) * pow(CONST_H*CONST_H-r2,3);
	}

	CUDA_DEVICE_FUNCTION float4 grid::gradientWPoly6 ( float4& d, float r2 ) {
		return -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * pow((CONST_H*CONST_H-r2),2) * d;
	}

	CUDA_DEVICE_FUNCTION float grid::laplacianWPoly6 ( float r2 ) {
	  	return -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * (CONST_H*CONST_H-r2) * (3.0*CONST_H*CONST_H - 7.0*r2);
	}

	CUDA_DEVICE_FUNCTION float4 grid::gradientWSpiky ( float4& d, float r ) {
		return -45.0/(MATH_PI*pow(CONST_H,6)) * pow(CONST_H-r, 2) * d/r;
	}

	CUDA_DEVICE_FUNCTION float grid::laplacianWViscosity ( float r ) {
	  	return 45.0/(MATH_PI*pow(CONST_H,6)) * (CONST_H - r);
	}

	CUDA_DEVICE_FUNCTION void grid::integrate( device_data& data, int i, double dt ) {

		float4 oldPosition = data.positions[i];
		float4 acceleration = data.force[i] / data.density[i];
		float4 newPosition = oldPosition + data.velocity[i] * dt + 0.5 * acceleration * dt * dt;

		data.positions[i] = make_float4(newPosition.x, newPosition.y, newPosition.z, 1.0f);
		data.velocity[i] = (data.positions[i] - oldPosition) / dt;

		/*printf("particle %d position (%f %f %f) with velocity (%f %f %f)\r\n",
			i, data.positions[i].x, data.positions[i].y, data.positions[i].z,
			data.velocity[i].x, data.velocity[i].y, data.velocity[i].z
			);*/

		wallCollision(data, i);
		//printf("%d start cell index: %d\n", i, data.cell_index[i]);
		reassignParticle(data, i);
		//printf("%d new cell index: %d\n", i, data.cell_index[i]);

	}

	CUDA_DEVICE_FUNCTION void grid::calculatePressure ( device_data& data, int i ) {
		float new_density = 0.0f;
		float4 local_pos = data.positions[i];

		int id = data.cell_index[i];
		int4 cell = make_int4(0,0,0,0);
		getXYZFromIndex(id, data.dimensions.x, data.dimensions.y, &cell.x, &cell.y, &cell.z);
		
		//iterate over all neighbors
		for (int x = -1; x <= 1; x++) {
	       	if (cell.x + x < 0) continue;
	        else if (cell.x + x >= data.dimensions.x) break;
	        
	       	for (int y = -1; y <= 1; y++) {
	       		if (cell.y + y < 0) continue;
	        	else if (cell.y + y >= data.dimensions.y ) break;
	          
		       	for (int z = -1; z <= 1; z++) {
		       		if (cell.z + z < 0) continue;
		        	else if (cell.z + z >= data.dimensions.z ) break;

		        	int neighbor_id = 0;
		        	getIndexFromXYZ(cell.x+x, cell.y+y, cell.z+z, data.dimensions.x, data.dimensions.y, &neighbor_id);
		        	int offset = data.cell_offset[neighbor_id];
		        	int count = data.cell_count[neighbor_id];

		        	//printf("source %d to index %d : offset = %d, count = %d\r\n", id, neighbor_id, offset, count);

		        	for (int k = 0; k < count; k++) {
		        		int j = data.sorted_particle_id[k + offset];
						float4 d = (local_pos - data.positions[j]);
						new_density += max(0.0f, WPoly6(cuDot4(d, d))*CONST_MASS);
		        	}
				}
			}
		}
		data.density[i] = new_density;
		data.pressure[i] = max(0.0f, GAS_CONSTANT*(new_density-CONST_REST_DENSITY));

		/*printf("particle %d , pressure at (%f) and density at (%f)\r\n",
			i, data.pressure[i], data.density[i]
			);*/

		/*data.density[i] = 0.0f;
		for (int j=0;j<data.particle_count;j++ ) {
			float4 d = data.positions[i] - data.positions[j];
			data.density[i] += max(0.0, WPoly6(cuDot4(d, d))*CONST_MASS);
		}
		data.pressure[i] = max(0.0f, GAS_CONSTANT*(data.density[i]-CONST_REST_DENSITY));*/
	}

	CUDA_DEVICE_FUNCTION void grid::calculateForces ( device_data& data, int i ) {
		float4 dimensions 	= make_float4(data.dimensions.x, data.dimensions.y, data.dimensions.z, data.dimensions.w);
		float4 local_pos	= data.positions[i];

		float4 fVortex 		= cuCross4(data.positions[i] - dimensions/2.0f, make_float4(0,0,1,0)) * data.density[i];
		float4 fGravity 	= make_float4(0.0, 0.0, -GRAVITATIONAL_ACCELERATION * data.density[i], 0.0);
		float4 fPressure 	= make_float4(0.0, 0.0, 0.0, 0.0);
		float4 fViscosity 	= make_float4(0.0, 0.0, 0.0, 0.0);
		float4 fSurface 	= make_float4(0.0, 0.0, 0.0, 0.0);

		float4 local_cf		= make_float4(0.0, 0.0, 0.0, 0.0);
		float4 local_vel	= data.velocity[i];
		float local_pres	= data.pressure[i];

		float smoothedColorFieldLaplacian = 0.0f;

		int id = data.cell_index[i];
		int4 cell = make_int4(0,0,0,0);;
		getXYZFromIndex(id, data.dimensions.x, data.dimensions.y, &cell.x, &cell.y, &cell.z);

		for (int x = -1; x <= 1; x++) {
	       	if (cell.x + x < 0) continue;
	        else if (cell.x + x >= data.dimensions.x) break;
	        
	       	for (int y = -1; y <= 1; y++) {
	       		if (cell.y + y < 0) continue;
	        	else if (cell.y + y >= data.dimensions.y ) break;
	          
		       	for (int z = -1; z <= 1; z++) {
		       		if (cell.z + z < 0) continue;
		        	else if (cell.z + z >= data.dimensions.z ) break;

		        	int neighbor_id = 0;
		        	getIndexFromXYZ(cell.x+x, cell.y+y, cell.z+z, data.dimensions.x, data.dimensions.y, &neighbor_id);
		        	int offset = data.cell_offset[neighbor_id];
		        	int count = data.cell_count[neighbor_id];

		        	//printf("count at %d (%d %d %d) = %d + %d\r\n", neighbor_id, cell.x+x, cell.y+y, cell.z+z, offset, count);

		        	for ( int k=0;k<count;k++ ) {
		        		int j = data.sorted_particle_id[k + offset];
			        	float4 d = local_pos - data.positions[j];
						float r2 = cuDot4(d, d);

						if ( r2 <= CONST_H*CONST_H ) {

							if ( r2 > 0.0 ) {
								float4 gradient = gradientWPoly6(d,r2);
								local_cf = local_cf + CONST_MASS * gradient / data.density[j];
								fPressure = fPressure + (local_pres + data.pressure[j])/(data.density[j] * data.density[j]) * gradient;
							}

							float r = std::sqrt(r2);
				    		smoothedColorFieldLaplacian += CONST_MASS * laplacianWPoly6(r) / data.density[j];
							fViscosity = fViscosity + (data.velocity[j] -local_vel) * laplacianWViscosity(r) / data.density[j];
							
						}
					}
		        }
		    }
		}

		/*for ( int j=0;j<data.particle_count;j++ ) {
			float4 d = data.positions[i] - data.positions[j];
			float r2 = cuDot4(d, d);

			if ( r2 <= CONST_H*CONST_H ) {

				if ( r2 > 0.0 ) {
					float4 gradient = gradientWPoly6(d,r2);
					data.color_field[i] = data.color_field[i] + CONST_MASS * gradient / data.density[j];
					fPressure = fPressure + (data.pressure[i] + data.pressure[j])/(data.density[j] * data.density[j]) * gradient;
				}

				float r = std::sqrt(r2);
	    		smoothedColorFieldLaplacian += CONST_MASS * laplacianWPoly6(r) / data.density[j];
				fViscosity = fViscosity + (data.velocity[j] - data.velocity[i]) * laplacianWViscosity(r) / data.density[j];
				
			}
		}*/

		fPressure = fPressure * -CONST_MASS * data.density[i];
		fViscosity = fViscosity * CONST_VISCOSITY * CONST_MASS;

		float cf_len = std::sqrt (cuDot4(local_cf, local_cf));
		if ( cf_len > CONST_SURFACE_TENSION_FORCE_THRESHOLD )
			fSurface = -CONST_SURFACE_TENSION * smoothedColorFieldLaplacian * (local_cf / cf_len);

		data.color_field[i] = local_cf;
		data.force[i] = fPressure + fSurface + fViscosity + fGravity + fVortex;

		/*printf("particle %d, force (%f %f %f)\r\n",
			i, data.force[i].x, data.force[i].y, data.force[i].z
			);*/
	}

}