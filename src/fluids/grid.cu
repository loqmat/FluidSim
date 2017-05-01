#include "fluids/grid.hpp"
#include "fluids/marching_cubes.hpp"

#include "curand.h"
#include "curand_kernel.h"
#include <iostream>

namespace Fluids {

	// once per particle
	__global__ void configureInitialSettings(grid::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		int x = 0, y = 0, z = 0;
		getXYZFromIndex(i, data.dimensions.x, data.dimensions.y, &x, &y, &z);

		printf("%d %d %d\r\n", x, y, z);

		curandState_t state;
		curand_init(0, 0, 0, &state);

		if ( i < data.particle_count ) {
			data.positions[i] = core::vec4(x + curand_normal(&state),y + curand_normal(&state),z + curand_normal(&state), 1.0);
			data.particles[i] = particle();
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

	void runCUDASimulation(grid& sim, MarchingCubes& mc, double dt, unsigned int frame) {
		//std::cerr << "------------------------------------------------------------" << std::endl;

		grid::device_data data(sim, sim.bindPositions());
		sim.uploadData(data);

		int block_count = data.particle_count/THREAD_COUNT + ((data.particle_count%THREAD_COUNT > 0)?1:0);
		int thread_count = std::min(data.particle_count, THREAD_COUNT);

		std::cerr << "--- FRAME " << frame << " ---------------------------------------------" << std::endl;

		//std::cerr << "BLOCK COUNT " << block_count << std::endl;
		//std::cerr << "THREAD COUNT " << thread_count << std::endl;
		//std::cerr << "PARTICLE COUNT " << data.particle_count << std::endl;

		calculatePressure<<<block_count,thread_count>>>( sim.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		//std::cerr << "FINISHED CALCULATING PRESSURE" << std::endl;

		calculateForces<<<block_count,thread_count>>>( sim.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		//std::cerr << "FINISHED CALCULATING FORCES" << std::endl;

		integrate<<<block_count,thread_count>>>( sim.getUploadedData(), dt );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		runMarchingCubes(sim.getUploadedData(), mc);

		//std::cerr << "FINISHED INTEGRATING POSITIONS" << std::endl;

		sim.unbindPositions();
	}

	CUDA_SHARED_FUNCTION bool grid::cell::addParticle ( int p ) {
		for (int i = 0; i < 8; i++) {
			if ( _my_particles[i] == -1 ) {
				_my_particles[i] = p;
				return true;
			}
		}
		return false;
	}

	CUDA_SHARED_FUNCTION bool grid::cell::removeParticle ( int p ) {
		for (int i = 0; i < 8; i++) {
			if ( _my_particles[i] == p ) {
				_my_particles[i] = -1;
				return true;
			}
		}
		return false;
	}


	grid::grid(int length, int width, int depth, float filled) :
		_device_data(1),
		_dimensions(length, width, depth),
		_cells(length*width*depth), 
		_particles(length*width*depth),
		_positions(length*width*depth) {

		int volume = length*width*depth;

		_particle_count = filled * volume;

		glBindBuffer(GL_COPY_WRITE_BUFFER, _positions.handleGL());
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(core::vec4) * volume, NULL);
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
		particle& p = data.particles[i];
		core::vec4& pos = data.positions[i];
		core::vec3 vel = p.getVelocity();

		//printf("Wall collision initial: %f %f %f\r\n", vel.x, vel.y, vel.z);

		if (pos.x < 0) {
			pos.x = -pos.x;
			vel.x = -vel.x * 0.9f;
		} else if (pos.x > data.dimensions.x) {
			pos.x = 2*data.dimensions.x - pos.x;
			vel.x = -vel.x * 0.9f;
		}

		if (pos.y < 0) {
			pos.y = -pos.y;
			vel.y = -vel.y * 0.9f;
		} else if (pos.y > data.dimensions.y) {
			pos.y = 2*data.dimensions.y - pos.y;
			vel.y = -vel.y * 0.9f;
		}

		if (pos.z < 0) {
			pos.z = -pos.z;
			vel.z = -vel.z * 0.9f;
		} else if (pos.z > data.dimensions.z) {
			pos.z = 2*data.dimensions.z - pos.z;
			vel.z = -vel.z * 0.9f;
		}

		//printf("Wall collision final: %f %f %f\r\n", vel.x, vel.y, vel.z);

		p.setVelocity(vel);
	}

	CUDA_DEVICE_FUNCTION void grid::reassignParticle( device_data& data, int i ) {

		particle& p = data.particles[i];
		core::vec4& pos = data.positions[i];
		int cindex = p.getCellIndex();
		int orig_cindex = cindex;
		const core::vec3& start = data.cells[cindex].getStart();

		if (pos.x < start.x) 			cindex--;
		else if (pos.x >= start.x+1.0f) cindex++;
		
		if (pos.y < start.y) 			cindex-=data.dimensions.x;
		else if (pos.y >= start.y+1.0f) cindex+=data.dimensions.x;
		
		if (pos.z < start.z) 			cindex-=data.dimensions.x*data.dimensions.y;
		else if (pos.z >= start.z+1.0f) cindex+=data.dimensions.x*data.dimensions.y;

		int vol = data.dimensions.x*data.dimensions.y*data.dimensions.z;
		if ( cindex < 0 || cindex > vol )
			printf("%d (%d) :: (%f %f %f) should be between (%f %f %f) and (%f %f %f) :: Got invalid cindex %d / %d\r\n", 
				i, orig_cindex, pos.x, pos.y, pos.z, start.x, start.y, start.z, start.x+1, start.y+1, start.z+1, cindex, vol);

		//printf("Reassign particle %d from cell %d to cell %d\r\n", i, p.getCellIndex(), cindex);
		if ( cindex != orig_cindex ) {
			printf("attempt to put %d from %d into %d\r\n", i, orig_cindex, cindex);
			if ( !data.cells[orig_cindex].removeParticle(i) )
				printf("cannot remove particle %d from %d\r\n", i, orig_cindex);
			if ( !data.cells[cindex].addParticle(i) )
				printf("cannot add particle %d from %d\r\n", i, cindex);
			printf("result %d = [%d %d %d %d %d %d %d %d] | %d = [%d %d %d %d %d %d %d %d]\r\n", 
				orig_cindex,
				data.cells[orig_cindex].getParticle(0),
				data.cells[orig_cindex].getParticle(1),
				data.cells[orig_cindex].getParticle(2),
				data.cells[orig_cindex].getParticle(3),
				data.cells[orig_cindex].getParticle(4),
				data.cells[orig_cindex].getParticle(5),
				data.cells[orig_cindex].getParticle(6),
				data.cells[orig_cindex].getParticle(7),
				cindex,
				data.cells[cindex].getParticle(0),
				data.cells[cindex].getParticle(1),
				data.cells[cindex].getParticle(2),
				data.cells[cindex].getParticle(3),
				data.cells[cindex].getParticle(4),
				data.cells[cindex].getParticle(5),
				data.cells[cindex].getParticle(6),
				data.cells[cindex].getParticle(7));
			p.assignCell(cindex);
		}
	}



	CUDA_DEVICE_FUNCTION double grid::WPoly6 ( double r2 ) {
		return 315.0/(64.0*MATH_PI*pow(CONST_H,9)) * pow(CONST_H*CONST_H-r2,3);
	}

	CUDA_DEVICE_FUNCTION core::vec3 grid::gradientWPoly6 ( core::vec3& d, double r2 ) {
		return -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * pow((CONST_H*CONST_H-r2),2) * d;
	}

	CUDA_DEVICE_FUNCTION double grid::laplacianWPoly6 ( double r2 ) {
	  	return -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * (CONST_H*CONST_H-r2) * (3.0*CONST_H*CONST_H - 7.0*r2);
	}

	CUDA_DEVICE_FUNCTION core::vec3 grid::gradientWSpiky ( core::vec3& d, double r ) {
		return -45.0/(MATH_PI*pow(CONST_H,6)) * pow(CONST_H-r, 2) * d/r;
	}

	CUDA_DEVICE_FUNCTION double grid::laplacianWViscosity ( double r ) {
	  	return 45.0/(MATH_PI*pow(CONST_H,6)) * (CONST_H - r);
	}

	CUDA_DEVICE_FUNCTION void grid::integrate( device_data& data, int i, double dt ) {

		particle& p = data.particles[i];
		
		core::vec4 oldPosition = data.positions[i];
		core::vec3 acceleration = p.getForce() / p.getDensity();
		core::vec3 newPosition = oldPosition.xyz() + p.getVelocity() * dt + 0.5 * acceleration * dt * dt;

		data.positions[i] = core::vec4(newPosition.x, newPosition.y, newPosition.z, 1.0f);
		p.setVelocity( (data.positions[i] - oldPosition).xyz() / dt );

		wallCollision(data, i);
		//reassignParticle(data, i);

	}

	CUDA_DEVICE_FUNCTION void grid::calculatePressure ( device_data& data, int i ) {
		particle& p = data.particles[i];
		float new_density = 0.0f;

		for (int j=0;j<data.particle_count;j++ ) {
			core::vec3 d = (data.positions[i] - data.positions[j]).xyz();
			new_density += max(0.0, WPoly6(Dot(d, d))*CONST_MASS);
		}

		p.setDensity( new_density );
		p.setPressure( max(0.0, GAS_CONSTANT * (new_density-CONST_REST_DENSITY)) );
	}

	CUDA_DEVICE_FUNCTION void grid::calculateForces ( device_data& data, int i ) {
		particle& p = data.particles[i];
		
		core::vec3 fGravity = core::vec3(0.0, 0.0, -GRAVITATIONAL_ACCELERATION) * p.getDensity();
		
		core::vec3 fVortex = core::Cross(data.positions[i].xyz() - data.dimensions/2.0f, core::vec3(0,0,1)) * p.getDensity();

		core::vec3 fPressure;
		core::vec3 fViscosity;
		core::vec3 fSurface;

		core::vec3 smoothedColorFieldGradient;
		double smoothedColorFieldLaplacian = 0.0f;

		for ( int j=0;j<data.particle_count;j++ ) {
			particle& neighbor = data.particles[ j ];

			core::vec3 d = (data.positions[i] - data.positions[j]).xyz();
			double r2 = Dot(d, d);

			if ( r2 <= CONST_H*CONST_H ) {

				if ( r2 > 0.0 ) {
					core::vec3 gradient = gradientWPoly6(d,r2);
					smoothedColorFieldGradient += CONST_MASS * gradient / neighbor.getDensity();
					fPressure += (p.getPressure() + neighbor.getPressure())/pow(neighbor.getDensity(),2) * gradient;
				}

				float r = std::sqrt(r2);
	    		smoothedColorFieldLaplacian += CONST_MASS * laplacianWPoly6(r) / neighbor.getDensity();
				fViscosity += (neighbor.getVelocity() - p.getVelocity()) * laplacianWViscosity(r) / neighbor.getDensity();
				
			}
		}

		fPressure *= -CONST_MASS * p.getDensity();
		fViscosity *= CONST_VISCOSITY * CONST_MASS;

		float smoothedColorFieldGradientLength = Length(smoothedColorFieldGradient);
		if ( smoothedColorFieldGradientLength > CONST_SURFACE_TENSION_FORCE_THRESHOLD )
			fSurface = -CONST_SURFACE_TENSION * smoothedColorFieldLaplacian * (smoothedColorFieldGradient / smoothedColorFieldGradientLength);

		p.setForce(fPressure + fSurface + fViscosity + fGravity + fVortex);
	}

}