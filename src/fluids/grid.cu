#include "fluids/grid.hpp"

#include <iostream>

namespace Fluids {

	__global__ void calculatePressure(grid::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.particle_count )
			grid::calculatePressure(data, i);
	}
	__global__ void calculateForces(grid::device_data& data) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.particle_count )
			grid::calculateForces(data, i);
	}
	__global__ void integrate(grid::device_data& data, double dt) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < data.particle_count )
			grid::integrate(data, i, dt);	
	}

	void runCUDASimulation(grid& sim, double dt) {
		std::cerr << "------------------------------------------------------------" << std::endl;

		int block_count = std::max(1, sim.getParticleCount()/THREAD_COUNT);
		int thread_count = std::min(sim.getParticleCount(), THREAD_COUNT);
		core::vec4* positions = sim.bindPositions();
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		grid::device_data data(sim, positions);
		sim.uploadData(data);

		std::cerr << "BLOCK COUNT " << block_count << std::endl;
		std::cerr << "THREAD COUNT " << thread_count << std::endl;
		std::cerr << "PARTICLE COUNT " << data.particle_count << std::endl;

		calculatePressure<<<block_count,thread_count>>>( sim.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		std::cerr << "FINISHED CALCULATING PRESSURE" << std::endl;

		calculateForces<<<block_count,thread_count>>>( sim.getUploadedData() );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		std::cerr << "FINISHED CALCULATING FORCES" << std::endl;

		integrate<<<block_count,thread_count>>>( sim.getUploadedData(), dt );
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		std::cerr << "FINISHED INTEGRATING POSITIONS" << std::endl;

		sim.unbindPositions();
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );
	}

	CUDA_SHARED_FUNCTION void grid::cell::addParticle ( int p ) {
		for (int i = 0; i < 8; i++) {
			if ( _my_particles[i] == -1 ) {
				_my_particles[i] = p;
				return;
			}
		}
	}

	CUDA_SHARED_FUNCTION void grid::cell::removeParticle ( int p) {
		for (int i = 0; i < 8; i++) {
			if ( _my_particles[i] == p ) {
				_my_particles[i] = -1;
				return;
			}
		}
	}


	grid::grid(int length, int width, int depth, float filled) :
		_device_data(1),
		_dimensions(length, width, depth),
		_cells(length*width*depth), 
		_particles(length*width*depth),
		_positions(length*width*depth) {

		int volume = length*width*depth;

		cell* cells = new cell [volume];
		particle* particles = new particle [volume];
		core::vec4* positions = new core::vec4 [volume];

		_particle_count = filled * volume;

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < depth; k++) {
					cells[i+j*length+k*length*width].setStart(i,j,k);
				}
			}
		}

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < depth; k++) {
					int index = i + j*length + k*length*width;
					if ( index >= _particle_count )
						continue;

					std::cerr << "set particle " << index << std::endl;

					positions[index] = core::vec4(i,j,k,1.0);
					particles[index].assignCell(index);
					cells[index].addParticle(index);
				}
			}
		}

		_cells.upload(cells);
		_particles.upload(particles);

		glBindBuffer(GL_COPY_WRITE_BUFFER, _positions.handleGL());
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(core::vec4) * volume, (void*)positions);
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
		
		delete[] cells;
		delete[] particles;
		delete[] positions;

		checkCUDAReturn( cudaDeviceSynchronize() );
	}

	CUDA_DEVICE_FUNCTION void grid::wallCollision( device_data& data, int i ) {
		particle& p = data.particles[i];
		core::vec4& pos = data.positions[i];
		core::vec3 vel = p.getVelocity();

		printf("Wall collision initial: %f %f %f\r\n", vel.x, vel.y, vel.z);

		if (pos.x < 0) {
			pos.x = 0;
			vel.x = -vel.x;
		} else if (pos.x > data.dimensions.x) {
			pos.x = data.dimensions.x;
			vel.x = -vel.x;
		}

		if (pos.y < 0) {
			pos.y = 0;
			vel.y = -vel.y;
		} else if (pos.y > data.dimensions.y) {
			pos.y = data.dimensions.y;
			vel.y = -vel.y;
		}

		if (pos.z < 0) {
			pos.z = 0;
			vel.z = -vel.z;
		} else if (pos.z > data.dimensions.z) {
			pos.z = data.dimensions.z;
			vel.z = -vel.z;
		}

		printf("Wall collision final: %f %f %f\r\n", vel.x, vel.y, vel.z);

		p.setVelocity(vel);
	}

	CUDA_DEVICE_FUNCTION void grid::reassignParticle( device_data& data, int i ) {

		particle& p = data.particles[i];
		core::vec4& pos = data.positions[i];
		int cindex = p.getCellIndex();
		const core::vec3& start = data.cells[cindex].getStart();
		data.cells[cindex].removeParticle(i);

		if (pos.x < start.x) 			cindex--;
		else if (pos.x >= start.x+1) 	cindex++;
		if (pos.y < start.y) 			cindex-=data.dimensions.x;
		else if (pos.y >= start.y+1) 	cindex+=data.dimensions.x;
		if (pos.z < start.z) 			cindex-=data.dimensions.x*data.dimensions.y;
		else if (pos.z >= start.z+1) 	cindex+=data.dimensions.x*data.dimensions.y;

		data.cells[cindex].addParticle(i);
		p.assignCell(cindex);

	}



	CUDA_DEVICE_FUNCTION float grid::WPoly6 ( float r ) {

		return 315.0/(64.0*MATH_PI*pow(CONST_H,9)) * pow(CONST_H*CONST_H-r*r,3);

	}

	CUDA_DEVICE_FUNCTION core::vec3 grid::gradientWPoly6 ( core::vec3& r, float d ) {

		return -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * pow((CONST_H*CONST_H-d*d),2) * r;

	}

	CUDA_DEVICE_FUNCTION float grid::laplacianWPoly6 ( float r ) {
	    
	  	return -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * (CONST_H*CONST_H-r*r) * (3.0*CONST_H*CONST_H - 7.0*r*r);

	}

	CUDA_DEVICE_FUNCTION core::vec3 grid::gradientWSpiky ( core::vec3& r, float d ) {
	 
		//Wspiky = ( 15.0/(MATH_PI*pow(CONST_H,6))) * pow((CONST_H-r),3 );
		return -45.0/(MATH_PI*pow(CONST_H,6)) * pow(CONST_H-d, 2) * r/d;

	}

	CUDA_DEVICE_FUNCTION float grid::laplacianWViscosity ( float r ) {

		//return ( (15.0/(2.0*MATH_PI*pow(CONST_H,3))) * (-pow(r,3)/(2*pow(CONST_H,3)) + ((r*r)/(CONST_H*CONST_H)) + CONST_H/(2.0*r) - 1) ); 
	  	return 45.0/(MATH_PI*pow(CONST_H,6)) * (CONST_H - r);   

	}

	CUDA_DEVICE_FUNCTION void grid::integrate( device_data& data, int i, float dt ) {

		particle& p = data.particles[i];
		
		core::vec4 oldPosition = data.positions[i];
		core::vec3 acceleration = p.getForce()/p.getDensity();
		//core::vec3 acceleration = p.getForce();
		core::vec3 temp = oldPosition.xyz() + p.getVelocity() * dt + 0.5 * acceleration * dt * dt;

		printf("(%f %f %f) => (%f %f %f), velocity(%f %f %f), acceleration(%f %f %f), density(%f)\r\n", 
			oldPosition.x, oldPosition.y, oldPosition.z,
			temp.x, temp.y, temp.z,
			p.getVelocity().x, p.getVelocity().y, p.getVelocity().z,
			acceleration.x, acceleration.y, acceleration.z,
			p.getDensity());

		data.positions[i] = core::vec4(temp.x, temp.y, temp.z, 1.0f);
		p.setVelocity( (data.positions[i] - oldPosition).xyz() / dt );

		wallCollision(data, i);
		reassignParticle(data, i);

	}

	CUDA_DEVICE_FUNCTION void grid::calculatePressure ( device_data& data, int i ) {
		particle& p = data.particles[i];
		p.setDensity(0.0);
		//printf("test particle %d in cell %d\r\n", i, p.getCellIndex());
		//printf("location is %f %f %f\r\n", data.positions[i].x, data.positions[i].y, data.positions[i].z);
		
		//iterate over all neighbors
		for (int x = -1; x < 2; x++) {
	       	if (data.positions[i].x + x < 0) continue;
	        else if (data.positions[i].x + x >= data.dimensions.x) continue;
	        
	       	for (int y = -1; y < 2; y++) {
	       		if (data.positions[i].y + y < 0) continue;
	        	else if (data.positions[i].y + y >= data.dimensions.y ) continue;
	          
		       	for (int z = -1; z < 2; z++) {
		       		if (data.positions[i].z + z < 0) continue;
		        	else if (data.positions[i].z + z >= data.dimensions.z ) continue;

		        	int cell = p.getCellIndex() + x + y*data.dimensions.x + z*data.dimensions.x*data.dimensions.y;
		        	for (int j = 0; j < 8; j++) {

						int neighbor_index = data.cells[cell].getParticle(j);		        		
		        		if (neighbor_index == -1 || neighbor_index == i ) continue;
		        		
						float d = Distance(data.positions[i], data.positions[neighbor_index]);
						//printf("for cell %d, distance %d => %d = %f\r\n", cell, i, neighbor_index, d);
						if ( d*d <= CONST_H*CONST_H ) {
							p.setDensity( p.getDensity() + WPoly6(d)*CONST_MASS );
						}
					}
				}
			}
		}	//end neighbor calcs
		p.setPressure( GAS_CONSTANT*(p.getDensity()-CONST_REST_DENSITY) );
	}

	CUDA_DEVICE_FUNCTION void grid::calculateForces ( device_data& data, int i ) {
		particle& p = data.particles[i];
		core::vec3 fGravity (0.0, GRAVITATIONAL_ACCELERATION, 0.0);
		core::vec3 fPressure;
		core::vec3 fViscosity;
		core::vec3 fSurface;
		core::vec3 smoothedColorFieldGradient;
		float smoothedColorFieldLaplacian = 0.0f;
		
		//iterate over all neighbors

		for (int x = -1; x < 2; x++) {
	       	if (data.positions[i].x + x < 0) continue;
	        else if ( data.positions[i].x + x >= data.dimensions.x) break;
	        
	       	for (int y = -1; y < 2; y++) {
	       		if (data.positions[i].y + y < 0) continue;
	        	else if ( data.positions[i].y + y >= data.dimensions.y ) break;
	          
		       	for (int z = -1; z < 2; z++) {
		       		if (data.positions[i].z + z < 0) continue;
		        	else if ( data.positions[i].z + z >= data.dimensions.z ) break;

		        	int cell = p.getCellIndex() + x+y*data.dimensions.x+z*data.dimensions.x*data.dimensions.y;

		        	for (int j = 0; j < 8; j++) {

		        		if (data.cells[cell].getParticle(j) == -1 || data.cells[cell].getParticle(j) == i ) continue;
		        		
		        		particle& neighbor = data.particles[ data.cells[cell].getParticle(j) ];
						float d = Distance(data.positions[i], data.positions[data.cells[cell].getParticle(j)]);
						core::vec3 r = (data.positions[i] - data.positions[data.cells[cell].getParticle(j)]).xyz();

						if ( Dot(r, r) <= CONST_H*CONST_H ) {

							fPressure += (p.getPressure()/pow(p.getDensity(),2) + neighbor.getPressure()/pow(neighbor.getDensity(),2)) * gradientWSpiky(r,d);
							fViscosity += (neighbor.getVelocity() - p.getVelocity()) * laplacianWViscosity(d) / neighbor.getDensity();
			        		smoothedColorFieldLaplacian += CONST_MASS * laplacianWPoly6(d) / neighbor.getDensity();
							smoothedColorFieldGradient += CONST_MASS * gradientWPoly6(r,d) / neighbor.getDensity();
							
			    		}
			    	}
		    	}
		    }
		}	//end neighbor iteration

		fPressure *= -CONST_MASS * p.getDensity();
		fViscosity *= CONST_VISCOSITY * CONST_MASS;

		if ( Length(smoothedColorFieldGradient) > CONST_SURFACE_TENSION_FORCE_THRESHOLD ) 
			fSurface = -CONST_SURFACE_TENSION * smoothedColorFieldLaplacian * smoothedColorFieldGradient / Length(smoothedColorFieldGradient);

		p.setForce(fPressure + fViscosity + fSurface + fGravity);
	}

}