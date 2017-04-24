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
		//std::cerr << "------------------------------------------------------------" << std::endl;

		core::vec4* positions = sim.bindPositions();
		checkCUDAResult();
		checkCUDAReturn( cudaDeviceSynchronize() );

		grid::device_data data(sim, positions);
		sim.uploadData(data);

		int block_count = data.particle_count/THREAD_COUNT + ((data.particle_count%THREAD_COUNT > 0)?1:0);
		int thread_count = std::min(data.particle_count, THREAD_COUNT);

		std::cerr << "BLOCK COUNT " << block_count << std::endl;
		std::cerr << "THREAD COUNT " << thread_count << std::endl;
		std::cerr << "PARTICLE COUNT " << data.particle_count << std::endl;

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

		//std::cerr << "FINISHED INTEGRATING POSITIONS" << std::endl;

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

	CUDA_SHARED_FUNCTION void grid::cell::removeParticle ( int p ) {
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
					cells[i+j*length+k*length*width].setStart(i,k,j);
				}
			}
		}

		for (int k = 0; k < depth; k++) {
			for (int j = 0; j < width; j++) {
				for (int i = 0; i < length; i++) {
					int index = i + j*length + k*length*width;
					if ( index >= _particle_count )
						continue;

					//std::cerr << "set particle " << index << std::endl;

					positions[index] = core::vec4(i,k,j,1.0);
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
			pos.y = 2*data.dimensions.y - pos.x;
			vel.y = -vel.y * 0.9f;
		}

		if (pos.z < 0) {
			pos.z = -pos.z;
			vel.z = -vel.z * 0.9f;
		} else if (pos.z > data.dimensions.z) {
			pos.z = 2*data.dimensions.z - pos.x;
			vel.z = -vel.z * 0.9f;
		}

		//printf("Wall collision final: %f %f %f\r\n", vel.x, vel.y, vel.z);

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

		//printf("Reassign particle %d from cell %d to cell %d\r\n", i, p.getCellIndex(), cindex);
		data.cells[cindex].addParticle(i);
		p.assignCell(cindex);

	}



	CUDA_DEVICE_FUNCTION double grid::WPoly6 ( double r2 ) {

		double result = 315.0/(64.0*MATH_PI*pow(CONST_H,9)) * pow(CONST_H*CONST_H-r2,3);
		//printf("%f => WPoly6 => %f\r\n", r2, result);
		return result;

	}

	CUDA_DEVICE_FUNCTION core::vec3 grid::gradientWPoly6 ( core::vec3& d, double r2 ) {

		core::vec3 result = -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * pow((CONST_H*CONST_H-r2),2) * d;
		//printf("(%f %f %f) and %f => gradientWPoly6 => (%f %f %f)\r\n", d.x, d.y, d.z, r2, result.x, result.y, result.z);
		return result;

	}

	CUDA_DEVICE_FUNCTION double grid::laplacianWPoly6 ( double r2 ) {
	    
	    double result = -945.0/(32.0*MATH_PI*pow(CONST_H,9)) * (CONST_H*CONST_H-r2) * (3.0*CONST_H*CONST_H - 7.0*r2);
	    //printf("%f => laplacianWPoly6 => %f\r\n", r2, result);
	  	return result;

	}

	CUDA_DEVICE_FUNCTION core::vec3 grid::gradientWSpiky ( core::vec3& d, double r2 ) {
	 
		//Wspiky = ( 15.0/(MATH_PI*pow(CONST_H,6))) * pow((CONST_H-r),3 );
		double r = std::sqrt(r2);
		core::vec3 result = -45.0/(MATH_PI*pow(CONST_H,6)) * pow(CONST_H-r, 2) * d/r;
		//printf("(%f %f %f) and %f => gradientWSpiky => (%f %f %f)\r\n", d.x, d.y, d.z, r2, result.x, result.y, result.z);
		return result;

	}

	CUDA_DEVICE_FUNCTION double grid::laplacianWViscosity ( double r2 ) {

		//return ( (15.0/(2.0*MATH_PI*pow(CONST_H,3))) * (-pow(r,3)/(2*pow(CONST_H,3)) + ((r*r)/(CONST_H*CONST_H)) + CONST_H/(2.0*r) - 1) ); 
	  	double r = std::sqrt(r2);
	  	double result = 45.0/(MATH_PI*pow(CONST_H,6)) * (CONST_H - r);
	  	//printf("%f => laplacianWViscosity => %f\r\n", r2, result);
	  	return result;

	}

	CUDA_DEVICE_FUNCTION void grid::integrate( device_data& data, int i, double dt ) {

		particle& p = data.particles[i];
		
		core::vec4 oldPosition = data.positions[i];
		core::vec3 acceleration = p.getForce() / p.getDensity();
		core::vec3 newPosition = oldPosition.xyz() + p.getVelocity() * dt + 0.5 * acceleration * dt * dt;

		/*printf("FIRST %d: (%f %f %f) => (%f %f %f), velocity(%f %f %f), force(%f %f %f), acceleration(%f %f %f), density(%f)\r\n", 
			i,
			oldPosition.x, oldPosition.y, oldPosition.z,
			temp.x, temp.y, temp.z,
			p.getVelocity().x, p.getVelocity().y, p.getVelocity().z,
			p.getForce().x, p.getForce().y, p.getForce().z,
			acceleration.x, acceleration.y, acceleration.z,
			p.getDensity());*/

		data.positions[i] = core::vec4(newPosition.x, newPosition.y, newPosition.z, 1.0f);
		p.setVelocity( (data.positions[i] - oldPosition).xyz() / dt );

		//wallCollision(data, i);
		//reassignParticle(data, i);

		/*printf("NEXT %d: (%f %f %f) => (%f %f %f), velocity(%f %f %f), force(%f %f %f), acceleration(%f %f %f), density(%f)\r\n", 
			i,
			oldPosition.x, oldPosition.y, oldPosition.z,
			temp.x, temp.y, temp.z,
			p.getVelocity().x, p.getVelocity().y, p.getVelocity().z,
			p.getForce().x, p.getForce().y, p.getForce().z,
			acceleration.x, acceleration.y, acceleration.z,
			p.getDensity());*/

	}

	CUDA_DEVICE_FUNCTION void grid::calculatePressure ( device_data& data, int i ) {
		particle& p = data.particles[i];
		p.setDensity(0.0001);
		//printf("test particle %d in cell %d\r\n", i, p.getCellIndex());
		//printf("location is %f %f %f\r\n", data.positions[i].x, data.positions[i].y, data.positions[i].z);

		for (int j=0;j<data.particle_count;j++ ) {
			if (i == j) continue;

			core::vec4 a = data.positions[i];
    		core::vec4 b = data.positions[j];
			core::vec3 d = (a - b).xyz();
			double r2 = Dot(d, d);
			//printf("for cell %d, distance %d(%f %f %f) => %d(%f %f %f) = %f\r\n", cell, i, a.x,a.y,a.z, neighbor_index, b.x,b.y,b.z, r2);
			if ( r2 <= CONST_H*CONST_H ) {
				p.setDensity( p.getDensity() + WPoly6(r2)*CONST_MASS );
			}
		}
		
		//iterate over all neighbors
		/*for (int x = -1; x < 2; x++) {
	       	if (data.positions[i].x + x < 0) continue;
	        else if (data.positions[i].x + x >= data.dimensions.x) break;
	        
	       	for (int y = -1; y < 2; y++) {
	       		if (data.positions[i].y + y < 0) continue;
	        	else if (data.positions[i].y + y >= data.dimensions.y ) break;
	          
		       	for (int z = -1; z < 2; z++) {
		       		if (data.positions[i].z + z < 0) continue;
		        	else if (data.positions[i].z + z >= data.dimensions.z ) break;

		        	int cell = p.getCellIndex() + x + y*data.dimensions.x + z*data.dimensions.x*data.dimensions.y;
		        	for (int j = 0; j < 8; j++) {

						int neighbor_index = data.cells[cell].getParticle(j);		        		
		        		if (neighbor_index == -1 || neighbor_index == i ) continue;
		        		
		        		core::vec4 a = data.positions[i];
		        		core::vec4 b = data.positions[neighbor_index];
						core::vec3 d = (a - b).xyz();
						double r2 = Dot(d, d);
						//printf("for cell %d, distance %d(%f %f %f) => %d(%f %f %f) = %f\r\n", cell, i, a.x,a.y,a.z, neighbor_index, b.x,b.y,b.z, r2);
						if ( r2 <= CONST_H*CONST_H ) {
							p.setDensity( p.getDensity() + WPoly6(r2)*CONST_MASS );
						}
					}
				}
			}
		}	//end neighbor calcs*/
		//printf("Done with pressure for %d\r\n", i);
		p.setPressure( GAS_CONSTANT*(p.getDensity()-CONST_REST_DENSITY) );
	}

	CUDA_DEVICE_FUNCTION void grid::calculateForces ( device_data& data, int i ) {
		particle& p = data.particles[i];
		
		core::vec3 fGravity = core::vec3(0.0, 0.0, GRAVITATIONAL_ACCELERATION * p.getDensity());
		core::vec3 fPressure;
		core::vec3 fViscosity;
		core::vec3 fSurface;

		core::vec3 smoothedColorFieldGradient;
		double smoothedColorFieldLaplacian = 0.0f;

		for ( int j=0;j<data.particle_count;j++ ) {
			if ( i == j ) continue;

			particle& neighbor = data.particles[ j ];

			core::vec3 d = (data.positions[i] - data.positions[j]).xyz();
			double r2 = Dot(d, d);

			if ( r2 <= CONST_H*CONST_H ) {

				core::vec3 add_pressure;

				if ( r2 > 0.0 ) {
					core::vec3 gradient = gradientWPoly6(d,r2);
					float p_pressure = p.getPressure();
					float p_density = p.getDensity();
					float n_pressure = neighbor.getPressure();
					float n_density = neighbor.getDensity();
					
					//add_pressure = ((p.getPressure()/pow(p.getDensity(),2) + neighbor.getPressure()/pow(neighbor.getDensity(),2)) * gradient);
					add_pressure = (p.getPressure()+neighbor.getPressure())/(2*neighbor.getDensity())*gradient;

					smoothedColorFieldGradient += CONST_MASS * gradient / neighbor.getDensity();
					fPressure += add_pressure;
				}

	    		smoothedColorFieldLaplacian += CONST_MASS * laplacianWPoly6(r2) / neighbor.getDensity();
	    		core::vec3 add_viscosity((neighbor.getVelocity() - p.getVelocity()) * laplacianWViscosity(r2) / neighbor.getDensity());
				fViscosity += add_viscosity;
				
			}
		}
		
		//iterate over all neighbors

		/*for (int x = -1; x < 2; x++) {
	       	if (data.positions[i].x + x < 0) continue;
	        else if ( data.positions[i].x + x >= data.dimensions.x) break;
	        
	       	for (int y = -1; y < 2; y++) {
	       		if (data.positions[i].y + y < 0) continue;
	        	else if ( data.positions[i].y + y >= data.dimensions.y ) break;
	          
		       	for (int z = -1; z < 2; z++) {
		       		if (data.positions[i].z + z < 0) continue;
		        	else if ( data.positions[i].z + z >= data.dimensions.z ) break;

		        	int cell = p.getCellIndex() + x + y*data.dimensions.x + z*data.dimensions.x*data.dimensions.y;

		        	for (int j = 0; j < 8; j++) {

		        		int neighbor_index = data.cells[cell].getParticle(j);
		        		if (neighbor_index == -1 || neighbor_index == i ) continue;
		        		
		        		particle& neighbor = data.particles[ neighbor_index ];

		        		//printf("I am %d and my neighbor is %d\r\n", i, neighbor_index);

		        		if ( p.getDensity() <= 0 ) {
		        			printf("My (%d) density is zero :(\r\n", i);
		        		}
		        		if ( neighbor.getDensity() <= 0 ) {
		        			printf("Neighbor (%d) density is zero :(\r\n", neighbor_index);
		        		}
	
						core::vec3 d = (data.positions[i] - data.positions[neighbor_index]).xyz();
						double r2 = Dot(d, d);

						if ( r2 <= CONST_H*CONST_H ) {

							core::vec3 add_pressure;

							if ( r2 > 0.0 ) {
								core::vec3 gradient = gradientWPoly6(d,r2);
								float p_pressure = p.getPressure();
								float p_density = p.getDensity();
								float n_pressure = neighbor.getPressure();
								float n_density = neighbor.getDensity();
								
								add_pressure = ((p.getPressure()/pow(p.getDensity(),2) + neighbor.getPressure()/pow(neighbor.getDensity(),2)) * gradient);

								//printf("My pressure %f\r\nMy density %f\r\nNeighbor(%d) pressure %f\r\nNeighbor(%d) density %f\r\nGradient (%f %f %f)\r\nAdded pressure (%f %f %f)\r\n",
									//p_pressure, p_density, neighbor_index, n_pressure, neighbor_index, n_density,
									//gradient.x, gradient.y, gradient.z,
									//add_pressure.x, add_pressure.y, add_pressure.z);

								smoothedColorFieldGradient += CONST_MASS * gradient / neighbor.getDensity();
								fPressure += add_pressure;		
							}

			        		smoothedColorFieldLaplacian += CONST_MASS * laplacianWPoly6(r2) / neighbor.getDensity();
			        		core::vec3 add_viscosity((neighbor.getVelocity() - p.getVelocity()) * laplacianWViscosity(r2) / neighbor.getDensity());
							fViscosity += add_viscosity;

							//printf("added pressure (%f %f %f) || added viscosity (%f %f %f)\r\n",
									//add_pressure.x, add_pressure.y, add_pressure.z,
									//add_viscosity.x, add_viscosity.y, add_viscosity.z);
							
			    		}
			    	}
		    	}
		    }
		}	//end neighbor iteration*/

		fPressure *= -CONST_MASS * p.getDensity();
		fViscosity *= CONST_VISCOSITY * CONST_MASS;

		//printf("%f and (%f %f %f)\r\n", smoothedColorFieldLaplacian, smoothedColorFieldGradient.x,smoothedColorFieldGradient.y,smoothedColorFieldGradient.z);

		float smoothedColorFieldGradientLength = Length(smoothedColorFieldGradient);
		if ( smoothedColorFieldGradientLength > CONST_SURFACE_TENSION_FORCE_THRESHOLD ) 
			fSurface = -CONST_SURFACE_TENSION * smoothedColorFieldLaplacian * (smoothedColorFieldGradient / smoothedColorFieldGradientLength);

		/*printf("fPressure(%f %f %f), fViscosity(%f %f %f), fSurface(%f %f %f), fGravity(%f %f %f)\r\n",
			fPressure.x, fPressure.y, fPressure.z,
			fViscosity.x, fViscosity.y, fViscosity.z,
			fSurface.x, fSurface.y, fSurface.z,
			fGravity.x, fGravity.y, fGravity.z);*/

		//p.setForce(fPressure + fSurface + fViscosity + fGravity);
		p.setForce(fPressure + fSurface + fViscosity);
	}

}