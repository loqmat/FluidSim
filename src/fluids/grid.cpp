#include "fluid/grid.hpp"

namespace Fluids {

void grid::cell::addParticle ( int p ) {
	for (int i = 0; i < 8; i++) {
		if ( _my_particles[i] == -1 ) {
			_my_particles[i] = p;
			return;
		}
	}
}

void grid::cell::removeParticle ( int p) {
	for (int i = 0; i < 8; i++) {
		if ( _my_particles[i] == p ) {
			_my_particles[i] = -1;
			return;
		}
	}
}


grid::grid(int length, int width, int depth): _dimensions(length,width,depth), _cells(sizeof(cell)*length*width*depth), 
_particles(sizeof(particle)*length*width*depth), _positions(sizeof(vec3)*length*width*depth) {

	int volume = length*width*depth;
	cell* cells = new cell [volume];
	particle* particles = new particle [volume];
	vec3* positions = new vec3 [volume];
	_particle_count = 0.2*(volume);

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < depth; k++) {
				cells[i+j*length+k*length*width].setStart(i,j,k);
			}
		}
	}

	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < 0.2*depth; k++) {
				int index = i+j*length+k*length*width;
				positions[index] = vec3(i,j,k);
				particles[index].assignCell(index);
				cells[index].addParticle(index);
			}
		}
	}

	_cells.upload((void*)cells);
	_particles.upload((void*)particles);

	glBindBuffer(GL_COPY_WRITE_BUFFER, _positions.handleGL());
	glBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(core::vec3) * volume, (void*)positions);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	
	delete[] cells;
	delete[] particles;
	delete[] positions;
}

void grid::reassignParticle( int i ) {

	particle& p = _particles[i];
	vec3& pos = _positions[i];
	int cindex = p.getCellIndex();
	const vec3& start = _cells[cindex].getStart();
	_cells[cindex].removeParticle(i);

	if (pos.x < start.x) 			cindex--;
	else if (pos.x >= start.x+1) 	cindex++;
	if (pos.y < start.y) 			cindex-=_dimensions.x;
	else if (pos.y >= start.y+1) 	cindex+=_dimensions.x;
	if (pos.z < start.z) 			cindex-=_dimensions.x*_dimensions.y;
	else if (pos.z >= start.z+1) 	cindex+=_dimensions.x*_dimensions.y;

	_cells[cindex].addParticle(i);
	p.assignCell(cindex);

}



float grid::WPoly6 ( float r ) {

	return ( 315.0/(64.0*MATH_PI*pow(H,9)) ) * pow((H*H-r*r),3 );

}

vec3 grid::gradientWPoly6 ( vec3& r, float d ) {

	return ( -945.0/(32.0*MATH*PI*pow(H,9)) * pow((H*H-d*d),2) * r );

}

float grid::laplacianWpoly6 ( float r ) {
    
  	return ( -945.0/(32.0*MATH_PI*pow(H,9)) * (H*H-r*r) * (3.0*H*H - 7.0*r*r);

}

vec3 grid::gradientWspiky ( vec3& r, float d ) {
 
	//Wspiky = ( 15.0/(MATH_PI*pow(H,6))) * pow((H-r),3 );
	return ( (-45.0/(MATH_PI*pow(H,6)) * pow(H-d, 2) * r/d );

}

float grid::laplacianWviscosity ( float r ) {

	//return ( (15.0/(2.0*MATH_PI*pow(H,3))) * (-pow(r,3)/(2*pow(H,3)) + ((r*r)/(H*H)) + H/(2.0*r) - 1) ); 
  	return ( 45.0/(MATH_PI*pow(H,6)) * (H - r) );   

}

void grid::integrate( float dt ) {

	calculateForces();

	for (int i = 0; i < _particle_count; i++) {

		particle& p = _particles[i];
		vec3 oldPosition = _positions[i];
		vec3 acceleration = p.getForce()/p.getDensity();
		_positions[i] = oldPosition + p.getVelocity()*dt + 0.5*acceleration*dt*dt;
		p.setVelocity( (_positions[i] - oldPosition)/dt );
		wallCollision(i);
		reassignParticle(i);
	}

}

void grid::wallCollision( int i ) {
	particle& p = _particles[i];
	vec3& pos = _positions[i];
	vec3 vel = p.getVelocity();

	if (pos.x < 0) {
		pos.x = -pos.x;
		vel.x = -vel.x;
	}
	else if (pos.x > _dimensions.x) {
		pos.x = 2*_dimensions.x - pos.x;
		vel.x = -vel.x;
	}
	if (pos.y < 0) {
		pos.y = -pos.y;
		vel.y = -vel.y;
	}
	else if (pos.y > _dimensions.y) {
		pos.y = 2*_dimensions.y - pos.y;
		vel.y = -vel.y;
	}
	if (pos.z < 0) {
		pos.z = -pos.z;
		vel.z = -vel.z;
	}
	else if (pos.z > _dimensions.z) {
		pos.z = 2*_dimensions.z - pos.z;
		vel.z = -vel.z;
	}

	p.setVelocity(vel);
}

void grid::calculateForces ( ) {
	
	for (int i = 0; i < _particle_count; i++) {
		particle& p = _particles[i];
		p.setDensity(0.0); 
		
		//iterate over all neighbors
		for (int x = -1; x < 2; x++) {
	       	if (_positions[i].x + x < 0) continue;
	        else if (_positions[i].x + x >= _dimensions.x) break;
	        
	       	for (int y = -1; y < 2; y++) {
	       		if (_positions[i].y + y < 0) continue;
	        	else if (_positions[i].y + y >= _dimensions.y ) break;
	          
		       	for (int z = -1; z < 2; z++) {
		       		if (_positions[i].z + z < 0) continue;
		        	else if (_positions[i].z + z >= _dimensions.z ) break;

		        	int cell = p.getCellIndex() + x+y*_dimensions.x+z*_dimensions.x*_dimensions.y;
		        	for (int j = 0; j < 8; j++) {
		        		
		        		if (_cells[cell].getParticle(j) == -1 or _cells[cell].getParticle(j) == i ) continue;

		        		particle& neighbor = _particles[ _cells[cell].getParticle(j) ];
		        		
						d = Distance(_positions[i], _positions[_cells[cell].getParticle(j)]);
						if ( d*d <= H*H )
							p.setDensity( p.getDensity + WPoly6(d)*MASS );
					}
				}
			}
		}	//end neighbor calcs
		p.setPressure( GAS_CONSTANT*(p.getDensity()-REST_DENSITY) );
	}

	for (int i = 0; i < _particle_count; i++) {
		particle& p = _particles[i];
		vec3 fGravity (0.0, particle.getDensity()*GRAVITATIONAL_ACCELERATION, 0.0);
		vec3 fPressure;
    	vec3 fViscosity;
		vec3 fSurface;
		vec3 smoothedColorFieldGradient;
		vec3 smoothedColorFieldLaplacian;
		
		//iterate over all neighbors

		for (int x = -1; x < 2; x++) {
	       	if (_positions[i].x + x < 0) continue;
	        else if ( _positions[i].x + x >= _dimensions.x) break;
	        
	       	for (int y = -1; y < 2; y++) {
	       		if (_positions[i].y + y < 0) continue;
	        	else if ( _positions[i].y + y >= _dimensions.y ) break;
	          
		       	for (int z = -1; z < 2; z++) {
		       		if (_positions[i].z + z < 0) continue;
		        	else if ( _positions[i].z + z >= _dimensions.z ) break;

		        	int cell = p.getCellIndex() + x+y*_dimensions.x+z*_dimensions.x*_dimensions.y;

		        	for (int j = 0; j < 8; j++) {

		        		if (_cells[cell].getParticle(j) == -1 or _cells[cell].getParticle(j) == i ) continue;
		        		
		        		particle& neighbor = _particles[ _cells[cell].getParticle(j) ];
						d = Distance(_positions[i], _positions[_cells[cell].getParticle(j)]);
						r = _positions[i] - _positions[_cells[cell].getParticle(j)];

						if ( r*r <= H*H ) {

							fPressure += (p.getPressure()/pow(p.getDensity(),2) + neighbor.getPressure()/pow(neighbor.getDensity(),2)) * gradientWspiky(r,d);
							fViscosity += (neighbor.getVelocity() - p.getVelocity()) * laplacianWviscosity(r) / neighbor.getDensity();
			        		smoothedColorFieldLaplacian += MASS*laplacianWpoly6(r)/neighbor.getDensity();
							smoothedColorFieldGradient += MASS*gradientWPoly(r,d)/neighbor.getDensity();
							
			    		}
			    	}
		    	}
		    }
		}	//end neighbor iteration

		fPressure *= -MASS * p.getDensity();
		fViscosity *= VISCOSITY * MASS;

		if ( Length(colorFieldGradient) > SURFACE_TENSION_FORCE_THRESHOLD ) 
			fSurface = -SURFACE_TENSION * smoothedColorFieldLaplacian * colorFieldGradient / Length(colorFieldGradient);

		p.setForce(fPressure + fViscosity + fSurface + fGravity);
    }
}

}