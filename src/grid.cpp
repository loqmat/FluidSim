grid(int length, int width, int depth): _dimensions(length,width,depth) {

	int volume = length*width*depth
	_cells = new cell [volume];
	_particles = new particle [volume];
	_particle_count = 0.2*(volume);

	for (int i = 0; i < volume; i++) {

	}

	

}

particle* addParticle(vec3 position);
void findNeighbors(particle& start);



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
		vec3 acceleration = p.getForce()/p.getDensity();
		vec3 position = p.getPosition() + p.getVelocity()*dt + 0.5*acceleration*dt*dt;
		vec3 velocity = (position - p.getPosition())/dt;
		p.setPosition(position);
		p.setVelocity(velocity);
	}

}

void grid::calculateForces ( ) {
	
	for (int i = 0; i < _particle_count; i++) {
		particle& p = _particles[i];
		p.setDensity(0.0); 
		
		//iterate over all neighbors
		for (int j=0; j < p.getNeighborCount(); j++) {
			particle& neighbor = p.accessNeighbor[j];
			d = Distance(p.getPosition(), neighbor.getPosition());
			if ( d*d <= H*H )
				p.setDensity( p.getDensity + WPoly6(d)*MASS );
		}
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
		for (int j=0; j < p.getNeighborCount(); j++) {
			particle& neighbor = p.accessNeighbor[j];
			
			d = Distance(p.getPosition(), neighbor.getPosition());
			r = p.getPosition() - neighbor.getPosition();
			if ( r*r <= H*H ) {

				fPressure += (p.getPressure()/pow(p.getDensity(),2) + neighbor.getPressure()/pow(neighbor.getDensity(),2)) * gradientWspiky(r,d);
				fViscosity += (neighbor.getVelocity() - p.getVelocity()) * laplacianWviscosity(r) / neighbor.getDensity();
        		smoothedColorFieldLaplacian += MASS*laplacianWpoly6(r)/neighbor.getDensity();
				smoothedColorFieldGradient += MASS*gradientWPoly(r,d)/neighbor.getDensity();
				
    		}
    
    		fPressure *= -MASS * p.getDensity();
    		fViscosity *= VISCOSITY * MASS;

    		if ( Length(colorFieldGradient) > SURFACE_TENSION_FORCE_THRESHOLD ) 
    			fSurface = -SURFACE_TENSION * smoothedColorFieldLaplacian * colorFieldGradient / Length(colorFieldGradient);

    		p.setForce(fPressure + fViscosity + fSurface + fGravity);
    	}
    }
}
