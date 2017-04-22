#ifndef GRID_HPP
#define GRID_HPP

#include "particle.hpp"
#include "vector.hpp"

#define MATH_PI = 3.14159265359	
#define GRAVITATIONAL_ACCELERATION
#define H = 0.3 //between 0 and 0.5
#define MASS = 1.0
#define GAS_CONSTANT 
#define REST_DENSITY
#define VISCOSITY
#define SURFACE_TENSION_FORCE_THRESHOLD
#define SURFACE_TENSION

//grid size? x,y, depth

class grid {

public:

	class cell {
	public: 

		cell ( ) { ; }
		cell (vec3 start) : _start(start) { ; }
		void addParticle ( particle* p );

	private:

		vec3 _start;
		particle* _my_particles[8];	
	};

	grid(int length, int width, int depth);
	particle* addParticle(vec3 position);
	void findNeighbors(particle& start);

	//smoothing kernel
	float Wpoly6 ( float r );
	vec3 gradientWPoly6 ( vec3& r, float d );
	float laplacianWpoly6 ( float r );
	vec3 gradientWspiky ( vec3& r, float d );
	float laplacianWviscosity ( float r );
	

	void integrate( float dt );
	void calculateForces ( );

private:

	vec3i _dimensions;
	cell* _cells;
	particle* _particles;
	int _particle_count;

};

#endif
