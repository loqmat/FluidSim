#ifndef GRID_HPP
#define GRID_HPP

#include "particle.hpp"
#include "core/vector.hpp"
#include "cuda_buffer.hpp"
#include "cuda_uniform_buffer.hpp"

#define GRAVITATIONAL_ACCELERATION -9.80665 //m/s^2
#define H 0.3 //between 0 and 0.5 //.0457
#define MASS 1.0
#define GAS_CONSTANT 3.0 
#define REST_DENSITY 998.29 //kg/m^3
#define VISCOSITY 3.5
#define SURFACE_TENSION_FORCE_THRESHOLD 7.065
#define SURFACE_TENSION 0.0728

using namespace core;

namespace Fluids {
	class grid {

	public:

		class cell {
		
		public: 

			cell () { for (int i = 0; i < 8; _my_particles [i] = -1, i++); }
			cell (vec3 start) : _start(start) { for (int i = 0; i < 8; _my_particles [i] = -1, i++); }

			void setStart ( float x, float y, float z ) { _start = vec3(x,y,z); }
			void addParticle ( int p );
			void removeParticle ( int p);
			
			const vec3& getStart () const { return _start; }
			int getParticle (int index) const { return _my_particles[index]; }

		private:

			vec3 _start;
			int _my_particles[8];	
		
		};

		grid(int length, int width, int depth);
		//particle* addParticle(vec3 position);
		void wallCollision( int i );
		void reassignParticle( int i );

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
		CUDABuffer _cells;
		CUDABuffer _particles;
		UniformBuffer _positions;
		int _particle_count;

	};
}

#endif
