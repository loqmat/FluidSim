#ifndef GRID_HPP
#define GRID_HPP

#include "particle.hpp"
#include "core/vector.hpp"
#include "cuda_buffer.hpp"
#include "cuda_uniform_buffer.hpp"

#define GRAVITATIONAL_ACCELERATION -9.80665 //m/s^2
#define CONST_H 0.3 //between 0 and 0.5 //.0457
#define CONST_MASS 1.0
#define GAS_CONSTANT 3.0 
#define CONST_REST_DENSITY 998.29 //kg/m^3
#define CONST_VISCOSITY 3.5
#define CONST_SURFACE_TENSION_FORCE_THRESHOLD 7.065
#define CONST_SURFACE_TENSION 0.0728

using namespace core;

namespace Fluids {
	class grid {

	public:

		class cell {
		
		public: 

			cell () { for (int i = 0; i < 8; _my_particles [i] = -1, i++); }
			cell (vec3 start) : _start(start) { for (int i = 0; i < 8; _my_particles [i] = -1, i++); }

			CUDA_SHARED_FUNCTION void setStart ( float x, float y, float z ) { _start = vec3(x,y,z); }
			CUDA_SHARED_FUNCTION void addParticle ( int p );
			CUDA_SHARED_FUNCTION void removeParticle ( int p);
			
			CUDA_SHARED_FUNCTION const vec3& getStart () const { return _start; }
			CUDA_SHARED_FUNCTION int getParticle (int index) const { return _my_particles[index]; }

		private:

			vec3 _start;
			int _my_particles[8];	
		
		};

		grid(int length, int width, int depth);
		//particle* addParticle(vec3 position);
		CUDA_EXPORTED_FUNCTION void wallCollision( int i );
		CUDA_EXPORTED_FUNCTION void reassignParticle( int i );

		int getParticleCount () { return _particle_count; }
		void bindPositions() { _positions.bindCUDA(); }
		void unbindPositions() { _positions.unbindCUDA(); }

		//smoothing kernel
		CUDA_EXPORTED_FUNCTION float WPoly6 ( float r );
		CUDA_EXPORTED_FUNCTION vec3 gradientWPoly6 ( vec3& r, float d );
		CUDA_EXPORTED_FUNCTION float laplacianWPoly6 ( float r );
		CUDA_EXPORTED_FUNCTION vec3 gradientWSpiky ( vec3& r, float d );
		CUDA_EXPORTED_FUNCTION float laplacianWViscosity ( float r );

		CUDA_EXPORTED_FUNCTION void integrate( float dt, int i );
		CUDA_EXPORTED_FUNCTION void calculateForces ( int i );

	private:

		vec3i _dimensions;
		CUDABuffer<cell> _cells;
		CUDABuffer<particle> _particles;
		UniformBuffer<vec3> _positions;
		int _particle_count;

	};
}

#endif
