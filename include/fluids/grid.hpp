#ifndef GRID_HPP
#define GRID_HPP

#include "particle.hpp"
#include "core/vector.hpp"
#include "cuda_buffer.hpp"
#include "cuda_uniform_buffer.hpp"

#define GRAVITATIONAL_ACCELERATION -9.80665 //m/s^2
#define CONST_H 5.0 //between 0 and 0.5 //.0457
#define CONST_MASS 10000.0
#define GAS_CONSTANT 3.0 
#define CONST_REST_DENSITY 998.29 //kg/m^3
#define CONST_VISCOSITY 3.5
#define CONST_SURFACE_TENSION_FORCE_THRESHOLD 7.065
#define CONST_SURFACE_TENSION 0.0728

namespace Fluids {
	class grid {

	public:

		class cell {
		
		public: 

			cell () { for (int i = 0; i < 8; _my_particles [i] = -1, i++); }
			cell (core::vec3 start) : _start(start) { for (int i = 0; i < 8; _my_particles [i] = -1, i++); }

			CUDA_SHARED_FUNCTION void setStart ( float x, float y, float z ) { _start = core::vec3(x,y,z); }
			CUDA_SHARED_FUNCTION void addParticle ( int p );
			CUDA_SHARED_FUNCTION void removeParticle ( int p);
			
			CUDA_SHARED_FUNCTION const core::vec3& getStart () const { return _start; }
			CUDA_SHARED_FUNCTION int getParticle (int index) const { return _my_particles[index]; }

		private:

			core::vec3 _start;
			int _my_particles[8];
		
		};

		class device_data {
		public:
			core::vec3i dimensions;
			cell* cells;
			particle* particles;
			core::vec4* positions;
			int particle_count;

			device_data(grid& g, core::vec4* pos) :
				dimensions(g._dimensions),
				cells(g._cells),
				particles(g._particles),
				positions(pos),
				particle_count(g._particle_count) { ; }
		};

		grid(int length, int width, int depth, float filled);
		//particle* addParticle(vec4 position);

		int getParticleCount () { return _particle_count; }
		core::vec4* bindPositions() { return _positions.bindCUDA(); }
		void unbindPositions() { _positions.unbindCUDA(); }
		UniformBuffer<core::vec4>& getUniformBuffer() { return _positions; }

		void uploadData(device_data& data) { _device_data.upload(&data); }
		device_data& getUploadedData() { return *((device_data*)_device_data); }

		CUDA_DEVICE_FUNCTION static void wallCollision( device_data&, int i );
		CUDA_DEVICE_FUNCTION static void reassignParticle( device_data&, int i );

		//smoothing kernel
		CUDA_DEVICE_FUNCTION static float WPoly6 ( float r );
		CUDA_DEVICE_FUNCTION static core::vec3 gradientWPoly6 ( core::vec3& r, float d );
		CUDA_DEVICE_FUNCTION static float laplacianWPoly6 ( float r );
		CUDA_DEVICE_FUNCTION static core::vec3 gradientWSpiky ( core::vec3& r, float d );
		CUDA_DEVICE_FUNCTION static float laplacianWViscosity ( float r );

		CUDA_DEVICE_FUNCTION static void integrate( device_data&, int i, float dt );
		CUDA_DEVICE_FUNCTION static void calculatePressure ( device_data&, int i );
		CUDA_DEVICE_FUNCTION static void calculateForces ( device_data&, int i );

	private:

		CUDABuffer<device_data> _device_data;

		core::vec3i _dimensions;
		CUDABuffer<cell> _cells;
		CUDABuffer<particle> _particles;
		UniformBuffer<core::vec4> _positions;
		int _particle_count;

	};

	__global__ void calculatePressure(grid::device_data&);
	__global__ void calculateForces(grid::device_data&);
	__global__ void integrate(grid::device_data&, double dt);
	void runCUDASimulation(grid&, double dt);
}

#endif
