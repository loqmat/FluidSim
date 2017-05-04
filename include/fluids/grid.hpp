#ifndef GRID_HPP
#define GRID_HPP

#include <thrust/device_vector.h>

#include "core/vector.hpp"
#include "cuda_buffer.hpp"
#include "cuda_uniform_buffer.hpp"
#include "simulation_defs.hpp"

#define GRAVITATIONAL_ACCELERATION 9.80665 //m/s^2
#define CONST_H 1.2 //between 0 and 0.5 //.0457
#define CONST_MASS 1.0
#define GAS_CONSTANT 461.5
#define CONST_REST_DENSITY .99829 //kg/m^3
#define CONST_VISCOSITY 1.5
#define CONST_SURFACE_TENSION_FORCE_THRESHOLD 7.065
#define CONST_SURFACE_TENSION 0.0728
#define CONST_SURFACE_PARTICLE_THRESHOLD 1.0

namespace Fluids {
	class grid {
	public:

		int perf_count;
		float perf_sort_time;
		float perf_calc_pressure;
		float perf_calc_forces;
		float perf_integrate;
		float perf_visualize;

		class device_data {
		public:
			int4 dimensions;
			int particle_count;
			int* cell_count;
			int* cell_offset;
			int* sorted_cell_id;
			int* cell_index;
			float4* velocity;
			float4* force;
			float* density;
			float* pressure;
			int* sorted_particle_id;
			float4* positions;
			float4* color_field;

			device_data(grid& g, float4* pos) :
				positions(pos),
				dimensions(g._dimensions),
				particle_count(g._particle_count),
				cell_count(thrust::raw_pointer_cast(g._cell_count.data())),
				cell_offset(thrust::raw_pointer_cast(g._cell_offset.data())),
				sorted_cell_id(thrust::raw_pointer_cast(g._sorted_cell_id.data())),
				cell_index(thrust::raw_pointer_cast(g._cell_index.data())),
				velocity(thrust::raw_pointer_cast(g._velocity.data())),
				force(thrust::raw_pointer_cast(g._force.data())),
				density(thrust::raw_pointer_cast(g._density.data())),
				pressure(thrust::raw_pointer_cast(g._pressure.data())),
				sorted_particle_id(thrust::raw_pointer_cast(g._sorted_particle_id.data())),
				color_field(thrust::raw_pointer_cast(g._color_field.data())) { ; }
		};

		grid(int length, int width, int depth, int count);

		int getParticleCount () { return _particle_count; }
		int getVolume () { return _dimensions.x*_dimensions.y*_dimensions.z; }

		thrust::device_vector<int>& sortedCellID () { return _sorted_cell_id; }
		thrust::device_vector<int>& sortedParticleID () { return _sorted_particle_id; }
		thrust::device_vector<int>& cellID 	() { return _cell_index; }
		thrust::device_vector<int>& cellCount () { return _cell_count; }
		thrust::device_vector<int>& cellOffset () { return _cell_offset; }

		float4* bindPositions() { return _positions.bindCUDA(); }
		void unbindPositions() { _positions.unbindCUDA(); }

		UniformBuffer<float4>& getUniformBuffer() { return _positions; }

		void uploadData(device_data& data) { _device_data.upload(&data); }
		device_data& getUploadedData() { return *((device_data*)_device_data); }

		CUDA_DEVICE_FUNCTION static void wallCollision( device_data&, int i );
		CUDA_DEVICE_FUNCTION static void reassignParticle( device_data&, int i );

		//smoothing kernel
		CUDA_DEVICE_FUNCTION static float WPoly6 ( float r );
		CUDA_DEVICE_FUNCTION static float4 gradientWPoly6 ( float4& r, float d );
		CUDA_DEVICE_FUNCTION static float laplacianWPoly6 ( float r );
		CUDA_DEVICE_FUNCTION static float4 gradientWSpiky ( float4& d, float r );
		CUDA_DEVICE_FUNCTION static float laplacianWViscosity ( float r );

		CUDA_DEVICE_FUNCTION static void integrate( device_data&, int i, double dt );
		CUDA_DEVICE_FUNCTION static void calculatePressure ( device_data&, int i );
		CUDA_DEVICE_FUNCTION static void calculateForces ( device_data&, int i );

	private:

		CUDABuffer<device_data> _device_data;

		int4 _dimensions;
		int _particle_count;

		//cell data
		thrust::device_vector<int> _cell_count;
		thrust::device_vector<int> _cell_offset;
		thrust::device_vector<int> _sorted_cell_id;

		//particle data
		thrust::device_vector<int> _cell_index;
		thrust::device_vector<float4> _velocity;
		thrust::device_vector<float4> _force;
		thrust::device_vector<float> _density;
		thrust::device_vector<float> _pressure;
		thrust::device_vector<int> _sorted_particle_id;

		thrust::device_vector<float4> _color_field;

		UniformBuffer<float4> _positions;

	};

	__global__ void calculatePressure(grid::device_data&);
	__global__ void calculateForces(grid::device_data&);
	__global__ void integrate(grid::device_data&, double dt);
	void runCUDASimulation(bool visualize, grid&, MarchingCubes&, double dt, unsigned int frame);
}

#endif
