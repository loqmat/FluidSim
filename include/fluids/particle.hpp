#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "common.hpp"
#include "cuda_setup.hpp"
#include "core/vector.hpp"

namespace Fluids {
	class particle {

	public: 

		//CONSTRUCTORS
		CUDA_SHARED_FUNCTION particle() {		
			_velocity = core::vec3(0.0,0.0,0.0); //0
			_force = core::vec3(0.0,-9.80665,0.0); //gravity only
			_density = 0.0;
			_pressure = 0.0;
			_cell_index = -1;
		}

		//ACCESSORS
	 	CUDA_SHARED_FUNCTION const core::vec3& getVelocity() const { return _velocity; }
	 	CUDA_SHARED_FUNCTION const core::vec3& getForce() const { return _force; }
	  	CUDA_SHARED_FUNCTION float getDensity() const  { return _density; }
	  	CUDA_SHARED_FUNCTION float getPressure() const { return _pressure; }
	  	CUDA_SHARED_FUNCTION int getCellIndex () const { return _cell_index; }

	  	//MODIFIERS
	 	CUDA_SHARED_FUNCTION void setVelocity( const core::vec3& velocity ) { _velocity = velocity; }
	 	CUDA_SHARED_FUNCTION void setForce( const core::vec3& force ) { _force = force; }
	  	CUDA_SHARED_FUNCTION void setDensity( float density )  { _density = density; }
	  	CUDA_SHARED_FUNCTION void setPressure( float pressure ) { _pressure = pressure; }
		CUDA_SHARED_FUNCTION void assignCell (int index) { _cell_index = index; }

	private:

		core::vec3 _velocity;
		core::vec3 _force;
		float _density;
		float _pressure;
		int _cell_index; 

	};
}

#endif
