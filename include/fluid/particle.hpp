#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "core/vector.hpp"

using namespace core;

namespace Fluids {
	class particle {

	public: 

		//CONSTRUCTORS
		particle() {		
			_velocity = vec3(0.0,0.0,0.0); //0
			_force = vec3(0.0,-9.80665,0.0); //gravity only
			_density = 0.0;
			_pressure = 0.0;
			_cell_index = -1;
		}


		//ACCESSORS
	 	const vec3& getVelocity() const { return _velocity; }
	 	const vec3& getForce() const { return _force; }
	  	float getDensity() const  { return _density; }
	  	float getPressure() const { return _pressure; }
	  	int getCellIndex () const { return _cell_index; }

	  	//MODIFIERS
	 	void setVelocity( const vec3& velocity ) { _velocity = velocity; }
	 	void setForce( const vec3& force ) { _force = force; }
	  	void setDensity( float density )  { _density = density; }
	  	void setPressure( float pressure ) { _pressure = pressure; }
		void assignCell (int index) { _cell_index = index; }

	private:

		vec3 _velocity;
		vec3 _force;
		float _density;
		float _pressure;
		int _cell_index; 

	};
}

#endif
