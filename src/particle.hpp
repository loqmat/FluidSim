#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "vector.hpp"

class particle {

public: 

	//CONSTRUCTORS
	//can't make particle if there's no grid
	particle( grid& source );
	particle( grid& source, vec3 position );


	//ACCESSORS
	const vec3& getPosition() const { return _position; }
 	const vec3& getVelocity() const { return _velocity; }
 	const vec3& getForce() const { return _force; }
  	float getDensity() const  { return _density; }
  	float getPressure() const { return _pressure; }
  	int getNeighborCount() const { return _neighbor_count; }
  	particle* accessNeighbor( int index ) { return neighbors[index]; }

  	//MODIFIERS
  	void setPosition( const vec3& position ) { _position = position; }
 	void setVelocity( const vec3& velocity ) { _velocity = velocity; }
 	void setForce( const vec3& force ) { _force = force; }
  	void setDensity( float density )  { _density = density; }
  	void setPressure( float pressure ) { _pressure = pressure; }
  	void addNeighbor( particle& neighbor ) {
  		neighbors[_neighbor_count] = neighbor;
  		_neighbor_count++;
  	}
  	void clearNeighbors() { _neighbor_count = 0; }
	void changeCell (size_t inc) {
		_cell += inc	
		_cell -> addParticle (this) 
	}

private:

	vec3 _position;
	vec3 _velocity;
	vec3 _force;
	float _density;
	float _pressure;
	grid& _source;
	cell* _cell;
	int _cell_index; 
	particle* _neighbors[32];
	int _neighbor_count;
};

#endif
