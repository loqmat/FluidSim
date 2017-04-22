particle( grid& source, vec3 position ) {

	_source = source
	_position = position
	_velocity = vec3(0.0,0.0,0.0) //0
	_force = vec3(0.0,-9.80665,0.0) //gravity only
	_density = 0.0
	_pressure = 0.0
	_neighbor_count = 0;
}