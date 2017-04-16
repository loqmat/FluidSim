#ifndef __CONVERT_H__
#define __CONVERT_H__

#include "core/matrix.hpp"
#include "core/axis_angle.hpp"
#include "core/euler_angles.hpp"
#include "core/quaternion.hpp"

namespace core {

	// convert 2D rotations
	float32 toFloat (const mat2&);
	mat2 toMatrix (float32);

	// convert 3D rotations to different representations
	mat3 toMatrix (const eulerangles_t&);
	mat3 toMatrix (const axisangle_t&);
	mat3 toMatrix (const quaternion_t&);

	axisangle_t toAxisAngle (const eulerangles_t&);
	axisangle_t toAxisAngle (const mat3&);
	axisangle_t toAxisAngle (const quaternion_t&);

	eulerangles_t toEuler (const mat3&);
	eulerangles_t toEuler (const axisangle_t&);
	eulerangles_t toEuler (const quaternion_t&);

	quaternion_t toQuaternion (const mat3&);
	quaternion_t toQuaternion (const axisangle_t&);
	quaternion_t toQuaternion (const eulerangles_t&);

}

#endif//__CONVERT_H__
