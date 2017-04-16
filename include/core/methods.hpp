#ifndef __MISC_FUNCTIONS_H__
#define __MISC_FUNCTIONS_H__

#include "core/vector.hpp"
#include "core/matrix.hpp"

namespace core {

// Core STD Functions
float32 rad2deg (float32);
float32 deg2rad (float32);

float32 Clamp (float32,float32=0.,float32=1.);

int32 MakePow2 (int32);
vec2i MakePow2 (vec2i);
vec3i MakePow2 (vec3i);
vec4i MakePow2 (vec4i);

uint32 MakePow2 (uint32);
vec2ui MakePow2 (vec2ui);
vec3ui MakePow2 (vec3ui);
vec4ui MakePow2 (vec4ui);

}

#endif//__MISC_FUNCTIONS_H__
