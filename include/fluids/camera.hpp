#pragma once

#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include "core/matrix.hpp"

namespace Fluids {

	struct Camera {
	public:
		float near_view;
		float far_view;
		float fov;

		float angle;
		float rise;
		float arm_length;

		core::vec3 root_position;

		Camera();

		void fillMatrix(float ratio, core::mat4& output);
	};

}

#endif//__CAMERA_HPP__