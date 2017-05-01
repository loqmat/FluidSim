#include "fluids/camera.hpp"
#include "core/convert.hpp"

namespace Fluids {

	Camera::Camera() :
		near_view(0.3f),
		far_view(1000.0f),
		fov(70.0f) { ; }

	void Camera::fillMatrix(float ratio, core::mat4& output) {
		core::mat4 proj = core::perspect( fov, ratio, near_view, far_view );
		core::mat3 mat_angle = core::toMatrix(core::axisangle_t(angle, core::vec3(0,1,0)));
		core::mat3 mat_rise = core::toMatrix(core::axisangle_t(rise, core::vec3(1,0,0)));
		core::mat4 combine = expand4(mat_rise * mat_angle);
		core::mat4 move = core::translate4( -core::vec3(0,0,arm_length) ) * combine * core::translate4(-root_position);
		output = proj * move * core::create4(1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1);
	}

}