#pragma once

#ifndef __GRAPHICS_HPP__
#define __GRAPHICS_HPP__

#include "setup.hpp"

namespace Fluids {

	struct Shader {
	private:
		GLuint _program;

		void __check_shader(uint);
		void __check_program(uint);

	public:
		const char* vertex;
		const char* fragment;

		Shader();
		~Shader();

		operator uint() const;
		void link();
	};

}

#endif//__GRAPHICS_HPP__