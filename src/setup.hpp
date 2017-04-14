#pragma once

#ifndef __SETUP_HPP__
#define __SETUP_HPP__

#include "gl/glew.h"
#include "glfw/glfw3.h"

namespace Fluids {

	typedef unsigned int uint;

	struct InitGLFW {
		private:
			bool _success;
		public:
			InitGLFW();
			~InitGLFW();
	};

	struct InitGLEW {
		public:
			InitGLEW();
	};

	struct Window {
		private:
			GLFWwindow* _window;
		public:
			Window(const char* name, int x, int y);
			~Window();
			operator GLFWwindow*();
			GLFWwindow* operator-> ();
	};
}

#endif//__SETUP_HPP__