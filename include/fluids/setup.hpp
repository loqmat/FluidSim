#pragma once

#ifndef __SETUP_HPP__
#define __SETUP_HPP__

#include "gl/glew.h"
#include "glfw/glfw3.h"

#define deg2rad 0.01745329251994329576923690768489f

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
			void initialize() const;
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