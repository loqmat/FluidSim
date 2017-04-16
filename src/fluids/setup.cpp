#include "fluid/setup.hpp"

namespace Fluids {

	InitGLFW::InitGLFW() : _success(false) {
		if (!glfwInit()) {
			throw "Cannot initialize GLFW";
		}
		_success = true;
	}
	InitGLFW::~InitGLFW() {
		glfwTerminate();
	}

	InitGLEW::InitGLEW() {
		if (glewInit() != GLEW_OK) {
			throw "Cannot initialize GLEW!";
		}
	}

	Window::Window(const char* name, int x, int y) : _window(glfwCreateWindow(x, y, name, NULL, NULL)) {
		if (!_window) {
			throw "Could not create window!";
		}
	}
	Window::~Window() {
		glfwDestroyWindow(_window);
	}
	Window::operator GLFWwindow*() {
		return _window;
	}
	GLFWwindow* Window::operator-> () {
		return _window;
	}
	
}