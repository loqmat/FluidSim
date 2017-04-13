#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gl/glew.h"
#include "glfw/glfw3.h"

#include <vector>
#include <iostream>

typedef unsigned int uint32;

struct InitGLFW {
private:
	bool _success;
public:
	InitGLFW() : _success(false) {
		if (!glfwInit()) {
			throw "Cannot initialize GLFW";
		}
		_success = true;
	}
	~InitGLFW() {
		glfwTerminate();
	}
};
struct InitGLEW {
public:
	InitGLEW() {
		if (glewInit() != GLEW_OK) {
			throw "Cannot initialize GLEW!";
		}
	}
};
struct Window {
private:
	GLFWwindow* _window;
public:
	Window(const char* name, int x, int y) : _window(glfwCreateWindow(x, y, name, NULL, NULL)) {
		if (!_window) {
			throw "Could not create window!";
		}
	}
	~Window() {
		glfwDestroyWindow(_window);
	}
	operator GLFWwindow*() {
		return _window;
	}
	GLFWwindow* operator-> () {
		return _window;
	}
};

void run(const std::vector<std::string>& args) {
	InitGLFW glfw;
	Window mainWindow("Fluid Simulator", 640, 480);
	glfwMakeContextCurrent(mainWindow);

	InitGLEW glew;

	glfwSwapInterval(1);

	while (!glfwWindowShouldClose(mainWindow))
	{
		float ratio;
		int width, height;
		glfwGetFramebufferSize(mainWindow, &width, &height);
		ratio = width / (float)height;

		glViewport(0, 0, width, height);

		double time = glfwGetTime();

		glfwSwapBuffers(mainWindow);
		glfwPollEvents();
	}
}

int main(int argv, char** argc) {
	try {
		std::vector<std::string> data(argv);
		for (int i = 0; i < argv; i++)
			data[i] = argc[i];
		run(data);
	} catch (const char* err) {
		std::cerr << "Got an error: \"" << err << "\"" << std::endl;
	}
	return 0;
}