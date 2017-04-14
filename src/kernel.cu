#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "setup.hpp"
#include "graphics.hpp"

#include "cuda_gl_interop.h"

#include <vector>
#include <iostream>

#define deg2rad 0.01745329251994329576923690768489f

using namespace Fluids;

void run(const std::vector<std::string>& args) {
	InitGLFW glfw;
	Window mainWindow("Fluid Simulator", 640, 480);
	glfwMakeContextCurrent(mainWindow);

	InitGLEW glew;
	glfwSwapInterval(1);

	Shader flat;
	flat.vertex =
		"#version 150\n"
		"in vec2 in_position;"
		"void main() {"
		"	gl_Position = vec4(in_position, 0, 1);"
		"}";
	flat.fragment =
		"#version 150\n"
		"out vec4 out_color;"
		"void main() {"
		"	out_color = vec4(1, 1, 1, 1);"
		"}";
	flat.link();

	GLuint gl_buffer;
	cudaGraphicsResource_t cuda_resource;

	float data[6]{std::cos(  0.0f * deg2rad), std::sin(  0.0f * deg2rad),
				  std::cos(120.0f * deg2rad), std::sin(120.0f * deg2rad),
				  std::cos(240.0f * deg2rad), std::sin(240.0f * deg2rad)};

	glGenBuffers(1, &gl_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, gl_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * 2, (void*)data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_buffer, cudaGraphicsRegisterFlagsNone);

	while (!glfwWindowShouldClose(mainWindow))
	{
		int width, height;
		glfwGetFramebufferSize(mainWindow, &width, &height);

		float* buffer_data;
		size_t buffer_size;

		cudaGraphicsMapResources( 1, &cuda_resource, 0 );
		cudaGraphicsResourceGetMappedPointer( (void**)&buffer_data, &buffer_size, cuda_resource );

		// CUDA STUFF

		cudaDeviceSynchronize();
		cudaGraphicsUnmapResources( 1, &cuda_resource, 0 );

		glViewport(0, 0, width, height);

		glUseProgram(flat);
		glBindBuffer(GL_ARRAY_BUFFER, gl_buffer);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		
		glDrawArrays(GL_TRIANGLES, 0, 3);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glUseProgram(0);

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