#include "cuda.h"

#include "fluid/cuda_err.hpp"
#include "fluid/setup.hpp"
#include "fluid/graphics.hpp"
#include "fluid/camera.hpp"
#include "fluid/cuda_buffer.hpp"
#include "fluid/cuda_uniform_buffer.hpp"

#include <vector>
#include <iostream>

using namespace Fluids;

#define VERTEX_COUNT 1024

void create_flat_shader(Shader& shad);

struct Vertex {
	core::vec3 position;
	core::vec3 velocity;
};

__global__ void setup_matrices(core::mat4* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	output[i].data[ 0] = 1;
	output[i].data[ 1] = 0;
	output[i].data[ 2] = 0;
	output[i].data[ 3] = 0;
	output[i].data[ 4] = 0;
	output[i].data[ 5] = 1;
	output[i].data[ 6] = 0;
	output[i].data[ 7] = 0;
	output[i].data[ 8] = 0;
	output[i].data[ 9] = 0;
	output[i].data[10] = 1;
	output[i].data[11] = 0;
	output[i].data[12] = 0;
	output[i].data[13] = 0;
	output[i].data[14] = 0;
	output[i].data[15] = 1;
}
__global__ void gravity(double dt, Vertex* input, core::mat4* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float distance_to_center = sqrt(input[i].position.x*input[i].position.x +
									input[i].position.y*input[i].position.y + 
									input[i].position.z*input[i].position.z);
	float to_center_x = -input[i].position.x / distance_to_center;
	float to_center_y = -input[i].position.y / distance_to_center;
	float to_center_z = -input[i].position.z / distance_to_center;

	input[i].velocity.x += 0.25f * to_center_x * dt;
	input[i].velocity.y += 0.25f * to_center_y * dt;
	input[i].velocity.z += 0.25f * to_center_z * dt;
	
	input[i].position.x += input[i].velocity.x * dt;
	input[i].position.y += input[i].velocity.y * dt;
	input[i].position.z += input[i].velocity.z * dt;

	output[i].data[12] = input[i].position.x;
	output[i].data[13] = input[i].position.y;
	output[i].data[14] = input[i].position.z;
}

void run(const std::vector<std::string>& args) {
	InitGLFW glfw;
	Window mainWindow("Fluid Simulator", 1024, 700);
	glfwMakeContextCurrent(mainWindow);

	InitGLEW glew;
	glfwSwapInterval(1);

	cudaSetup();
	cudaGLSetup();

	UniformBuffer matrix_data(sizeof(core::mat4) * VERTEX_COUNT);
	CUDABuffer input_data(sizeof(Vertex) * VERTEX_COUNT);
	{
		Vertex data[VERTEX_COUNT];
		srand(time(NULL));
		for( int i=0;i<VERTEX_COUNT;i++ ) {
			data[i].position.x = 2.0f * rand() / RAND_MAX - 1.0f;
			data[i].position.y = 2.0f * rand() / RAND_MAX - 1.0f;
			data[i].position.z = 2.0f * rand() / RAND_MAX - 1.0f;

			data[i].velocity.x = 2.0f * rand() / RAND_MAX - 1.0f;
			data[i].velocity.y = 2.0f * rand() / RAND_MAX - 1.0f;
			data[i].velocity.z = 2.0f * rand() / RAND_MAX - 1.0f;
		}
		input_data.upload((void*)data);
	}

	glEnable( GL_PROGRAM_POINT_SIZE );

	Camera main_camera;
	main_camera.arm_length = 6.0f;

	Shader shader_flat;
	GLuint shader_proj_view = 0;
	GLuint shader_vertex_id = 0;

	create_flat_shader(shader_flat);
	glUseProgram(shader_flat);
	shader_vertex_id = glGetUniformLocation(shader_flat, "u_vertex_id");
	shader_proj_view = glGetUniformLocation(shader_flat, "u_projection_view");

	GLuint modelview_index = glGetUniformBlockIndex(shader_flat, "ModelView");   
	glUniformBlockBinding(shader_flat, modelview_index, 0);

	GLuint vertex_buffer;
	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	{
		core::vec3 position;
		glBufferData(GL_ARRAY_BUFFER, sizeof(position), &position, GL_STATIC_DRAW);
	}

	if ( glGetError() != GL_NO_ERROR ) {
		throw "Got OpenGL Error during Setup!";
	}

	setup_matrices<<<32,32>>>((core::mat4*)matrix_data.bindCUDA());
	checkCUDAResult();
	matrix_data.unbindCUDA();

	matrix_data.bindGL(0);

	double _current_time = glfwGetTime();
	double _delta_time = 0.0;
	while (!glfwWindowShouldClose(mainWindow))
	{
		int width, height;
		glfwGetFramebufferSize(mainWindow, &width, &height);

		main_camera.angle += 0.01f;
		main_camera.rise = std::cos(_current_time / 3.0f);
		{
			core::mat4 data;
			main_camera.fillMatrix((float)width/height, data);
			glUniformMatrix4fv(shader_proj_view, 1, false, (float*)&data);
		}

		gravity<<<32,32>>>(_delta_time, (Vertex*)input_data, (core::mat4*)matrix_data.bindCUDA());
		checkCUDAResult();
		matrix_data.unbindCUDA();

		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		for( int i=0;i<VERTEX_COUNT;i++ ) {
			glUniform1i(shader_vertex_id, i);
			glDrawArrays(GL_POINTS, 0, 1);
		}

		glFinish();

		glfwSwapBuffers(mainWindow);
		glfwPollEvents();

		double ct = glfwGetTime();
		_delta_time = ct - _current_time;
		_current_time = ct;

		if ( glGetError() != GL_NO_ERROR ) {
			throw "Got OpenGL Error in Frame!";
		}
	}

	matrix_data.unbindGL(0);
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

void create_flat_shader(Shader& shad) {
	shad.vertex =
		"#version 330\n"
		"layout (std140) uniform ModelView"
		"{"
		"	mat4 u_model[1024];"
		"};"
		"uniform mat4 u_projection_view;"
		"uniform int u_vertex_id;"
		"in vec3 in_position;"
		"void main() {"
		"	mat4 local = u_model[u_vertex_id];"
		"   gl_Position = u_projection_view * local * vec4(in_position, 1);"
		"	gl_PointSize = max(1.0, min(8.0, 8.0 / gl_Position.z));"
		"}";
		
	shad.fragment =
		"#version 330\n"
		"out vec4 out_color;"
		"void main() {"
		"	out_color = vec4(1,1,1,1);"
		"}";
	shad.link();
}