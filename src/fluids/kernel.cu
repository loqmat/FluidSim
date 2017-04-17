#include "cuda.h"

#include "fluid/cuda_setup.hpp"
#include "fluid/setup.hpp"
#include "fluid/graphics.hpp"
#include "fluid/camera.hpp"
#include "fluid/cuda_buffer.hpp"
#include "fluid/cuda_uniform_buffer.hpp"

#include <vector>
#include <iostream>

using namespace Fluids;

#define VERTEX_COUNT 1048576
#define THREAD_COUNT 512

void create_flat_shader(Shader& shad);

struct Vertex {
	core::vec3 position;
	core::vec3 velocity;
};

__global__ void setup_matrices(core::vec4* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	output[i].x = 0;
	output[i].y = 0;
	output[i].z = 0;
	output[i].w = 0;
}
__global__ void gravity(double dt, Vertex* input, core::vec4* output) {
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

	output[i].x = input[i].position.x;
	output[i].y = input[i].position.y;
	output[i].z = input[i].position.z;
}

void run(const std::vector<std::string>& args) {
	InitGLFW glfw;
	Window mainWindow("Fluid Simulator", 1080, 800);
	glfwMakeContextCurrent(mainWindow);

	InitGLEW glew;
	glfwSwapInterval(1);

	CUDA cuda;
	cuda.Setup();
	cuda.GLSetup();

	glEnable( GL_PROGRAM_POINT_SIZE );

	GLint draw_iterations;
	GLint max_uniform_buffer_range;
	GLint max_uniform_buffer_units;

	glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_uniform_buffer_range);

	draw_iterations = 4 * VERTEX_COUNT / max_uniform_buffer_range;
	max_uniform_buffer_units = max_uniform_buffer_range / 4;

	std::cerr << "Max Number of Floats: " << max_uniform_buffer_range << std::endl;
	std::cerr << "Number of Draw Calls: " << draw_iterations << std::endl;

	GLuint vertex_buffer;
	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	{
		core::vec4* position = new core::vec4[max_uniform_buffer_units];
		for (int i=0;i<max_uniform_buffer_units;i++)
			position[i].w = 1.0f;
		glBufferData(GL_ARRAY_BUFFER, sizeof(core::vec4) * max_uniform_buffer_units, position, GL_STATIC_DRAW);
		delete[] position;
	}

	UniformBuffer matrix_data(sizeof(core::vec4) * VERTEX_COUNT);
	CUDABuffer input_data(sizeof(Vertex) * VERTEX_COUNT);
	{
		float position_range = 2.0f;
		float velocity_range = 8.0f;
		Vertex* data = new Vertex[VERTEX_COUNT];
		srand(time(NULL));
		for( int i=0;i<VERTEX_COUNT;i++ ) {
			data[i].position.x = position_range * ((float)rand() / RAND_MAX - 0.5f);
			data[i].position.y = position_range * ((float)rand() / RAND_MAX - 0.5f);
			data[i].position.z = position_range * ((float)rand() / RAND_MAX - 0.5f);

			data[i].velocity.x = velocity_range * ((float)rand() / RAND_MAX - 0.5f);
			data[i].velocity.y = velocity_range * ((float)rand() / RAND_MAX - 0.5f);
			data[i].velocity.z = velocity_range * ((float)rand() / RAND_MAX - 0.5f);
		}
		input_data.upload((void*)data);
		delete[] data;
	}

	Camera main_camera;
	main_camera.arm_length = 32.0f;

	Shader shader_flat;
	GLuint shader_proj_view = 0;

	create_flat_shader(shader_flat);
	glUseProgram(shader_flat);
	shader_proj_view = glGetUniformLocation(shader_flat, "u_projection_view");

	GLuint modelview_index = glGetUniformBlockIndex(shader_flat, "ModelView");   
	glUniformBlockBinding(shader_flat, modelview_index, 0);

	if ( glGetError() != GL_NO_ERROR ) {
		throw "Got OpenGL Error during Setup!";
	}

	setup_matrices<<<VERTEX_COUNT/THREAD_COUNT,THREAD_COUNT>>>((core::vec4*)matrix_data.bindCUDA());
	checkCUDAResult();
	matrix_data.unbindCUDA();

	double _current_time = glfwGetTime();
	double _delta_time = 0.0;
	while (!glfwWindowShouldClose(mainWindow))
	{
		//--------------------------------------------------------------------------------------------------
		// CUDA Segment
		//--------------------------------------------------------------------------------------------------
		{
			gravity<<<VERTEX_COUNT/THREAD_COUNT,THREAD_COUNT>>>(_delta_time, (Vertex*)input_data, (core::vec4*)matrix_data.bindCUDA());
			checkCUDAResult();
			matrix_data.unbindCUDA();
		}

		//--------------------------------------------------------------------------------------------------
		// OpenGL Segment
		//--------------------------------------------------------------------------------------------------
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

			glViewport(0, 0, width, height);
			glClear(GL_COLOR_BUFFER_BIT);

			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

			for ( int iteration=0;iteration<draw_iterations;iteration++ ) {
				matrix_data.bindGL(0, iteration * max_uniform_buffer_range, max_uniform_buffer_range);
				glDrawArrays(GL_POINTS, 0, max_uniform_buffer_units);
			}

			glFinish();
		}

		//--------------------------------------------------------------------------------------------------
		// Finish Frame
		//--------------------------------------------------------------------------------------------------
		{
			glfwSwapBuffers(mainWindow);
			glfwPollEvents();

			double ct = glfwGetTime();
			_delta_time = ct - _current_time;
			_current_time = ct;

			if ( glGetError() != GL_NO_ERROR ) {
				throw "Got OpenGL Error in Frame!";
			}
		}
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

void create_flat_shader(Shader& shad) {
	shad.vertex =
		"#version 330\n"
		"layout (std140) uniform ModelView"
		"{"
		"	vec4 u_model[1024];"
		"};"
		"uniform mat4 u_projection_view;"
		"in vec4 in_position;"
		"void main() {"
		"	vec4 local = u_model[gl_VertexID];"
		"   gl_Position = u_projection_view * (local + in_position);"
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