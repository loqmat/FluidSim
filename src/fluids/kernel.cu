#include "cuda.h"

#include "fluid/cuda_setup.hpp"
#include "fluid/setup.hpp"
#include "fluid/graphics.hpp"
#include "fluid/camera.hpp"
#include "fluid/cuda_buffer.hpp"
#include "fluid/cuda_uniform_buffer.hpp"
#include "fluid/gen_sphere.hpp"

#include <vector>
#include <iostream>

using namespace Fluids;

#define VERTEX_COUNT 1024
#define THREAD_COUNT 512
#define GRAVITY_POINTS 4	

void createFlatShader(Shader& shad);

__global__ void gravity(float dt, core::vec4* gravpts, core::vec4* velocity, core::vec4* position) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float dx[GRAVITY_POINTS],
		  dy[GRAVITY_POINTS],
		  dz[GRAVITY_POINTS],
		  mg[GRAVITY_POINTS];

	for( int j=0;j<GRAVITY_POINTS;j++ ) {
		dx[j] = gravpts[j].x - position[i].x;
		dy[j] = gravpts[j].y - position[i].y;
		dz[j] = gravpts[j].z - position[i].z;

		mg[j] = sqrt(dx[j]*dx[j] + dy[j]*dy[j] + dz[j]*dz[j]) * gravpts[j].w;

		velocity[i].x += dx[j] / mg[j] * dt;
		velocity[i].y += dy[j] / mg[j] * dt;
		velocity[i].z += dz[j] / mg[j] * dt;
	}
	
	position[i].x += velocity[i].x * dt;
	position[i].y += velocity[i].y * dt;
	position[i].z += velocity[i].z * dt;
}

void run(const std::vector<std::string>& args) {
//--------------------------------------------------------------------------------------------------
// Initialize Components
//--------------------------------------------------------------------------------------------------
	InitGLFW glfw;
	Window mainWindow("Fluid Simulator", 1600, 900);
	glfwMakeContextCurrent(mainWindow);

	InitGLEW glew;
	glew.initialize();

	glfwSwapInterval(1);

	CUDA cuda;
	cuda.Setup();
	cuda.GLSetup();

	glEnable( GL_PROGRAM_POINT_SIZE );

//--------------------------------------------------------------------------------------------------
// Discover Rendering Restrictions
//--------------------------------------------------------------------------------------------------

	GLint draw_iterations;
	GLint max_uniform_buffer_range;
	GLint max_uniform_buffer_units;

	glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_uniform_buffer_range);

	draw_iterations = 4 * VERTEX_COUNT / max_uniform_buffer_range;
	max_uniform_buffer_units = max_uniform_buffer_range / 4;

	std::cerr << "Max Number of Vec4s  : " << max_uniform_buffer_units << std::endl;
	std::cerr << "Max Number of Floats : " << max_uniform_buffer_range << std::endl;
	std::cerr << "Number of Draw Calls : " << draw_iterations << std::endl;

//--------------------------------------------------------------------------------------------------
// Generate Draw Buffer
//--------------------------------------------------------------------------------------------------

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

	int draw_vertex_count, // number of vertices in the sphere
		draw_face_count; // number of faces in the sphere
	GLuint array_buffer;
	GLuint index_buffer;

	generateSphere( draw_vertex_count, draw_face_count, array_buffer, index_buffer, 1.0f, 4 );

//--------------------------------------------------------------------------------------------------
// Initialize Position and Velocity Buffers
//--------------------------------------------------------------------------------------------------

	srand(time(NULL));

	UniformBuffer matrix_data(sizeof(core::vec4) * VERTEX_COUNT);
	{
		glBindBuffer(GL_COPY_WRITE_BUFFER, matrix_data.handleGL());
		float position_range = 64.0f;
		core::vec4* position = new core::vec4[VERTEX_COUNT];
		for (int i=0;i<VERTEX_COUNT;i++) {
			position[i].x = position_range * ((float)rand() / RAND_MAX - 0.5f);
			position[i].y = position_range * ((float)rand() / RAND_MAX - 0.5f);
			position[i].z = position_range * ((float)rand() / RAND_MAX - 0.5f);
			position[i].w = 0.0f;
		}
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(core::vec4) * VERTEX_COUNT, (void*)position);
		delete[] position;
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	}

	CUDABuffer gravity_data(sizeof(core::vec4) * GRAVITY_POINTS);
	{
		float gravity_range = 128.0f;
		float strength_range = 0.75f;
		core::vec4* gravpts = new core::vec4[GRAVITY_POINTS];
		for( int i=0;i<GRAVITY_POINTS;i++ ) {
			gravpts[i].x = gravity_range * ((float)rand() / RAND_MAX - 0.5f);
			gravpts[i].y = gravity_range * ((float)rand() / RAND_MAX - 0.5f);
			gravpts[i].z = gravity_range * ((float)rand() / RAND_MAX - 0.5f);
			gravpts[i].w = strength_range * rand() / RAND_MAX + 0.25f;
		}
		gravity_data.upload((void*)gravpts);
		delete[] gravpts;
	}

	CUDABuffer input_data(sizeof(core::vec4) * VERTEX_COUNT);
	{
		float velocity_range = 24.0f;
		core::vec4* velocity = new core::vec4[VERTEX_COUNT];
		for( int i=0;i<VERTEX_COUNT;i++ ) {
			velocity[i].x = velocity_range * ((float)rand() / RAND_MAX - 0.5f);
			velocity[i].y = velocity_range * ((float)rand() / RAND_MAX - 0.5f);
			velocity[i].z = velocity_range * ((float)rand() / RAND_MAX - 0.5f);
			velocity[i].w = 0.0f;
		}
		input_data.upload((void*)velocity);
		delete[] velocity;
	}

//--------------------------------------------------------------------------------------------------
// OpenGL Shaders and Uniforms
//--------------------------------------------------------------------------------------------------

	Camera main_camera;
	main_camera.arm_length = 128.0f;
	main_camera.rise = -1.0f;

	Shader shader_flat;
	GLuint shader_proj_view = 0;
	GLuint shader_light_direc = 0;

	createFlatShader(shader_flat);
	glUseProgram(shader_flat);
	shader_proj_view = glGetUniformLocation(shader_flat, "u_projection_view");
	shader_light_direc = glGetUniformLocation(shader_flat, "u_light_direction");

	GLuint modelview_index = glGetUniformBlockIndex(shader_flat, "ModelView");   
	glUniformBlockBinding(shader_flat, modelview_index, 0);

	if ( glGetError() != GL_NO_ERROR ) 
		throw "Got OpenGL Error during Setup!";

//--------------------------------------------------------------------------------------------------
// MAIN LOOP
//--------------------------------------------------------------------------------------------------
	double _current_time = glfwGetTime();
	double _delta_time = 0.0;
	double _fps = 0.0f;

	while (!glfwWindowShouldClose(mainWindow))
	{

	//----------------------------------------------------------------------------------------------
	// CUDA Segment
	//----------------------------------------------------------------------------------------------
		{
			gravity<<<VERTEX_COUNT/THREAD_COUNT,THREAD_COUNT>>>(
				(float)_delta_time,						// frame time
				(core::vec4*)gravity_data,				// gravity
				(core::vec4*)input_data,				// velocity
				(core::vec4*)matrix_data.bindCUDA());	// position
			checkCUDAResult();
			matrix_data.unbindCUDA();
		}

	//----------------------------------------------------------------------------------------------
	// OpenGL Segment
	//----------------------------------------------------------------------------------------------
		{
			int width, height;
			glfwGetFramebufferSize(mainWindow, &width, &height);

			main_camera.angle += 1.0f * _delta_time;
			main_camera.rise = std::sin(_current_time / 3.0f);
			{
				core::mat4 data;
				main_camera.fillMatrix((float)width/height, data);
				glUniformMatrix4fv(shader_proj_view, 1, false, (float*)&data);
				glUniform3f(shader_light_direc, 0,1,0);
			}

			glViewport(0, 0, width, height);
			glClearColor(1,0,0,1);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			checkGLResult();

			glBindBuffer(GL_ARRAY_BUFFER, array_buffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
			checkGLResult();

			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 0));
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 3));
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 6));
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 8));

			checkGLResult();

			for ( int iteration=0;iteration<draw_iterations;iteration++ ) {
				matrix_data.bindGL(0, iteration * max_uniform_buffer_range, max_uniform_buffer_range);
				glDrawElementsInstanced(GL_TRIANGLES, 3 * draw_face_count, GL_UNSIGNED_BYTE, 0, max_uniform_buffer_units);
			}

			//glDrawElements(GL_TRIANGLES, 3 * draw_face_count, GL_UNSIGNED_BYTE, 0);

			checkGLResult();

			glFinish();
			checkGLResult();
		}

	//----------------------------------------------------------------------------------------------
	// Finish Frame
	//----------------------------------------------------------------------------------------------
		{
			glfwSwapBuffers(mainWindow);
			glfwPollEvents();

			checkGLResult();

			double ct = glfwGetTime();
			_delta_time = ct - _current_time;
			_current_time = ct;
			_fps = 0.9 * _fps + 0.1 * (1.0 / _delta_time);
			//std::cerr << "FPS " << std::setw(10) << _fps << std::endl;
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

void createFlatShader(Shader& shad) {
	shad.vertex =
		"#version 330\r\n"

		"layout (std140) uniform ModelView"
		"{"
		"	vec4 u_model[1024];"
		"};"
		"uniform mat4 u_projection_view;"

		"in vec3 in_position;"
		"in vec3 in_normal;"
		"in vec2 in_uv;"
		"in vec3 in_color;"

		"out vec2 v_uv;"
		"out vec3 v_normal;"
		"out vec3 v_color;"

		"void main() {"
		"	vec4 local = u_model[gl_InstanceID];"
		"   gl_Position = u_projection_view * (local + vec4(in_position, 1));"
		"	v_uv = in_uv;"
		"	v_normal = in_normal;"
		"	v_color = in_color;"
		"}";
		
	shad.fragment =
		"#version 330\r\n"

		"uniform vec3 u_light_direction;"

		"in vec2 v_uv;"
		"in vec3 v_normal;"
		"in vec3 v_color;"

		"out vec4 out_color;"

		"void main() {"
		"	float lighting = 0.8 * clamp(dot(u_light_direction, v_normal) + 0.2, 0, 1) + 0.2;"
		"	out_color = vec4(lighting * v_color,1);"
		"}";
	shad.link();
}