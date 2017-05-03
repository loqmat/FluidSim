#include "fluids/grid.hpp"
#include "fluids/marching_cubes.hpp"
#include "fluids/cuda_setup.hpp"
#include "fluids/setup.hpp"
#include "fluids/graphics.hpp"
#include "fluids/camera.hpp"
#include "fluids/cuda_buffer.hpp"
#include "fluids/cuda_uniform_buffer.hpp"
#include "fluids/gen_sphere.hpp"

//#include <windows.h>
#include <vector>
#include <iostream>

using namespace Fluids;

#define CUBE_DIMENSIONS 30
#define GRID_DIMENSIONS 10
#define PARTICLE_COUNT 300

void createBoxShader(Shader& shad);
void createSurfaceShader(Shader& shad);
void createFlatShader(Shader& shad);

struct GlobalData {
	static bool rightMousePressed;

	static double xPosition;
	static double yPosition;

	static double xOldPosition;
	static double yOldPosition;

	static double xDeltaPosition;
	static double yDeltaPosition;

	static double xScroll;
	static double yScroll;

	static bool escapePressed;
	static bool spacePressed;

	static void UpdateGlobalData() {
		xDeltaPosition = xPosition - xOldPosition;
		xOldPosition = xPosition;

		yDeltaPosition = yPosition - yOldPosition;
		yOldPosition = yPosition;

		xScroll = 0.0;
		yScroll = 0.0;

		escapePressed = false;
		spacePressed = false;
	}
};

bool GlobalData::rightMousePressed = false;
bool GlobalData::escapePressed = false;
bool GlobalData::spacePressed = false;
double GlobalData::xPosition = 0;
double GlobalData::yPosition = 0;
double GlobalData::xOldPosition = 0;
double GlobalData::yOldPosition = 0;
double GlobalData::xDeltaPosition = 0;
double GlobalData::yDeltaPosition = 0;
double GlobalData::xScroll = 0;
double GlobalData::yScroll = 0;

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	GlobalData::xPosition = xpos;
	GlobalData::yPosition = ypos;
}
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if ( button == GLFW_MOUSE_BUTTON_RIGHT ) {
		if (action == GLFW_PRESS) {
			GlobalData::rightMousePressed = true;
		} else if (action == GLFW_RELEASE) {
			GlobalData::rightMousePressed = false;
		}
	}
}
void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	GlobalData::xScroll = xoffset;
	GlobalData::yScroll = yoffset;
}
void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        GlobalData::spacePressed = true;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        GlobalData::escapePressed = true;
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

	glfwSetCursorPosCallback(mainWindow, mousePositionCallback);
	glfwSetMouseButtonCallback(mainWindow, mouseButtonCallback);
	glfwSetScrollCallback(mainWindow, mouseScrollCallback);
	glfwSetKeyCallback(mainWindow, keyboardCallback);

	CUDA cuda;
	cuda.Setup();
	cuda.GLSetup();

	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

//--------------------------------------------------------------------------------------------------
// Discover Rendering Restrictions
//--------------------------------------------------------------------------------------------------

	GLint max_uniform_buffer_range;
	GLint max_uniform_buffer_units;

	glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_uniform_buffer_range);
	max_uniform_buffer_units = max_uniform_buffer_range / 4;

	std::cerr << "Max Number of Vec4s  : " << max_uniform_buffer_units << std::endl;
	std::cerr << "Max Number of Floats : " << max_uniform_buffer_range << std::endl;

//--------------------------------------------------------------------------------------------------
// Generate Draw Buffer
//--------------------------------------------------------------------------------------------------

	int draw_vertex_count, // number of vertices in the sphere
		draw_face_count; // number of faces in the sphere
	GLuint array_buffer;
	GLuint index_buffer;

	generateSphere( draw_vertex_count, draw_face_count, array_buffer, index_buffer, 0.5f, 4 );

	GLuint draw_cube_buffer;
	glGenBuffers(1,&draw_cube_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, draw_cube_buffer);
	float data[] = {
		0,0,0, 1,0,0,
		0,0,0, 0,1,0,
		0,0,0, 0,0,1,

		1,1,1, 0,1,1,
		1,1,1, 1,0,1,
		1,1,1, 1,1,0,

		1,0,0, 1,0,1,
		1,0,0, 1,1,0,

		0,1,0, 1,1,0, 
		0,1,0, 0,1,1,

		0,0,1, 1,0,1,
		0,0,1, 0,1,1
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

//--------------------------------------------------------------------------------------------------
// MAKE A FANCY, FANCY PARTICLE GRID!!!
//--------------------------------------------------------------------------------------------------

	int cubes_dimension = CUBE_DIMENSIONS;
	float grid_dimension = GRID_DIMENSIONS;

	MarchingCubes cubes(cubes_dimension, cubes_dimension, cubes_dimension,
						core::vec3i(grid_dimension,grid_dimension,grid_dimension));
	grid sim(grid_dimension, grid_dimension, grid_dimension, PARTICLE_COUNT);

//--------------------------------------------------------------------------------------------------
// OpenGL Shaders and Uniforms
//--------------------------------------------------------------------------------------------------

	Camera main_camera;
	main_camera.arm_length = 4.0f;
	main_camera.rise = -1.0f;
	main_camera.root_position = core::vec3(grid_dimension/4,grid_dimension/4,grid_dimension/4);

//------------------------------------------

	Shader shader_box;
	GLuint shader_box_proj_view = 0;
	core::mat4 scaling_matrix = core::create4(
		grid_dimension / 2, 0,                  0,                  0,
		0,                  grid_dimension / 2, 0,                  0,
		0,                  0,                  grid_dimension / 2, 0,
		0,                  0,                  0,                  1 );

	createBoxShader(shader_box);
	glUseProgram(shader_box);
	shader_box_proj_view = glGetUniformLocation(shader_box, "u_projection_view");

//------------------------------------------

	Shader shader_surface;
	GLuint surf_proj_view;
	GLuint surf_light_direc;

	createSurfaceShader(shader_surface);
	glUseProgram(shader_box);
	surf_proj_view = glGetUniformLocation(shader_surface, "u_projection_view");
	surf_light_direc = glGetUniformLocation(shader_surface, "u_light_direction");

//------------------------------------------

	Shader shader_flat;
	GLuint shader_proj_view = 0;
	GLuint shader_light_direc = 0;

	createFlatShader(shader_flat);
	glUseProgram(shader_flat);
	shader_proj_view = glGetUniformLocation(shader_flat, "u_projection_view");
	shader_light_direc = glGetUniformLocation(shader_flat, "u_light_direction");

	GLuint modelview_index = glGetUniformBlockIndex(shader_flat, "ModelView");   
	glUniformBlockBinding(shader_flat, modelview_index, 0);

//------------------------------------------

	if ( glGetError() != GL_NO_ERROR ) 
		throw "Got OpenGL Error during Setup!";

//--------------------------------------------------------------------------------------------------
// MAIN LOOP
//--------------------------------------------------------------------------------------------------
	double _current_time = glfwGetTime();
	double _delta_time = 0.016;
	double _fps = 0.0f;
	unsigned int _frame_count = 1;
	bool _draw_particles = true;

	while (!glfwWindowShouldClose(mainWindow))
	{
		if ( GlobalData::spacePressed )
			_draw_particles = !_draw_particles;
		if ( GlobalData::escapePressed )
			glfwSetWindowShouldClose(mainWindow, GLFW_TRUE);

	//----------------------------------------------------------------------------------------------
	// CUDA Segment
	//----------------------------------------------------------------------------------------------
		
		runCUDASimulation(sim, cubes, min(0.016, _delta_time), _frame_count);

		std::cerr << "face count " << cubes.getFaceCount() << std::endl;

	//----------------------------------------------------------------------------------------------
	// OpenGL Segment
	//----------------------------------------------------------------------------------------------
		{
			GLint draw_iterations = 4 * sim.getParticleCount() / max_uniform_buffer_range;
			GLint draw_remainder = sim.getParticleCount() - draw_iterations * 4 * max_uniform_buffer_range;


			int width, height;
			glfwGetFramebufferSize(mainWindow, &width, &height);

			if ( GlobalData::rightMousePressed ) {
				main_camera.angle -= 0.8f * deg2rad * (float)GlobalData::xDeltaPosition;
				main_camera.rise = (float)std::min( 80.0 * deg2rad, 
										  std::max( -80.0 * deg2rad, 
										  (double)main_camera.rise - 0.3 * deg2rad * GlobalData::yDeltaPosition ) );
			}
			if ( GlobalData::yScroll < 0 )
				main_camera.arm_length = std::min(1024.0f, main_camera.arm_length * (float)std::pow(1.1f, -GlobalData::yScroll));
			else if ( GlobalData::yScroll > 0 )
				main_camera.arm_length = std::max(2.0f, main_camera.arm_length * (float)std::pow(0.9f, GlobalData::yScroll));

			glViewport(0, 0, width, height);
			glClearColor(0,0,0,1);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

			// --- DRAW BOX ---------------------------------------------------
			glUseProgram(shader_box);

			{
				core::mat4 data;
				main_camera.fillMatrix((float)width/height, data);
				data = data * scaling_matrix;
				glUniformMatrix4fv(shader_box_proj_view, 1, false, (float*)&data);
			}

			glBindBuffer(GL_ARRAY_BUFFER, draw_cube_buffer);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

			glDrawArrays(GL_LINES, 0, sizeof(data) / (2*sizeof(float)));

			// --- DRAW PARTICLES ---------------------------------------------
			if ( _draw_particles ) {

				glUseProgram(shader_flat);

				{
					core::mat4 data;
					main_camera.fillMatrix((float)width/height, data);
					glUniformMatrix4fv(shader_proj_view, 1, false, (float*)&data);
				}

				glUniform3f(shader_light_direc, 0,1,0);

				glBindBuffer(GL_ARRAY_BUFFER, array_buffer);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

				glEnableVertexAttribArray(0);
				glEnableVertexAttribArray(1);
				glEnableVertexAttribArray(2);
				glEnableVertexAttribArray(3);

				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 0));
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 3));
				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 6));
				glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, (void*)(std::size_t)(draw_vertex_count * sizeof(float) * 8));

				for ( int iteration=0; iteration<draw_iterations; iteration++ ) {
					sim.getUniformBuffer().bindGL(0, iteration * max_uniform_buffer_range, max_uniform_buffer_range);
					glDrawElementsInstanced(GL_TRIANGLES, 3 * draw_face_count, GL_UNSIGNED_BYTE, 0, max_uniform_buffer_units);
				}
				if ( draw_remainder > 0 ) {
					sim.getUniformBuffer().bindGL(0, draw_iterations * max_uniform_buffer_range, draw_remainder);
					glDrawElementsInstanced(GL_TRIANGLES, 3 * draw_face_count, GL_UNSIGNED_BYTE, 0, draw_remainder);	
				}

				glDisableVertexAttribArray(0);
				glDisableVertexAttribArray(1);
				glDisableVertexAttribArray(2);
				glDisableVertexAttribArray(3);

			} else {

				glUseProgram(shader_surface);
				{
					core::mat4 data;
					main_camera.fillMatrix((float)width/height, data);
					glUniformMatrix4fv(surf_proj_view, 1, false, (float*)&data);
				}

				glUniform3f(surf_light_direc, 0,1,0);

				cubes.bindGL();
				glDrawElements(GL_TRIANGLES, cubes.getFaceCount(), GL_UNSIGNED_INT, 0);

			}

			glFinish();
		}

	//----------------------------------------------------------------------------------------------
	// Finish Frame
	//----------------------------------------------------------------------------------------------
		{
			glfwSwapBuffers(mainWindow);
			GlobalData::UpdateGlobalData();
			glfwPollEvents();

			double ct = glfwGetTime();
			
			_delta_time = ct - _current_time;
			_current_time = ct;
			_fps = 0.9 * _fps + 0.1 * (1.0 / _delta_time);

			//Sleep(1000);
		}

		_frame_count ++ ;

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

void createBoxShader(Shader& shad) {
	shad.vertex =
		"#version 330\r\n"
		"uniform mat4 u_projection_view;"
		"in vec3 in_position;"
		"out vec3 v_position;"
		"void main() {"
		"   gl_Position = u_projection_view * vec4(in_position, 1);"
		"	gl_PointSize = 8.0;"
		"	v_position = in_position;"
		"}";
		
	shad.fragment =
		"#version 330\r\n"
		"in vec3 v_position;"
		"out vec4 out_color;"
		"void main() {"
		"	out_color = vec4(v_position,1);"
		"}";

	shad.link();
}
void createSurfaceShader(Shader& shad) {
	shad.vertex =
		"#version 330\r\n"
		"uniform mat4 u_projection_view;"
		"in vec3 in_position;"
		"in vec3 in_normal;"
		"out vec3 v_normal;"
		"void main() {"
		"   gl_Position = u_projection_view * vec4(in_position, 1);"
		"   gl_PointSize = 4.0;"
		"	v_normal = in_normal;"
		"}";
		
	shad.fragment =
		"#version 330\r\n"
		"uniform vec3 u_light_direction;"
		"in vec3 v_normal;"
		"out vec4 out_color;"
		"void main() {"
		"	float lighting = 0.8 * clamp(dot(u_light_direction, v_normal) + 0.2, 0, 1) + 0.2;"
		"	out_color = vec4(vec3(0.25,0.640625,0.87109375)*lighting,1);"
		//"	out_color = vec4(1,1,1,1);"
		"}";

	shad.link();
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