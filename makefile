build:
	nvcc ./src/kernel.cu -Iinclude -llib/glew32 -llib/glfw3dll -lopengl32 -o ./bin/fluid_sim.exe