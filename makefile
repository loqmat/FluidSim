build:
	nvcc ./src/kernel.cu -arch=compute_61 -I./include -llib/glew32 -llib/glfw3dll -lopengl32 -o ./bin/fluid_sim.exe