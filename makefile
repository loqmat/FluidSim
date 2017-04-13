build:
	nvcc ./src/kernel.cu ./lib/libGLEW.so -arch=compute_61 -I./include -lGL -o ./bin/fluid_sim

windows:
	nvcc ./src/kernel.cu -arch=compute_61 -I./include -llib/glew32 -llib/glfw3dll -lopengl32 -o ./bin/fluid_sim.exe