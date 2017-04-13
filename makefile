OUTPUT_NAME = fluid_sim
CC = nvcc

SRCS := ./src/kernel.cu
LIBS := ./lib/libGLEW.so ./lib/libGLFW.so

build:
	$(CC) $(SRCS) $(LIBS) -arch=compute_61 -I./include -lGL -o ./bin/$(OUTPUT_NAME)

windows:
	$(CC) $(SRCS) -arch=compute_61 -I./include -llib/glew32 -llib/glfw3dll -lopengl32 -o ./bin/$(OUTPUT_NAME).exe
