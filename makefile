OUTPUT_NAME = fluid_sim
CC = nvcc

SRCS := $(shell find ./src/*/ \( -name '*.cpp' -o -name '*.cu' \))
LIBS := ./lib/libGLEW.so ./lib/libGLFW.so
WIN_LIBS := -llib/glew32 -llib/glfw3dll -lopengl32

build_run: build
	./bin/$(OUTPUT_NAME)

build:
	$(CC) $(SRCS) $(LIBS) -I./include -lGL -o ./bin/$(OUTPUT_NAME)

windows_run: windows
	./bin/$(OUTPUT_NAME).exe

windows:
	$(CC) $(SRCS) -arch=compute_50 -I./include $(WIN_LIBS) -o ./bin/$(OUTPUT_NAME).exe