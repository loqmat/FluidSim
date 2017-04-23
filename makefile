OUTPUT_NAME = fluid_sim
CC = nvcc

SRCS := $(shell find ./src/ \( -name '*.cpp' -o -name '*.cu' \))
LIBS := ./lib/libGLEW.so ./lib/libGLFW.so
WIN_LIBS := -llib/glew32 -llib/glfw3dll -lopengl32
OPTIONS := -Wfatal-errors

build:
	$(CC) $(OPTIONS) $(SRCS) $(LIBS) -arch=compute_61 -I./include -lGL -o ./bin/$(OUTPUT_NAME)

windows_run: windows
	./bin/$(OUTPUT_NAME).exe

windows:
	$(CC) $(OPTIONS) $(SRCS) -arch=compute_50 -I./include $(WIN_LIBS) -o ./bin/$(OUTPUT_NAME).exe