build:
	gcc -g -Wall -O3 -o main main.c `pkg-config vips --cflags --libs` -lOpenCL
