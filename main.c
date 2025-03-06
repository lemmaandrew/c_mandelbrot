#include <CL/cl.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vips/vips.h>

const char HELPTEXT[] =
    "-r    real center (default -0.8)\n"
    "-i    imaginary center (default 0.0)\n"
    "-x    width (default 100)\n"
    "-y    height (default 100)\n"
    "-s    stepsize (default 0.05)\n"
    "-m    max iterations (default 20)\n"
    "-b    bailout (default 1e6)\n"
    "-o    image output path (default test.png)\n"
    "-g    use gpu instead of serial (default serial)\n";

// i should probably create a .cl file or something for this
// but whatever

// kernel source for gpu_main.
// includes a duplicate of `derbail`
const char mandelbrot_kernel_source[] =
    "\n"
    "unsigned int derbail(double real0, double imag0, unsigned int max_iters,\n"
    "                     double bailout) {\n"
    "    double real = real0;\n"
    "    double imag = imag0;\n"
    "\n"
    "    double dreal = 1.0;\n"
    "    double dimag = 0.0;\n"
    "    double dreal_sum = 0.0;\n"
    "    double dimag_sum = 0.0;\n"
    "\n"
    "    for (unsigned int n = 1; n < max_iters; ++n) {\n"
    "        double x = real * real - imag * imag;\n"
    "        imag = 2 * real * imag + imag0;\n"
    "        real = x + real0;\n"
    "\n"
    "        double dx = 2 * (dreal * real - dimag * imag) + 1;\n"
    "        double dy = 2 * (dreal * imag + dimag * real);\n"
    "        dreal = dx;\n"
    "        dimag = dy;\n"
    "\n"
    "        dreal_sum += dreal;\n"
    "        dimag_sum += dimag;\n"
    "\n"
    "        if (dreal_sum * dreal_sum + dimag_sum * dimag_sum >= bailout) {\n"
    "            return n;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    return 0;\n"
    "}\n"
    "__kernel void mb(\n"
    "    __global unsigned char *out,\n"
    "    const unsigned int width,\n"
    "    const double real_start,\n"
    "    const double imag_start,\n"
    "    const double stepsize,\n"
    "    const unsigned int max_iters,\n"
    "    const double bailout\n"
    ") {\n"
    "    unsigned int i = get_global_id(0);\n"
    "    unsigned int w = i % width;\n"
    "    unsigned int h = i / width;\n"
    "    double real = real_start + w * stepsize;\n"
    "    double imag = imag_start - h * stepsize;\n"
    "    unsigned int iters = derbail(real, imag, max_iters, bailout);\n"
    "    if (iters > UCHAR_MAX) {\n"
    "        out[h * width + w] = UCHAR_MAX;\n"
    "    } else {\n"
    "        out[h * width + w] = iters;\n"
    "    }\n"
    "}\n";

// Derivative bailout procedure for getting the iterations of a point on the
// mandelbrot set
unsigned int derbail(double real0, double imag0, unsigned int max_iters,
                     double bailout) {
    double real = real0;
    double imag = imag0;

    double dreal = 1.0;
    double dimag = 0.0;
    double dreal_sum = 0.0;
    double dimag_sum = 0.0;

    for (unsigned int n = 1; n < max_iters; ++n) {
        // (x + yi)^2 = x * x + 2ixy + -(y * y)
        double x = real * real - imag * imag;
        imag = 2 * real * imag + imag0;
        real = x + real0;

        // (a + bi) * (c + di) = ac + adi + bci - bd
        //                     = (ac - bd) + (adi + bci)
        double dx = 2 * (dreal * real - dimag * imag) + 1;
        double dy = 2 * (dreal * imag + dimag * real);
        dreal = dx;
        dimag = dy;

        dreal_sum += dreal;
        dimag_sum += dimag;

        if (dreal_sum * dreal_sum + dimag_sum * dimag_sum >= bailout) {
            return n;
        }
    }

    return 0;
}

// Serially assigning iterations on an image array
void mandelbrot(unsigned char *out, double real_center, double imag_center,
                unsigned int width, unsigned int height, double stepsize,
                unsigned int max_iters, double bailout) {
    double real_start = real_center - width * stepsize / 2;
    double imag_start = imag_center + height * stepsize / 2;

    for (unsigned int h = 0; h < height; ++h) {
        double imag = imag_start - h * stepsize;
        for (unsigned int w = 0; w < width; ++w) {
            double real = real_start + w * stepsize;
            unsigned int iters = derbail(real, imag, max_iters, bailout);
            if (iters > UCHAR_MAX) {
                out[h * width + w] = UCHAR_MAX;
            } else {
                out[h * width + w] = iters;
            }
        }
    }
}

// Saving the created mandelbrot image to `output` filepath
void save_image(unsigned char *mem, unsigned int width, unsigned int height,
                const char output[256]) {
    VipsImage *image;
    image =
        vips_image_new_from_memory(mem, sizeof(unsigned char) * width * height,
                                   width, height, 1, VIPS_FORMAT_UCHAR);
    image->Type = VIPS_INTERPRETATION_B_W;
    vips_image_write_to_file(image, output, NULL);
    g_object_unref(image);
}

// Looooooooooooooong function for parallel-y creating and writing a mandelbrot
// image
int gpu_main(double real_center, double imag_center, unsigned int width,
             unsigned int height, double stepsize, unsigned int max_iters,
             double bailout, const char output[256]) {
    double real_start = real_center - width * stepsize / 2;
    double imag_start = imag_center + height * stepsize / 2;

    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) {
        printf("Zero platforms! (error code: %d)\n", err);
        return EXIT_FAILURE;
    }

    cl_platform_id platforms[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id device_id;
    for (int i = 0; i < num_platforms; ++i) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1,
                             &device_id, NULL);
        if (err == CL_SUCCESS) {
            break;
        }
    }
    if (device_id == NULL) {
        printf("Failed to acquire device_id (error code: %d)\n", err);
        return EXIT_FAILURE;
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create context (error code: %d)\n", err);
        return EXIT_FAILURE;
    }
    cl_command_queue commands =
        clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create commands (error code: %d)\n", err);
        return EXIT_FAILURE;
    }
    cl_program program = clCreateProgramWithSource(
        context, 1, (const char **)&mandelbrot_kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create program (error code: %d)\n", err);
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to build program (error code: %d)\n", err);

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    cl_kernel mandelbrot_kernel = clCreateKernel(program, "mb", &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel (error code: %d)\n", err);
        return EXIT_FAILURE;
    }

    size_t data_size = width * height * sizeof(unsigned char);
    cl_mem cl_out =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create buffer (error code: %d)\n", err);
        return EXIT_FAILURE;
    }

    err = clSetKernelArg(mandelbrot_kernel, 0, sizeof(cl_mem), &cl_out);
    err |= clSetKernelArg(mandelbrot_kernel, 1, sizeof(unsigned int), &width);
    err |= clSetKernelArg(mandelbrot_kernel, 2, sizeof(double), &real_start);
    err |= clSetKernelArg(mandelbrot_kernel, 3, sizeof(double), &imag_start);
    err |= clSetKernelArg(mandelbrot_kernel, 4, sizeof(double), &stepsize);
    err |=
        clSetKernelArg(mandelbrot_kernel, 5, sizeof(unsigned int), &max_iters);
    err |= clSetKernelArg(mandelbrot_kernel, 6, sizeof(double), &bailout);

    size_t global = width * height;
    err = clEnqueueNDRangeKernel(commands, mandelbrot_kernel, 1, NULL, &global,
                                 NULL, 0, NULL, NULL);

    unsigned char *out = (unsigned char *)malloc(data_size);
    err = clEnqueueReadBuffer(commands, cl_out, CL_TRUE, 0, data_size, out, 0,
                              NULL, NULL);

    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    clReleaseDevice(device_id);
    clReleaseProgram(program);

    save_image(out, width, height, output);
    free(out);

    return 0;
}

int main(int argc, char *argv[]) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit(NULL);
    }

    double real_center = -0.8;
    double imag_center = 0.0;
    unsigned int width = 100;
    unsigned int height = 100;
    double stepsize = 0.05;
    unsigned int max_iters = 20;
    double bailout = 1e6;
    char output[256] = "test.png";
    bool gpu = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf(HELPTEXT);
            return EXIT_SUCCESS;
        } else if (!strcmp(argv[i], "-r")) {
            ++i;
            real_center = atof(argv[i]);
        } else if (!strcmp(argv[i], "-i")) {
            ++i;
            imag_center = atof(argv[i]);
        } else if (!strcmp(argv[i], "-x")) {
            ++i;
            width = atoi(argv[i]);
        } else if (!strcmp(argv[i], "-y")) {
            ++i;
            height = atoi(argv[i]);
        } else if (!strcmp(argv[i], "-s")) {
            ++i;
            stepsize = atof(argv[i]);
        } else if (!strcmp(argv[i], "-m")) {
            ++i;
            max_iters = atoi(argv[i]);
        } else if (!strcmp(argv[i], "-b")) {
            ++i;
            bailout = atof(argv[i]);
        } else if (!strcmp(argv[i], "-o")) {
            ++i;
            strcpy(output, argv[i]);
        } else if (!strcmp(argv[i], "-g")) {
            gpu = true;
        }
    }

    if (gpu) {
        return gpu_main(real_center, imag_center, width, height, stepsize,
                        max_iters, bailout, output);
    }

    unsigned char *mem =
        (unsigned char *)malloc(sizeof(unsigned char) * width * height);

    mandelbrot(mem, real_center, imag_center, width, height, stepsize,
               max_iters, bailout);

    save_image(mem, width, height, output);
    free(mem);
}
