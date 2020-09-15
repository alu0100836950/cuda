#include <iostream>
#include <cstdio>
#include <memory>
#include <opencv2/opencv.hpp>
#include <chrono>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
#include "omp.h"
#include <math.h>


using namespace std;
using namespace cv;



__global__ void applyFilter(uchar* imagen, uchar* result, size_t rows, size_t cols, uchar* kernel){

    int filterHeight = 5;
    int filterWidth = 5;

    int j,h,w;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < rows){  
        for (j=0 ; j<cols ; j++) {
            float sum = 0, val = 0;
            for (h=i ; h<i+filterHeight ; h++) {
                for (w=j ; w<j+filterWidth ; w++) {
                    float weight = kernel[(h - i) * filterWidth + (w-j)];
                    val += imagen[h * cols + w] * weight;
                    sum+= weight;

                    //result[i * cols + j] += kernel[((h-i) * filterWidth) + (w-j)] * imagen[h * filterWidth + w];
                }
                result[i * cols + j] = round(val/sum);
            }
        }
    }
}

int main(int argc, char *argv[]){

    Mat *kernel = new Mat(5, 5, CV_8UC1);

    double sigma= 5.0;
    double sum=0.0;

    int filterHeight = kernel->rows;
    int filterWidth = kernel->cols;
    

    for (int i=0 ; i<filterHeight ; i++) {
        for (int j=0 ; j<filterWidth ; j++) {
            kernel->at<double>(i,j) = exp(-(i*i+j*j)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);    
            sum += kernel->at<double>(i,j);
        }
    }

    for (int i=0 ; i<filterHeight ; i++) {
        for (int j=0 ; j<filterWidth ; j++) {
            kernel->at<double>(i,j) /= sum;
        }
    }    
    
    //Mat* imagen = new cv::Mat(cv::imread(argv[1], cv::IMREAD_GRAYSCALE));
    //Mat* resultCPU = new cv::Mat(imagen->rows, imagen->cols, CV_8UC1);
    //cout << "La imagen mide " << imagen->rows << " x " << imagen->cols << " pixeles" << endl;
   
   
    Mat *imagen = new Mat(imread(argv[1], IMREAD_GRAYSCALE));
    
    
    cout << "La imagen mide " << imagen->rows << " x " << imagen->cols << " pixeles" << endl;
    Mat *resultCPU = new Mat(imagen->rows, imagen->cols, CV_8UC1);
    
    
    uchar * src;
    uchar * result;
    uchar * kernel_;

    cudaDeviceProp  prop;

    cudaGetDeviceProperties(&prop, 0);
    printf( "Usando dispositivo CUDA:  %s\n", prop.name );

    cudaMalloc(&src, imagen->total() * sizeof(uchar));
    cudaMalloc(&result, imagen->total() * sizeof(uchar));
    cudaMalloc(&kernel_, kernel->total() * sizeof(uchar));

    
    int num_B = 16;
    dim3 threadsPerBlock(imagen->rows / num_B);
 
    cout << "Ejecutamos el filtro" << endl;
    auto t1 = chrono::high_resolution_clock::now();
    //double startTime = omp_get_wtime();
    
    cudaMemcpy(src, imagen->data, imagen->total() * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_, kernel->data, kernel->total() * sizeof(uchar), cudaMemcpyHostToDevice);
    
    applyFilter<<<num_B, threadsPerBlock>>>(src, result, imagen->rows, imagen->cols, kernel_);

    cudaMemcpy(resultCPU->data, result, imagen->total(), cudaMemcpyDeviceToHost);

    auto t2 = chrono::high_resolution_clock::now();
   //double totalTime = omp_get_wtime() - startTime;

    auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "Tiempo de ejecucion: " << (float) (duration / 1000.0) << " sec" << endl;

    imwrite(argv[2], *resultCPU);
    //cout << "Time: " << totalTime << "segundos" << endl;  
  

    cudaFree(src);
    cudaFree(result);
    cudaFree(kernel_);
    return 0;
}