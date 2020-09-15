//#include </usr/include/opencv4/opencv2/highgui/highgui.hpp>
//#include </usr/include/opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "mpi.h"
#include <math.h>



using namespace std;
using namespace cv;

Mat getGaussian(int height, int width, double sigma){
    
    Mat kernel;
    kernel.create(height, width,cv::DataType<double>::type);

    double sum=0.0;
    int i,j;

    for (i=0 ; i<height ; i++) {
        for (j=0 ; j<width ; j++) {
            kernel.at<double>(i,j) = exp(-(i*i+j*j)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);    
            sum += kernel.at<double>(i,j);
        }
    }

    for (i=0 ; i<height ; i++) {
        for (j=0 ; j<width ; j++) {
            kernel.at<double>(i,j) /= sum;
        }
    }

    return kernel;
}

Mat applyFilter(Mat &image, Mat &filter){
    
    int height = image.rows;
    int width = image.cols;
    int filterHeight = filter.rows;
    int filterWidth = filter.cols;
    int newImageHeight = height-filterHeight+1;
    int newImageWidth = width-filterWidth+1;
    int i,j,h,w;

    Mat newImage(newImageHeight, newImageWidth, CV_8UC1);
 

    for (i=0 ; i<newImageHeight ; i++) {
        for (j=0 ; j<newImageWidth ; j++) {
            for (h=i ; h<i+filterHeight ; h++) {
                for (w=j ; w<j+filterWidth ; w++) {
                    
                    newImage.at<uchar>(i,j) += filter.at<double>(h-i, w-j) * image.at<uchar>(h,w);
                }
            }
        }
    }

    return newImage;
}
int main(int argc, char *argv[]){


    Mat imagen = imread(argv[1], IMREAD_GRAYSCALE);
    cout << "La imagen mide " << imagen.rows << " x " << imagen.cols << " pixeles" << endl;
    
    Mat filter = getGaussian(5, 5, 5.0);
    Mat result;
    
    //Show filter
    //cout << filter << endl;

    double startTime = MPI_Wtime();
    result = applyFilter(imagen, filter);
    double totalTime = MPI_Wtime() - startTime;


    imwrite(argv[2], result);
    cout << "Time: " << totalTime << "segundos" << endl;    
    return 0;
}