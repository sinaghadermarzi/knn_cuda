#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <limits>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <float.h>
using namespace std;


#define KNN_K 4

__device__ double  dist(double* data ,int i,int j , int num_dim){
	double dis = 0;
	for (int attr_idx  = 0; attr_idx<num_dim; attr_idx++){
		dis += (*(data+i*num_dim+attr_idx)-*(data+j*num_dim+attr_idx))*(*(data+i*num_dim+attr_idx)-*(data+j*num_dim+attr_idx));
	}
	return sqrt(dis);
}







//double dist(ArffData* dataset,int i,int j)
//{
//	return (i-j)*(i-j);
//}


__global__ void KNN_prediction(double* arrayData,int* predictions,int num_instances,int num_classes,int num_attributes)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double distances[8192];
	int votes[16];
	if (i<num_instances)
	{
		for (int j = 0; j < num_instances ; j++)
		{
			if (j != i)
			{
				distances[j] = dist(arrayData,i,j, num_attributes);
			}
			else
			{
				distances[j] = DBL_MAX;
			}
		}

//		int* votes = (int*) malloc(num_classes * sizeof(int));

		for(int j =0; j < num_classes ; j++)
			votes [j] =0;

		for(int j =0; j < KNN_K ; j++)
		{
			int min_cl = 0;//find the class of the jth smallest elment in distances array
			int min_idx=0;
			double min_dist = DBL_MAX;
			for (int k =0;k<num_instances;k++)
			{
				if ((distances[k]<min_dist)&&(k!=i))
				{
					min_dist = distances[k];
					distances[k] = DBL_MAX;
					min_idx = k;
					min_cl =*(arrayData+k*num_attributes+(num_attributes-1));//dataset->get_instance(k)->get(dataset->num_attributes() - 1)->operator int32();
				}
			}
	//    		cout<<"min_cl:"<<min_cl<<std::endl;
	//    		cout<<"min_dist:"<<min_dist<<std::endl;
	//    		cout<<"min_idx:"<<min_idx<<std::endl;
			votes[min_cl]++;
		}
		//find the index of the largest element in votes->prediction
		int majority_class = 0;
		int max_votes =0;
	//    	int max_votes_idx = 0;

	//    	std::cout<<"votes:";
	//    	for(int k = 0; k<dataset->num_classes();k++)
	//    	{
	//    		std::cout<<votes[k]<<",";
	//
	//    	}
	//    	int hgk = 2;
	//    	std::cin>>hgk;



		for (int j = 0;j<num_classes;j++)
		{
			if (votes[j]>max_votes)
			{
				max_votes = votes[j];
				majority_class=j;
			}
		}
		predictions[i] = majority_class;
	}

}

int* KNN(ArffData* dataset)
{
	int num_instances = dataset->num_instances();
	int num_classes = dataset->num_classes();
	int num_attributes  = dataset->num_attributes();

    int* h_predictions = (int*)malloc(num_instances * sizeof(int));
    double * h_arrayData = (double*) malloc(num_instances*num_attributes * sizeof(double));

    for (int i = 0; i <num_instances; i++)
    {
//    	printf("__________%d\n",i);
    	for (int j = 0; j <num_attributes; j++)
    	{
 //(double) (dataset->get_instance(i)->get(j)->operator float());
//        	printf("%d\n",j);
        	*(h_arrayData+i*num_attributes+j) = (double) (dataset->get_instance(i)->get(j)->operator float());
    	}
    }
    printf("data successfully put into array form\n");
    int* d_predictions;
    double * d_arrayData;
    cudaMalloc(&d_predictions,num_instances * sizeof(int));
    cudaMalloc(&d_arrayData,num_instances*num_attributes * sizeof(double));
    printf("memory allocated on device\n");
    cudaMemcpy(d_arrayData, h_arrayData, num_instances*num_attributes * sizeof(double), cudaMemcpyHostToDevice);
    printf("data copied to device\n");
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_instances + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    KNN_prediction<<<blocksPerGrid, threadsPerBlock>>>(d_arrayData,d_predictions,num_instances,num_classes,num_attributes);
    printf("kernel launched\n");

    cudaMemcpy(h_predictions,d_predictions, num_instances * sizeof(int), cudaMemcpyDeviceToHost);
    printf("result copied back to memory\n");
    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_predictions);
    cudaFree(d_arrayData);

    // Free host memory
    free(h_arrayData);
    for (int i = 0 ; i<num_instances;i++)
    	printf("%d,",h_predictions[i]);

    return h_predictions;

}
int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagnoal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }
//    const char* fileaddr = "datasets/small.arff";

    ArffParser parser(argv[1]);
//    ArffParser parser(fileaddr);
    ArffData *dataset = parser.parse();
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int* predictions = KNN(dataset);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}




