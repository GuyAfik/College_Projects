
#include "main.h"



__device__ int getFSignCuda(double fVal)
{
	if (fVal >= 0)
		return 1;
	else
		return -1;
}

__global__ void calcSignValue(double* w, int* cudaPointDimentionSize, 
	int* task_per_proc, double* locationPoints, int* signs) //each thread here calculate the fValue and sign for 1 point
{
	
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= *task_per_proc)
		return;
	double fValue = w[*cudaPointDimentionSize];
	for (int i = 0; i < *cudaPointDimentionSize; i++)
	{
		fValue += w[i] * locationPoints[id * (*cudaPointDimentionSize) + i];
	}
	signs[id] = getFSignCuda(fValue);
}



__global__ void calcNewPositions(int* cudaPointDimentionSize, double* t, double* pointsLocations, 
	double* initialPointsLocations, double* pointsVelocities, int* task_per_proc) //each thread here calculates the new position for 1 point
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= *task_per_proc)
		return;

	for (int i = 0; i < *cudaPointDimentionSize; i++)
	{
		pointsLocations[id *(*cudaPointDimentionSize) + i] = initialPointsLocations[id *(*cudaPointDimentionSize) + i]
			+ pointsVelocities[id *(*cudaPointDimentionSize) + i] * (*t);
	}
}



void positionCalculationWithCuda(double* device_points_locations, double* device_points_initial_locations, 
	double* device_points_velocities,int* device_dimention_size, int* device_task_per_proc
	,double* device_time,double* cpu_points_locations,double* cpu_time, int task_per_proc, 
	int K, int numThreads, int numBlocks, int extraBlock)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMemcpy(device_time, cpu_time, sizeof(double), cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_time failed!", device_time);

	//kernel method to calculate the new positions for all points that this specific process is responsible for
	calcNewPositions <<<numBlocks + extraBlock, numThreads>>> (device_dimention_size, device_time, 
		device_points_locations, device_points_initial_locations, device_points_velocities, device_task_per_proc);

	cudaStatus = cudaMemcpy(cpu_points_locations, device_points_locations,
		sizeof(double) * K * task_per_proc, cudaMemcpyDeviceToHost);
	checkErrors(cudaStatus, "cudaMemcpy of device_points_locations failed!", device_points_locations);

}



void fAndSignCalculationWithCuda(int* device_dimention_size, int K, double* device_w,
	int* device_task_per_proc, int* device_signs,  double* device_points_locations, int task_per_proc,
	double* cpu_points_locations, int* cpu_signs, double* cpu_w, int numThreads, int numBlocks, int extraBlock)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	
	cudaStatus = cudaMemcpy(device_w, cpu_w, sizeof(double) * (K + 1), cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_w failed!", device_w);

	cudaStatus = cudaMemcpy(device_points_locations, cpu_points_locations, 
		sizeof(double) * K * task_per_proc, cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_points_locations failed!", device_points_locations);
	
	//kernel method to calculate the new sign values for all points that this specific process is responsible about
	calcSignValue << <numBlocks + extraBlock, numThreads >> > (device_w, device_dimention_size, device_task_per_proc, 
		device_points_locations, device_signs);

	cudaStatus = cudaMemcpy(cpu_signs, device_signs, sizeof(int) * task_per_proc, cudaMemcpyDeviceToHost);
	checkErrors(cudaStatus, "cudaMemcpy of device_signs failed!", device_signs);

}


double* allocateDevicePointsLocation(int task_per_proc, int K)
{
	cudaError_t cudaStatus;
	double* device_points_locations;
	cudaStatus = cudaMalloc((void**) &(device_points_locations), sizeof(double)*task_per_proc*K);
	checkErrors(cudaStatus, "cudaMalloc of device_points_locations failed!", device_points_locations);
	return device_points_locations;
}

double* allocateDeviceWArr(int K)
{
	cudaError_t cudaStatus;
	double* device_w;
	cudaStatus = cudaMalloc((void**) &(device_w), sizeof(double)*(K+1));
	checkErrors(cudaStatus, "cudaMalloc of device_w failed!", device_w);
	return device_w;
}

double* allocateAndCopyDeviceInitialPointsPositions(double* cpu_initial_locations,int task_per_proc, int K)
{
	cudaError_t cudaStatus;
	double* device_initial_locations;
	cudaStatus = cudaMalloc((void**) &(device_initial_locations), sizeof(double)*K*task_per_proc);
	checkErrors(cudaStatus, "cudaMalloc of device_initial_locations failed!", device_initial_locations);
	cudaStatus = cudaMemcpy(device_initial_locations, cpu_initial_locations, 
		sizeof(double) * K * task_per_proc, cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_initial_locations failed!", device_initial_locations);
	return device_initial_locations;
}

double* allocateAndCopyDevicePointsVelocities(double* cpu_points_velocities, int task_per_proc, int K)
{
	cudaError_t cudaStatus;
	double* device_points_velocities;
	cudaStatus = cudaMalloc((void**) &(device_points_velocities), sizeof(double)*K*task_per_proc);
	checkErrors(cudaStatus, "cudaMalloc of device_points_velocities failed!", device_points_velocities);
	cudaStatus = cudaMemcpy(device_points_velocities, cpu_points_velocities,
		sizeof(double) * K * task_per_proc, cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_points_velocities failed!", device_points_velocities);
	return device_points_velocities;
}


int* allocateAndCopyDeviceDimentionSize(int* cpu_dimention_size)
{
	cudaError_t cudaStatus;
	int* device_dimention_size;
	cudaStatus = cudaMalloc((void**) &(device_dimention_size), sizeof(int));
	checkErrors(cudaStatus, "cudaMalloc of device_dimention_size failed!", device_dimention_size);
	cudaStatus = cudaMemcpy(device_dimention_size, cpu_dimention_size,
		sizeof(int), cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_dimention_size failed!", device_dimention_size);
	return device_dimention_size;
}


double* allocateDeviceTime()
{
	cudaError_t cudaStatus;
	double* device_time;
	cudaStatus = cudaMalloc((void**) &(device_time), sizeof(double));
	checkErrors(cudaStatus, "cudaMalloc of device_time failed!", device_time);
	return device_time;
}


int* allocateDeviceSigns(int task_per_proc)
{
	cudaError_t cudaStatus;
	int* device_signs;
	cudaStatus = cudaMalloc((void**) &(device_signs), sizeof(int)*task_per_proc);
	checkErrors(cudaStatus, "cudaMalloc of device_signs failed!", device_signs);
	return device_signs;
}

int *allocateAndCopyDeviceAmountOfTask(int* cpu_task_per_proc)
{
	cudaError_t cudaStatus;
	int* device_task_per_proc;
	cudaStatus = cudaMalloc((void**) &(device_task_per_proc), sizeof(int));
	checkErrors(cudaStatus, "cudaMalloc of device_task_per_proc failed!", device_task_per_proc);
	cudaStatus = cudaMemcpy(device_task_per_proc, cpu_task_per_proc,
		sizeof(int), cudaMemcpyHostToDevice);
	checkErrors(cudaStatus, "cudaMemcpy of device_task_per_proc failed!", device_task_per_proc);
	return device_task_per_proc;
}


void freeAllCudaVariables(int* device_task_per_proc, int* device_signs, double* device_time, int* device_dimention_size,
	double* device_points_velocities, double* device_initial_locations, double* device_w, double* device_points_locations)
{
	cudaFree(device_task_per_proc);
	cudaFree(device_signs);
	cudaFree(device_time);
	cudaFree(device_dimention_size);
	cudaFree(device_points_velocities);
	cudaFree(device_initial_locations);
	cudaFree(device_w);
	cudaFree(device_points_locations);
}

void checkErrors(cudaError_t cudaStatus, const char* errorMessage, void* arr)
{
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		cudaFree(arr);
		
	}
}