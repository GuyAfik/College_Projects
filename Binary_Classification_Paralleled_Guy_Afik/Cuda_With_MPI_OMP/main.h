#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>


#define INPUT_FILE "C:\\data4.txt"
#define OUTPUT_FILE "C:\\output.txt"
#define ROOT 0
#define DATA_TAG 300


void parellelBinaryClassification(int argc, char* argv[]);
void copyInitialPointsLocation(double* currentPointLocation, double* initialPointLocation, int K, int N);
void setWZeros(double* w, int K);
double f(double* w, double* pointValues, int K);
extern void fAndSignCalculationWithCuda(int* device_dimention_size, int K, double* device_w,
	int* device_task_per_proc, int* device_signs, double* device_points_locations, int task_per_proc,
	double* cpu_points_locations, int* cpu_signs, double* cpu_w, int numThreads, int numBlocks, int extraBlock);
int getFSign(double fValue);
void updateWValues(double* w, double* pointValues, int sign, int K, double alpha);
int getQuailtyWithOpenMP(int* signs, int* sets, int task_per_proc);
extern void positionCalculationWithCuda(double* device_points_locations, double* device_points_initial_locations,
	double* device_points_velocities, int* device_dimention_size, int* device_task_per_proc
	,double* device_time, double* cpu_points_locations, double* cpu_time, int task_per_proc, int K,
	int numThreads, int numBlocks, int extraBlock);
void readDataFromFile(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsCurrentLocation, double** pointsInitialLocations, double** pointsVelocities,
	int** pointsSign, int** pointsSets);

void printMinimumTimeQAndWValuesToTextFile(double t, double tMax, double q, double* w, int K);

void initAllCudaVariables(double** device_w, double** device_points_locations, double** device_initial_points_locations,
	double** device_points_velocities, int** device_signs, int **device_task_per_proc, int** device_dimention_size, 
	double** device_time, int task_per_proc, int K, double* cpu_initial_points_location, double* cpu_points_velocities);

double* allocateDevicePointsLocation(int task_per_proc, int K);
double* allocateDeviceWArr(int K);
double* allocateAndCopyDeviceInitialPointsPositions(double* cpu_initial_locations, int task_per_proc, int K);
double* allocateAndCopyDevicePointsVelocities(double* cpu_points_velocities, int task_per_proc, int K);
int* allocateAndCopyDeviceDimentionSize(int* cpu_dimention_size);
double* allocateDeviceTime();
int* allocateDeviceSigns(int task_per_proc);
int *allocateAndCopyDeviceAmountOfTask(int* cpu_task_per_proc);

void checkErrors(cudaError_t cudaStatus, const char* errorMessage, void* arr);

void checkNumOfThreadsAndBlocksRequired(cudaDeviceProp props, int* numThreads,
	int* numBlocks, int* extraBlock, int task_per_proc);

void freeAllCudaVariables(int* device_task_per_proc, int* device_signs, double* device_time, int* device_dimention_size,
	double* device_points_velocities, double* device_initial_locations, double* device_w, double* device_points_locations);

void initPointsVariables(double** points_current_location, double** points_initial_location,
	double** points_velocities, int** points_signs, int** points_sets, int K, int task_per_proc);

void freePointsVariables(double** points_current_location, double** points_initial_location,
	double** points_velocities, int** points_signs, int** points_sets);

void sendTasksToSlaves(double* all_points_current_position, double* all_points_initial_position,
	double* all_points_velocities, int *all_points_sets, int k, int task_per_proc,int reminder, int numprocs);

void sendWToSlaves(double* w, int K, int numprocs);

void recieveSignsFromSlaves(int* all_points_signs, int task_per_proc,int reminder, int numprocs, MPI_Status status);

int checkAllPoints(int* signs, int* sets, int N, int iterCount, int lastBadPointIndex);

int checkPointsAccordingToInputFile(int* signs, int* sets, int start, int end);

void sendSlavesIfAllPointsAreOk(int allPointsAreOk, int numprocs);

void recieveNumOfMissesFromSlaves(int* numOfMisses, int numprocs, MPI_Status status);

double calculate_q(int numOfMisses, int N);

void sendTimeToSlaves(double t, int numprocs);

void receiveCurrentPointPositionFromSlaves(double* all_points_current_position, int task_per_proc, int reminder,
	int K, int numprocs, MPI_Status status);

void sendSlavesIfToKeepFindingQuality(int keepTryToFindQuality, int numprocs);

void recievePointsValuesFromMaster(double* myPointsCurrentLocation, double* myPointsInitialLocation,
	double* myPointsVelocities, int* myPointsSets, int task_per_proc, int K, MPI_Status status);

int findMin(int* arr, int size);

void slaveProcess(double* w, int task_per_proc, int K, double t, double* device_w, double* device_points_locations,
	double* device_initial_points_locations, double* device_points_velocities,
	int* device_signs, int* device_task_per_proc, int* device_dimention_size, double* device_time, MPI_Status status);


void masterProcess(double alpha, double tMax, double t, double dT, double* w, int LIMIT, int N, int K,
	double QC, int task_per_proc, int numprocs, int reminder, double* all_points_current_position
	, double* all_points_initial_position, double* all_points_velocities, int* all_points_sets, int* all_points_signs,
	double* device_w, double* device_points_locations, double* device_initial_points_locations,
	double* device_points_velocities, int* device_signs, int* device_task_per_proc,
	int* device_dimention_size, double* device_time, MPI_Status status);

void CalcNewPointPositionSequential(double* all_points_current_location, double* all_points_initial_loaction, double* velocities,
	int N, int K, double t);

double getQuailtySequential(int* signs, int* sets, int N);

void sequantialSolution(double QC, double t, double tMax, double* w, int K, int LIMIT, double alpha, double dT,
	int N, double* all_points_current_position, double* all_points_initial_position, double* all_points_velocities,
	int* all_points_signs, int* all_points_sets);

void printMinimumTimeQAndWValues(double t, double tMax, double q, double* w, int K);

void printTimeWasNotFound();

int chooseFirstMin(int* arr, int size);

void initBadPointsIndexesArr(int* arr, int size);

void parellelSolution(double alpha, double tMax, double dT, double* w, int LIMIT, int N, int K,
	double QC, int numprocs, double* all_points_current_position
	, double* all_points_initial_position, double* all_points_velocities, int* all_points_sets, int* all_points_signs,
	double* device_w, double* device_points_locations, double* device_initial_points_locations,
	double* device_points_velocities, int* device_signs, int* device_task_per_proc,
	int* device_dimention_size, double* device_time, MPI_Status status, int myid);

void sendResultsToRoot(double* w, double K, double current_time, double q);
void recieveResultsFromSlaves(double* w, double K, double* current_time, double* q, int best_time_process_id, MPI_Status* status);
int checkIfTimeWasFound(int* best_time_arr, int numprocs);
void broadcastAllData(double* dT, double* alpha, int* LIMIT, double* tMax, double* QC, double* all_points_current_position,
	double* all_points_initial_position, double* all_points_velocities, int* all_points_sets, int N, int K);