//Guy Afik 311361612


//first solution:
//each processes gets all the points and calculates diffrent Times in parellel, for example: if we have 3 processes,
//the first 3 diffrent Times will be calculated in parellel, if we didn't find a matching q then we keep looping until we do or stop if there is not
//each process uses cuda in order to calculate f function and it's signs.
//each process uses cuda also in order to calculate new point positions according to the new update time
//each process also uses openMP to calculate the number of misses and the quality
//each process uses openMP to check the wrong point according to the order in the file for his own time that it is responsible about
//the reason I chose to work in that way is to save as much as communication possible between the processes


//second solution:
//I decided to work in a static methology which means every process gets the same amount of points
//each process works on his points an shares only the relevent data to the Master process
//the Master process is also responsible to manage the entire algorithm and to tell slaves what to do next
//if there is a reminder, the master slaves calculates it as well
//each process uses cuda in order to calculate f function and it's signs.
//each process uses cuda also in order to calculate new point positions according to the new update time
//each process also uses openMP to calculate the number of misses and the quality
//the master slave uses openMP to check the wrong point according to the order in the file
//the reason i chose to do this on in a static methology was because it I preffered as less as possible data sharing beetween the processes
//this is because the communication between the processes costs time

#include "main.h"


int main(int argc, char* argv[])
{
	parellelBinaryClassification(argc, argv);
	return 0;
}

void parellelBinaryClassification(int argc, char* argv[])
{
	double t0, t1, dT, tMax, alpha, QC, *w, t = 0;
	int numprocs, myid, N, K, LIMIT;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	double* all_points_current_position, *all_points_initial_position, *all_points_velocities;
	int* all_points_signs, *all_points_sets;
	t0 = MPI_Wtime();
	MPI_Status status;
	if (myid == ROOT)
	{
		readDataFromFile(&N, &K, &dT, &tMax, &alpha, &LIMIT, &QC, &all_points_current_position,
			&all_points_initial_position, &all_points_velocities, &all_points_signs, &all_points_sets);
		copyInitialPointsLocation(all_points_current_position, all_points_initial_position, K, N);
	}

	if (numprocs != 1) //in case we have more than 1 process, it means we want the parellel code to be excecuted
	{
		double* device_w, *device_points_locations, *device_initial_points_locations,
			*device_points_velocities, *device_time;
		int* device_signs, *device_task_per_proc, *device_dimention_size;

		int chooseSolution = 1; //change to 1 for the first solution, change to 2 for second solution

		MPI_Bcast(&K, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(&N, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

		if (myid != ROOT)
			initPointsVariables(&all_points_current_position, &all_points_initial_position, &all_points_velocities,
				&all_points_signs, &all_points_sets, K, N);
		//first solution
		if (chooseSolution == 1)
		{
			//broadcast all data to all processes
			broadcastAllData(&dT, &alpha, &LIMIT, &tMax, &QC, all_points_current_position, all_points_initial_position,
				all_points_velocities, all_points_sets, N, K);

			w = (double*)malloc(K + 1 * sizeof(double));


			parellelSolution(alpha, tMax, dT, w, LIMIT, N, K, QC, numprocs, all_points_current_position,
				all_points_initial_position, all_points_velocities, all_points_sets, all_points_signs, device_w,
				device_points_locations, device_initial_points_locations, device_points_velocities, device_signs,
				device_task_per_proc, device_dimention_size, device_time, status, myid);


			//free cuda variables and points variables
			freePointsVariables(&all_points_current_position, &all_points_initial_position, &all_points_velocities,
				&all_points_signs, &all_points_sets);
			freeAllCudaVariables(device_task_per_proc, device_signs, device_time, device_dimention_size, device_points_velocities,
				device_initial_points_locations, device_w, device_points_locations);

		}
		//second solution
		if (chooseSolution == 2)
		{
			int task_per_proc = N / numprocs, reminder = N % numprocs; //check number of works to be done by each process
			//in case there is a reminder, the master process will calculate it as well

			w = (double*)malloc(K + 1 * sizeof(double));

			if (myid == ROOT)
			{
				masterProcess(alpha, tMax, t, dT, w, LIMIT, N, K, QC, task_per_proc, numprocs, reminder,
					all_points_current_position, all_points_initial_position, all_points_velocities, all_points_sets,
					all_points_signs, device_w, device_points_locations, device_initial_points_locations,
					device_points_velocities, device_signs, device_task_per_proc, device_dimention_size, device_time, status);
			}
			else
			{
				slaveProcess(w, task_per_proc, K, t, device_w, device_points_locations,
					device_initial_points_locations, device_points_velocities, device_signs, device_task_per_proc,
					device_dimention_size, device_time, status);
			}
			if (myid == ROOT)
				freePointsVariables(&all_points_current_position, &all_points_initial_position, &all_points_velocities,
					&all_points_signs, &all_points_sets);
			freeAllCudaVariables(device_task_per_proc, device_signs, device_time, device_dimention_size, device_points_velocities,
				device_initial_points_locations, device_w, device_points_locations);
			free(w);
		}
	}
	else //we have here only 1 process, so he will do sequential solution
	{
		sequantialSolution(QC, t, tMax, w, K, LIMIT, alpha, dT, N, all_points_current_position, 
			all_points_initial_position ,all_points_velocities, all_points_signs, all_points_sets);
	}
	t1 = MPI_Wtime();
	if (myid == ROOT)
		printf("\ntime: %lf", t1 - t0);
	MPI_Finalize();
}


void printTimeWasNotFound()
{
	printf("Time was not found!");
}

void sequantialSolution(double QC, double t, double tMax, double* w, int K, int LIMIT, double alpha, double dT,
	int N, double* all_points_current_position, double* all_points_initial_position, double* all_points_velocities,
	int* all_points_signs, int* all_points_sets) //sequantial solution
{
	w = (double*)malloc(K + 1 * sizeof(double));
	double fValue, q = QC + 1;
	t = 0;
	int lastBadIndex, badPointIndex;
	while (q > QC && t < tMax)
	{
		int iterCount = 0;
		int numOfMisses = 0;
		setWZeros(w, K);
		int allPointsAreOk = -1;
		lastBadIndex = 0;
		while (iterCount < LIMIT && allPointsAreOk == -1)
		{
			allPointsAreOk = 1;
			for (int i = 0; i < N; i++)
			{
				fValue = f(w, all_points_current_position + i * K, K);
				all_points_signs[i] = getFSign(fValue);
			}
			badPointIndex = checkAllPoints(all_points_signs, all_points_sets, N, iterCount, lastBadIndex);
			if (badPointIndex != -1)
			{
				updateWValues(w, all_points_current_position + badPointIndex * K,
					all_points_signs[badPointIndex], K, alpha);
				allPointsAreOk = -1;
				lastBadIndex = badPointIndex;
			}
			iterCount++;
		}
		q = getQuailtySequential(all_points_signs, all_points_sets, N);
		if (q > QC)
		{
			t += dT;
			CalcNewPointPositionSequential(all_points_current_position, all_points_initial_position,
				all_points_velocities, N, K, t);
		}
	}
	//printMinimumTimeQAndWValuesToTextFile(t, tMax, q, w, K);
	printMinimumTimeQAndWValues(t, tMax, q, w, K); 
	free(w);
}


void parellelSolution(double alpha, double tMax, double dT, double* w, int LIMIT, int N, int K,
	double QC, int numprocs, double* all_points_current_position ,double* all_points_initial_position, 
	double* all_points_velocities, int* all_points_sets, int* all_points_signs, double* device_w, double* device_points_locations
	,double* device_initial_points_locations, double* device_points_velocities, int* device_signs, 
	int* device_task_per_proc, int* device_dimention_size, double* device_time, MPI_Status status, int myid)
{
	int numOfMisses = 0, lastBadPointIndex, iterCount;
	int numThreads, numBlocks, extraBlock;
	int best_time_process_id = -1;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	//init all cuda variables
	initAllCudaVariables(&device_w, &device_points_locations, &device_initial_points_locations, //initilize all cuda variables
		&device_points_velocities, &device_signs, &device_task_per_proc, &device_dimention_size,
		&device_time, N, K, all_points_initial_position ,all_points_velocities);

	//here we caluclate only once the number of threads required to be used in cuda
	checkNumOfThreadsAndBlocksRequired(props, &numThreads, &numBlocks, &extraBlock, N);

	int q_is_good = 0;
	int* best_time = (int*)malloc(sizeof(int)*numprocs); //arr to save the processes with the best time
	double q = QC + 1;
	for (double current_time = myid * dT; current_time < tMax && best_time_process_id == -1; current_time +=  numprocs * dT)
	{
		setWZeros(w, K);
		iterCount = 0;
		numOfMisses = 0;
		int allPointsAreOk = -1;
		if (q >= QC && current_time != 0)
			positionCalculationWithCuda(device_points_locations, device_initial_points_locations,
				device_points_velocities, device_dimention_size, device_task_per_proc,
				device_time, all_points_current_position, &current_time, N, K, numThreads, numBlocks, extraBlock);

		while (iterCount < LIMIT && allPointsAreOk == -1)
		{
			//calc signs and f with cuda
			fAndSignCalculationWithCuda(device_dimention_size, K, device_w,
				device_task_per_proc, device_signs, device_points_locations,
				N, all_points_current_position, all_points_signs, w, numThreads, numBlocks, extraBlock);

			//check for bad point with openMP
			int badPointIndex = checkAllPoints(all_points_signs, all_points_sets, N, iterCount, lastBadPointIndex);

			//if there is wrong point calculate updated W
			if (badPointIndex != -1)
			{
				updateWValues(w, all_points_current_position + badPointIndex * K,
					all_points_signs[badPointIndex], K, alpha);
				allPointsAreOk = -1;
				lastBadPointIndex = badPointIndex; 
			}
			iterCount++;
		}
		//check num of misses after each internal loop iteration
		numOfMisses = getQuailtyWithOpenMP(all_points_signs, all_points_sets, N);
		//calc q
		q = calculate_q(numOfMisses, N);

		//it means we found a correct q!
		if (q < QC)
			q_is_good = 1;

		printf("q is : %lf myid is: %d \n", q, myid);

		//send to master all processes result if they found a matching q
		MPI_Gather(&q_is_good, 1, MPI_INT, best_time, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

		//check if time was found, in case best_time_processes_id variable is still -1, we didn't find the right time or q
		if (myid == ROOT)
			best_time_process_id = checkIfTimeWasFound(best_time, numprocs);

		//let all the processes know if to keep and try finding q or to stop
		MPI_Bcast(&best_time_process_id, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

		if (myid == best_time_process_id && myid != ROOT)
			sendResultsToRoot(w, K, current_time, q);
		else if (myid == ROOT && myid == best_time_process_id)
		{
			//printMinimumTimeQAndWValuesToTextFile(current_time, tMax, q, w, K);
			printMinimumTimeQAndWValues(current_time, tMax, q, w, K);
		} 
		if (myid == ROOT && myid != best_time_process_id && best_time_process_id != -1)
		{
			recieveResultsFromSlaves(w, K, &current_time, &q, best_time_process_id, &status);
			//printMinimumTimeQAndWValuesToTextFile(current_time, tMax, q, w, K);
			printMinimumTimeQAndWValues(current_time, tMax, q, w, K);
		}
	}
	//free array
	free(best_time);
}


void broadcastAllData(double* dT, double* alpha, int* LIMIT, double* tMax, double* QC, double* all_points_current_position,
	double* all_points_initial_position, double* all_points_velocities, int* all_points_sets, int N, int K)
{
	MPI_Bcast(dT, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(alpha, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(LIMIT, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(tMax, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(QC, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

	MPI_Bcast(all_points_current_position, N*K, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(all_points_initial_position, N*K, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(all_points_velocities, N*K, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(all_points_sets, N, MPI_INT, ROOT, MPI_COMM_WORLD);
}

void recieveResultsFromSlaves(double* w, double K, double* current_time, double* q, int best_time_process_id, MPI_Status* status)
{
	MPI_Recv(w, K + 1, MPI_DOUBLE, best_time_process_id, DATA_TAG, MPI_COMM_WORLD, status);
	MPI_Recv(current_time, 1, MPI_DOUBLE, best_time_process_id, DATA_TAG, MPI_COMM_WORLD, status);
	MPI_Recv(q, 1, MPI_DOUBLE, best_time_process_id, DATA_TAG, MPI_COMM_WORLD, status);
}

void sendResultsToRoot(double* w, double K, double current_time, double q)
{
	MPI_Send(w, K + 1, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD);
	MPI_Send(&current_time, 1, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD);
	MPI_Send(&q, 1, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD);
}

int checkIfTimeWasFound(int* best_time_arr, int numprocs)
{
	for (int i = 0; i < numprocs; i++)
	{
		if (best_time_arr[i] == 1)
		{
			return i;
		}
	}
	return -1;
}

void printMinimumTimeQAndWValuesToTextFile(double t, double tMax, double q, double* w, int K)
{
	FILE *f = fopen(OUTPUT_FILE, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	if (t != tMax)
	{
		fprintf(f, "T minimum: %lf       q =  %lf", t, q);
		for (int i = 0; i < K + 1; i++)
		{
			fprintf(f, "\n%lf", w[i]);
		}
	}
	else 
	{
		fprintf(f, "time was not found!");
	}
	fclose(f);
}

void printMinimumTimeQAndWValues(double t,double tMax, double q, double* w, int K)
{
	if (t != tMax)
	{
		printf("T minimum: %lf       q =  %lf", t, q);
		for (int i = 0; i < K + 1; i++)
		{
			printf("\n%lf", w[i]);
		}
	}
	else
	{
		printf("time was not found");
	}
}

double getQuailtySequential(int* signs, int* sets, int N)
{
	int numOfMisses = 0;
	int i;
	for (i = 0; i < N; i++)
	{
		if (signs[i] != sets[i])
			numOfMisses++;
	}
	return (double)numOfMisses / N;
}


void CalcNewPointPositionSequential(double* all_points_current_location, double* all_points_initial_loaction, double* velocities,
	int N, int K, double t)
{
	for (int i = 0; i < N*K; i++)
	{
		all_points_current_location[i] = all_points_initial_loaction[i] + velocities[i] * t;
	}
}


void masterProcess(double alpha, double tMax, double t, double dT, double* w, int LIMIT, int N, int K,
	double QC, int task_per_proc, int numprocs, int reminder, double* all_points_current_position
,double* all_points_initial_position, double* all_points_velocities, int* all_points_sets, int* all_points_signs,
double* device_w, double* device_points_locations, double* device_initial_points_locations, 
double* device_points_velocities, int* device_signs, int* device_task_per_proc,
int* device_dimention_size, double* device_time, MPI_Status status)
{
	int lastBadPointIndex, iterCount, numOfMisses;
	sendTasksToSlaves(all_points_current_position, all_points_initial_position, all_points_velocities,
		all_points_sets, K, task_per_proc, reminder, numprocs); //each slave recieve same amount of tasks and work on his part the entire algorithm
	cudaDeviceProp props;
	int numThreads, numBlocks, extraBlock;


	cudaGetDeviceProperties(&props, 0);
	initAllCudaVariables(&device_w, &device_points_locations, &device_initial_points_locations, //initilize all cuda variables
		&device_points_velocities, &device_signs, &device_task_per_proc, &device_dimention_size,
		&device_time, task_per_proc + reminder, K, all_points_initial_position
		,all_points_velocities);
	checkNumOfThreadsAndBlocksRequired(props, &numThreads, &numBlocks,  //here we caluclate only once the number of threads required to be used in cuda
		&extraBlock, task_per_proc + reminder);
	
	int keepTryToFindQuality = -1;
	double q = QC + 1;
	t = 0;
	while (q > QC && t < tMax)
	{
		setWZeros(w, K);
		iterCount = 0;
		numOfMisses = 0;
		int allPointsAreOk = -1;
		while (iterCount < LIMIT && allPointsAreOk == -1)
		{
			allPointsAreOk = 1;

			sendWToSlaves(w, K, numprocs);

			//calculate f function and the sign values
			fAndSignCalculationWithCuda(device_dimention_size, K, device_w,
				device_task_per_proc, device_signs, device_points_locations,
				task_per_proc + reminder, all_points_current_position,
				all_points_signs, w, numThreads, numBlocks, extraBlock);

			//recieve signs after f calculation from every slave
			recieveSignsFromSlaves(all_points_signs, task_per_proc,reminder, numprocs, status);

			//pass on all points and check if all points are ok with openMP
			int badPointIndex = checkAllPoints(all_points_signs, all_points_sets, N, iterCount, lastBadPointIndex);

			//if there is wrong point calculate updated W
			if (badPointIndex != -1)
			{
				updateWValues(w, all_points_current_position + badPointIndex * K,
					all_points_signs[badPointIndex], K, alpha);
				allPointsAreOk = -1;
				lastBadPointIndex = badPointIndex;
			}
			iterCount++;
			if (iterCount == LIMIT)
				allPointsAreOk = 1;

			//tell the slaves if to keep calculate f function and signs for them or to stop
			sendSlavesIfAllPointsAreOk(allPointsAreOk, numprocs);

		}
		//check the number of misses with openMP
		numOfMisses = getQuailtyWithOpenMP(all_points_signs, all_points_sets, reminder + task_per_proc);

		//get slaves numOfMisses from slaves
		recieveNumOfMissesFromSlaves(&numOfMisses, numprocs, status);

		//calc q
		q = calculate_q(numOfMisses, N);

		if (q >= QC)
			t += dT;
		//send here the time to slaves processes in order to calculate new positions
		sendTimeToSlaves(t, numprocs);

		//calculate new postions of the points with cuda
		positionCalculationWithCuda(device_points_locations, device_initial_points_locations,
			device_points_velocities, device_dimention_size, device_task_per_proc,
			device_time, all_points_current_position, &t, task_per_proc + reminder, K,
			numThreads, numBlocks, extraBlock);

		//recieve from slaves the new position for the rest of the points
		receiveCurrentPointPositionFromSlaves(all_points_current_position, task_per_proc,reminder, K, numprocs, status);

		//check to see if the master can tell slaves to stop the algorithm or to keep going
		if (t >= tMax || q < QC)
			keepTryToFindQuality = 1;
		else
			keepTryToFindQuality = -1;

		//send the slaves if they need to keep going with the algorithm
		sendSlavesIfToKeepFindingQuality(keepTryToFindQuality, numprocs);
	}
	//if we reached the tMax it means time was not found and we need to print it, otherwise printing time minimum, q and w vector

	//printMinimumTimeQAndWValuesToTextFile(t, tMax, q, w, K);
	printMinimumTimeQAndWValues(t, tMax, q, w, K); 
}


void slaveProcess(double* w, int task_per_proc, int K, double t, double* device_w, double* device_points_locations,
	double* device_initial_points_locations, double* device_points_velocities,
	int* device_signs, int* device_task_per_proc, int* device_dimention_size, double* device_time, MPI_Status status)
{
	int keepCalculationFValues = -1, keepCalculatingQualitiesAndVelocity = -1;
	double* myPointsCurrentLocation, *myPointsInitialLocation, *myPointsVelocities;
	int* myPointsSets, *myPointsSign, numOfMisses = 0;
	cudaDeviceProp props;
	int numThreads, numBlocks, extraBlock;

	initPointsVariables(&myPointsCurrentLocation, &myPointsInitialLocation, &myPointsVelocities,
		&myPointsSign, &myPointsSets, K, task_per_proc); //allocate memory for the part that this specific slaves work on

	recievePointsValuesFromMaster(myPointsCurrentLocation, myPointsInitialLocation, myPointsVelocities,
		myPointsSets, task_per_proc, K, status); //here each slave recives from master the part he is responsible about

	initAllCudaVariables(&device_w, &device_points_locations, &device_initial_points_locations,
		&device_points_velocities, &device_signs, &device_task_per_proc, &device_dimention_size,
		&device_time, task_per_proc, K, myPointsInitialLocation, myPointsVelocities); //allocate cuda memory for each slave

	cudaGetDeviceProperties(&props, 0);

	checkNumOfThreadsAndBlocksRequired(props, &numThreads, &numBlocks, &extraBlock, task_per_proc); //caluclate num of threads required once

	while (keepCalculatingQualitiesAndVelocity == -1)
	{
		while (keepCalculationFValues == -1)
		{

			MPI_Recv(w, K + 1, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD, &status); //receive here the W vector each iteration

			//here its gonna caluclate F and Sign with cuda for each slave
			fAndSignCalculationWithCuda(device_dimention_size, K, device_w, 
				device_task_per_proc, device_signs, device_points_locations,
				task_per_proc, myPointsCurrentLocation, myPointsSign, w, numThreads, numBlocks, extraBlock);

			MPI_Send(myPointsSign, task_per_proc, MPI_INT, ROOT, DATA_TAG, MPI_COMM_WORLD); //each process here sends the signs of his part to the master 

			MPI_Recv(&keepCalculationFValues, 1, MPI_INT, ROOT, DATA_TAG, MPI_COMM_WORLD, &status); //here the slaves will know if to keep calculating f function and the signs or to stop

		}
		keepCalculationFValues = -1;

		numOfMisses = getQuailtyWithOpenMP(myPointsSign, myPointsSets, task_per_proc); //get num of misses with openMP

		MPI_Send(&numOfMisses, 1, MPI_INT, ROOT, DATA_TAG, MPI_COMM_WORLD); //after calculation of num of misses, each slave sends his result to master

		double slavesT = t;
		MPI_Recv(&t, 1, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD, &status);

		if (slavesT < t) //if slavesT remains the same, it means we found a minimumT, and we do not want the slaves to keep calculate the point positions because the algorithm has ended
			positionCalculationWithCuda(device_points_locations, device_initial_points_locations,
				device_points_velocities, device_dimention_size, device_task_per_proc,
				device_time, myPointsCurrentLocation, &t, task_per_proc, K, numThreads, numBlocks, extraBlock); //calculate new positions with cuda


		MPI_Send(myPointsCurrentLocation, task_per_proc * K, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD); //send new postions to the master process

		MPI_Recv(&keepCalculatingQualitiesAndVelocity, 1, MPI_INT, ROOT, DATA_TAG, MPI_COMM_WORLD, &status); //let the slave know if to keep running the algorithm or to stop

	}
	freePointsVariables(&myPointsCurrentLocation, &myPointsInitialLocation,
		&myPointsVelocities, &myPointsSign, &myPointsSets); // free the point variables
}


void checkNumOfThreadsAndBlocksRequired(cudaDeviceProp props, int* numThreads,
	int* numBlocks, int* extraBlock, int task_per_proc)
{
	*numThreads = props.maxThreadsPerBlock < task_per_proc ? props.maxThreadsPerBlock : task_per_proc;
	*numBlocks = task_per_proc / *numThreads;
	*extraBlock = task_per_proc % *numThreads != 0;
}


void recievePointsValuesFromMaster(double* myPointsCurrentLocation, double* myPointsInitialLocation,
	double* myPointsVelocities, int* myPointsSets, int task_per_proc, int K, MPI_Status status)
{
	MPI_Recv(myPointsCurrentLocation, K*task_per_proc, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(myPointsInitialLocation, K*task_per_proc, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(myPointsVelocities, K*task_per_proc, MPI_DOUBLE, ROOT, DATA_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(myPointsSets, task_per_proc, MPI_INT, ROOT, DATA_TAG, MPI_COMM_WORLD, &status);
}



void sendTimeToSlaves(double t, int numprocs)
{
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Send(&t, 1, MPI_DOUBLE, id, DATA_TAG, MPI_COMM_WORLD);
	}
}



void sendSlavesIfToKeepFindingQuality(int keepTryToFindQuality, int numprocs)
{
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Send(&keepTryToFindQuality, 1, MPI_INT, id, DATA_TAG, MPI_COMM_WORLD);
	}
}



void receiveCurrentPointPositionFromSlaves(double* all_points_current_position, int task_per_proc, int reminder,
	int K, int numprocs, MPI_Status status)
{
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Recv((all_points_current_position + id * K * task_per_proc) + reminder * K, K * task_per_proc, MPI_DOUBLE, id, DATA_TAG, MPI_COMM_WORLD, &status);
	}
}

double calculate_q(int numOfMisses, int N)
{
	return double(numOfMisses) / N;
}


void recieveNumOfMissesFromSlaves(int* numOfMisses, int numprocs, MPI_Status status)
{
	int currentNumOfMisses;
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Recv(&currentNumOfMisses, 1, MPI_INT, id, DATA_TAG, MPI_COMM_WORLD, &status);
		*numOfMisses += currentNumOfMisses;
	}
}

void sendSlavesIfAllPointsAreOk(int allPointsAreOk, int numprocs)
{
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Send(&allPointsAreOk, 1, MPI_INT, id, DATA_TAG, MPI_COMM_WORLD);
	}
}



int chooseFirstMin(int* arr, int size)
{
	int min = -1;
	for (int i = 0; i < size; i++)
	{
		if (arr[i] != -1)
		{
			min = arr[i];
			break;
		}
	}
	return min;
}

int findMin(int* arr, int size)
{
	int min = chooseFirstMin(arr, size); //in case we have [-1,7,-1,9] for example, we don't want to consider -1's
	//it means in this example that thread number 0 and 2 didn't find any wrong point.
	//it's for a case where the threads found less than 4 wrong points. or in general case we have less wrong points than Max threads in a particular machine
	for (int i = 1; i < size; i++)
	{
		if (arr[i] != -1)
		{
			if (min > arr[i])
				min = arr[i];
		}
	}
	return min;
}


void initBadPointsIndexesArr(int* arr, int size)
{
	for (int j = 0; j < size; j++)
	{
		arr[j] = -1;
	}
}

int checkPointsAccordingToInputFile(int* signs, int* sets, int start, int end) 
{
	//instead of calculating the points in an iterative way, I decided to take the maximum threads in the computer
	//each thread will find a bad point index, and then just take the minimum index from all of them
	//that would promise that we take the right bad point
	//find minimum from a short array wouldn't hurt that bad in the performence 
	//when each thread finds his bad point, he breaks from the loop
	int i, badPointIndex = -1;
	int maxThreads = omp_get_max_threads();
	int* badPointsIndexesArr = (int*)malloc(sizeof(int)*maxThreads);
	initBadPointsIndexesArr(badPointsIndexesArr, maxThreads);
#pragma omp parallel private(i)
	{
		int threadId = omp_get_thread_num();
		int numOfThreads = omp_get_num_threads();
		for (i = threadId + start; i < end; i += numOfThreads)
		{
			if (signs[i] != sets[i])
			{
				badPointsIndexesArr[threadId] = i;
				break;
			}
		}
	}
	//find min
	badPointIndex = findMin(badPointsIndexesArr, maxThreads);
	free(badPointsIndexesArr);

	return badPointIndex;
}

int checkAllPoints(int* signs, int* sets, int N, int iterCount, int lastBadPointIndex)
{
	int badPointIndex = -1;
	if (iterCount == 0)
		lastBadPointIndex = -1;
	badPointIndex = checkPointsAccordingToInputFile(signs, sets, lastBadPointIndex + 1, N);
	if (badPointIndex == -1 && lastBadPointIndex != -1)
		badPointIndex = checkPointsAccordingToInputFile(signs, sets, 0, lastBadPointIndex);
	return badPointIndex;
}


void recieveSignsFromSlaves(int* all_points_signs, int task_per_proc,int reminder, int numprocs, MPI_Status status)
{
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Recv((all_points_signs + id * task_per_proc) + reminder, task_per_proc, MPI_INT,
			id, DATA_TAG, MPI_COMM_WORLD, &status);
	}
}


void sendWToSlaves(double* w, int K, int numprocs)
{
	for (int id = 1; id < numprocs; id++)
	{
		MPI_Send(w, K + 1, MPI_DOUBLE, id, DATA_TAG, MPI_COMM_WORLD);
	}
}

int getQuailtyWithOpenMP(int* signs, int* sets, int task_per_proc)
{
	int numOfMisses = 0;
	int i;
#pragma omp parallel for reduction(+:numOfMisses)
	for (i = 0; i < task_per_proc; i++)
	{
		if (signs[i] != sets[i])
			numOfMisses++;
	}
	return numOfMisses;
}



void copyInitialPointsLocation(double* currentPointLocation, double* initialPointLocation, int K, int N)
{
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < K*N; i++)
		initialPointLocation[i] = currentPointLocation[i];
}


void setWZeros(double* w, int K)
{
	for (int i = 0; i < K + 1; i++)
		w[i] = 0;
}

double f(double* w, double* pointValues, int K)
{
	int i;
	double result = w[K];
	for (i = 0; i < K; i++)
		result += w[i] * pointValues[i];
	return result;
}

int getFSign(double fValue)
{
	if (fValue >= 0)
		return 1;
	else 
		return -1;
}

void updateWValues(double* w, double* pointValues, int sign, int K, double alpha)
{
	//i thought about parreleling this function, but dimention size is maximum 20, it is not worth 
	//to make this parellel, the context switches between the threads would take longer. 
	//we want to parralelize big arrays usually
	w[K] += (-sign)*alpha;
	int i;
	for (i = 0; i < K; i++)
	{
		w[i] += -(sign)* alpha * pointValues[i];
	}
}

void sendTasksToSlaves(double* all_points_current_position, double* all_points_initial_position,
	double* all_points_velocities, int *all_points_sets, int K, int task_per_proc, int reminder, int numprocs)
{

	for (int id = 1; id < numprocs; id++)
	{
		MPI_Send((all_points_current_position + id * K * task_per_proc) + reminder * K, task_per_proc * K, MPI_DOUBLE,
			id, DATA_TAG, MPI_COMM_WORLD);
		MPI_Send((all_points_initial_position + id * K * task_per_proc) + reminder * K, task_per_proc * K, MPI_DOUBLE,
			id, DATA_TAG, MPI_COMM_WORLD);
		MPI_Send((all_points_velocities + id * K * task_per_proc) + reminder * K, task_per_proc * K, MPI_DOUBLE,
			id, DATA_TAG, MPI_COMM_WORLD);
		MPI_Send((all_points_sets + id * task_per_proc) + reminder, task_per_proc, MPI_INT, id, DATA_TAG, MPI_COMM_WORLD);
	}
}


void readDataFromFile(int* N, int* K, double* dT, double* tMax, double*alpha, int* LIMIT, double*QC,
	double** pointsCurrentLocation, double** pointsInitialLocation,
	double** pointsVelocities, int** pointsSign, int** pointsSets)
{
	FILE* f = fopen(INPUT_FILE, "r");
	if (f == NULL)
	{
		perror("Error");
		printf("Could not open file %s", INPUT_FILE);
	}
	fscanf(f, "%d %d %lf %lf %lf %d %lf\n", N, K, dT, tMax, alpha, LIMIT, QC);
	initPointsVariables(pointsCurrentLocation, pointsInitialLocation, pointsVelocities, pointsSets, pointsSign, *K, *N);
	int currentPointForLocations = 0;
	int currentPointForVelocities = 0;
	for (int i = 0; i < *N; i++)
	{
		for (int j = 0; j < *K; j++)
		{
			fscanf(f, "%lf", &((*pointsCurrentLocation)[currentPointForLocations]));
			currentPointForLocations++;
		}
		for (int j = 0; j < *K; j++)
		{
			fscanf(f, "%lf", &((*pointsVelocities)[currentPointForVelocities]));
			currentPointForVelocities++;
		}
		fscanf(f, "%d\n", &((*pointsSets)[i]));
	}
	fclose(f);

}

void initAllCudaVariables(double** device_w, double** device_points_locations, double** device_initial_points_locations,
	double** device_points_velocities, int** device_signs, int **device_task_per_proc,
	int** device_dimention_size, double** device_time,
	int task_per_proc, int K, double* cpu_initial_points_location, double* cpu_points_velocities)
{
	*device_w = allocateDeviceWArr(K);
	*device_points_locations = allocateDevicePointsLocation(task_per_proc, K);
	*device_initial_points_locations = allocateAndCopyDeviceInitialPointsPositions(cpu_initial_points_location,
		task_per_proc, K);
	*device_points_velocities = allocateAndCopyDevicePointsVelocities(cpu_points_velocities, task_per_proc, K);
	*device_signs = allocateDeviceSigns(task_per_proc);
	*device_dimention_size = allocateAndCopyDeviceDimentionSize(&K);
	*device_time = allocateDeviceTime();
	*device_task_per_proc = allocateAndCopyDeviceAmountOfTask(&task_per_proc);

}

void initPointsVariables(double** points_current_location, double** points_initial_location,
	double** points_velocities, int** points_signs, int** points_sets, int K, int task_per_proc)
{
	*points_current_location = (double*)malloc(sizeof(double)*task_per_proc*K);
	*points_initial_location = (double*)malloc(sizeof(double)*task_per_proc*K);
	*points_velocities = (double*)malloc(sizeof(double)*task_per_proc*K);
	*points_signs = (int*)malloc(sizeof(int)*task_per_proc);
	*points_sets = (int*)malloc(sizeof(int)*task_per_proc);
}

void freePointsVariables(double** points_current_location, double** points_initial_location,
	double** points_velocities, int** points_signs, int** points_sets)
{
	free(*points_current_location);
	free(*points_initial_location);
	free(*points_velocities);
	free(*points_signs);
	free(*points_sets);
}