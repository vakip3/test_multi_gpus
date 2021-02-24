#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "book.h"
#include <stdio.h>
#include "book.h"


int MAX_GPU_COUNT = 6;
//const int DATA_N = 1048576 * 32;
const int DATA_N = 8192;
const int blocks = 32;
const int threads = 256;
//kroz broj blokova
const int numOfStreams = 2;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void vectorProduct(int* vec_a, int* vec_b, int* res, int countPerStream)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tid = bid * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;
	//broj niti u bloku
	int threadsNum = blockDim.x * blockDim.y;
	int blocksNum = gridDim.x * gridDim.y;

	//tid je jedinstven u gridu
	//svaka nit cita svoje elemente niza i radi skalarni proizvod
	int ind = tid;
	while (ind < countPerStream)
	{
		//ne mora u sh jer se cita jednom
		res[ind] = vec_a[ind] * vec_b[ind];
		//printf("nit %d racuna %d * %d = %d\n", tid, vec_a[ind], vec_b[ind], res[ind]);
		ind += threadsNum*blocksNum;
	}
}

int main()
{
	//provere
	printf("Checking GPUs count\n");
	int gpu_n;
	cudaGetDeviceCount(&gpu_n);
	printf("CUDA-capable device count: %i\n", gpu_n);
	if (gpu_n < MAX_GPU_COUNT)
		MAX_GPU_COUNT = gpu_n;

	//broj elemenata niza po gpu
	int numPerGPU = DATA_N / MAX_GPU_COUNT;
	//ovoliko zauzimamo za svaki tok
	int numPerGPUStream = numPerGPU / numOfStreams;

	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);
	//printf("CUDA GPU %s UVA\n", prop.unifiedAddressing ? "supports" : "does not suppport");

	//multigpu deo

	//host
	int* vector_a, * vector_b, * res_vector;

	HANDLE_ERROR(cudaHostAlloc((void**)& vector_a, DATA_N * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)& vector_b, DATA_N * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)& res_vector, DATA_N * sizeof(int), cudaHostAllocDefault));

	for (int i = 0; i < DATA_N; i++)
	{
		vector_a[i] = rand()%100+1;
		vector_b[i] = rand()%50+1;
	}

	//gpus
	int*** gpu_a, *** gpu_b, *** gpu_c;
	//treba za svaki GPU
	gpu_a = new int** [MAX_GPU_COUNT];
	gpu_b = new int** [MAX_GPU_COUNT];
	gpu_c = new int** [MAX_GPU_COUNT];

	//kreiranje strimova, ima ih numOfStreams*gpucount
	cudaStream_t *streams;
	streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * MAX_GPU_COUNT*numOfStreams);
	//cudaStream_t streams[numOfStreams*MAX_GPU_COUNT];
	for (int i = 0; i < MAX_GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		for (int j = 0; j < numOfStreams; j++)
			cudaStreamCreate(&streams[i * MAX_GPU_COUNT + j]);
	}

	//zauzimanje memorije na gpuovima
	for (int i = 0; i < MAX_GPU_COUNT; i++)
	{
		//svaki GPU radi sa strimovima
		gpu_a[i] = new int* [numOfStreams];
		gpu_b[i] = new int* [numOfStreams];
		gpu_c[i] = new int* [numOfStreams];

		//postavljanje uredjaja da bi sve cuda operacije isle preko njega
		cudaSetDevice(i);

		for (int j = 0; j < numOfStreams; j++)
		{
			//zauzimanje memorije za strim
			//za svaki strim se zauzima po numPerGPuStream
			HANDLE_ERROR(cudaMalloc((void**)& gpu_a[i][j], numPerGPUStream*sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)& gpu_b[i][j], numPerGPUStream * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)& gpu_c[i][j], numPerGPUStream * sizeof(int)));

		}

	}

	//vreme
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	printf("Computing with %d GPUs.\n", MAX_GPU_COUNT);

	//asinhroni pozivi
	for (int i = 0; i < MAX_GPU_COUNT; i++)
	{
		cudaSetDevice(i);

		//kopiranje memorije, razdvojeno zbog preglednosti
		for (int j = 0; j < numOfStreams; j++)
		{
			//treba odrediti pocetak, odakle se salje s hosta
			int startIndex = i * numPerGPU + j * numPerGPUStream;
			int streamIndex = i * MAX_GPU_COUNT + j;
			//od startIndex se u svaki strim kopira numPerGPUSTream podataka
			HANDLE_ERROR(cudaMemcpyAsync(gpu_a[i][j], &vector_a[startIndex], numPerGPUStream * sizeof(int), cudaMemcpyHostToDevice, streams[streamIndex]));
			HANDLE_ERROR(cudaMemcpyAsync(gpu_b[i][j], &vector_b[startIndex], numPerGPUStream * sizeof(int), cudaMemcpyHostToDevice, streams[streamIndex]));

			vectorProduct << <blocks, threads, 0, streams[streamIndex] >> > (gpu_a[i][j], gpu_b[i][j], gpu_c[i][j], numPerGPUStream);

			HANDLE_ERROR(cudaMemcpyAsync(&res_vector[startIndex], gpu_c[i][j], numPerGPUStream * sizeof(int), cudaMemcpyDeviceToHost, streams[streamIndex]));

		}

	}

	//sinhronizacija strimova
	for (int i = 0; i < MAX_GPU_COUNT; i++)
	{
		for (int j = 0; j < numOfStreams; j++)
			HANDLE_ERROR(cudaStreamSynchronize(streams[i*MAX_GPU_COUNT+j]));
	}

	float elapsed;
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
	printf("Execution time %.3f ms.\n", elapsed);

	for (int i = 0; i < DATA_N; i++)
	{
		printf("%d * %d = %d\n", vector_a[i], vector_b[i], res_vector[i]);
	}

	//unistavanje strimova i brisanje gpu memorije
	for (int i = 0; i < MAX_GPU_COUNT; i++)
	{
		cudaSetDevice(i);
		for (int j = 0; j < numOfStreams; j++)
		{
			cudaStreamDestroy(streams[i * MAX_GPU_COUNT + j]);
			cudaFree(gpu_a[i][j]);
			cudaFree(gpu_b[i][j]);
			cudaFree(gpu_c[i][j]);
		}
		cudaFree(gpu_a[i]);
		cudaFree(gpu_b[i]);
		cudaFree(gpu_c[i]);
	}

	cudaFreeHost(vector_a);
	cudaFreeHost(vector_b);
	cudaFreeHost(res_vector);

	delete gpu_a;
	delete gpu_b;
	delete gpu_c;

    return 0;
}

