#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cblas.h>

static inline double timestamp()
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);

	return (ts.tv_sec + ts.tv_nsec * 1e-9) * 1000;
}

int main(int argc, char * argv[]){

	srand((unsigned int)time(NULL));

	const int M = atoi(argv[1]);
	const int K = atoi(argv[2]);
	const int N = atoi(argv[3]);

	int sizeA = M*K;
	int sizeB = K*N;
	int sizeC = M*N;
	
	float R = 1.0;
	int *numPtrMatA = malloc(sizeof(float) * sizeA );
	int *numPtrMatB = malloc(sizeof(float) * sizeB );
	int *numPtrMatC = malloc(sizeof(float) * sizeC );
	
	for (int i = 0; i < sizeA; i++)
	{
		numPtrMatA[i] = R*(float)rand()/(float)(RAND_MAX)*R;
	}

	for (int i = 0; i < sizeB; i++)
	{
		numPtrMatB[i] = R*(float)rand()/(float)(RAND_MAX);
	}

	for (int i = 0; i < sizeC; i++)
	{
		numPtrMatC[i] = 1.0;
	}
	
	double ts1 = timestamp();
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,M,N,K,1.0,numPtrMatA, K, numPtrMatB, N,0.0,numPtrMatC,N);
	double ts2 = timestamp();

	free(numPtrMatA);
	free(numPtrMatB);
	free(numPtrMatC);

	printf("OpenBlas float time: %f \n",ts2-ts1);

}
