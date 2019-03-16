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

	const double M = atof(argv[1]);
	const double K = atof(argv[2]);
	const double N = atof(argv[3]);

	int sizeA = M*K;
	int sizeB = K*N;
	int sizeC = M*N;
	
	double R = 10.0;
	int *numPtrMatA = malloc(sizeof(int) * sizeA );
	int *numPtrMatB = malloc(sizeof(int) * sizeB );
	int *numPtrMatC = malloc(sizeof(int) * sizeC );
	
	for (int i = 0; i < sizeA; i++)
	{
		numPtrMatA[i] = R*(double)rand()/(double)(RAND_MAX)*R;
	}

	for (int i = 0; i < sizeB; i++)
	{
		numPtrMatB[i] = R*(double)rand()/(double)(RAND_MAX);
	}

	for (int i = 0; i < sizeC; i++)
	{
		numPtrMatC[i] = 1;
	}
	
	double ts1 = timestamp();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,M,N,K,1,numPtrMatA, K, numPtrMatB, N,0,numPtrMatC,M);
	double ts2 = timestamp();

	free(numPtrMatA);
	free(numPtrMatB);
	free(numPtrMatC);

	printf("OpenBlas time: %f \n",ts2-ts1);

}
