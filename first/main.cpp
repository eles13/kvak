#include <iostream>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <omp.h>
using namespace std;

typedef complex<double> complexd;

complexd* init(int n) {
	unsigned long long i, m = 1 << n;
	complexd *A = new complexd[m];
	double sum = 0;
	unsigned int seed = omp_get_wtime();
	#pragma omp parallel for shared(A, m) firstprivate(seed) private(i) reduction(+: sum)
		for (i = 0; i < m; ++i) {
			seed+=omp_get_wtime();
			A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5);
			A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5);
			sum += abs(A[i]*A[i]);
		}
	sum = sqrt(sum);
	#pragma omp parallel for
	for (i = 0; i < m; ++i) {
		A[i] /= sum;
	}
	return A;
}

complexd* f(complexd *A, int n, complexd *H, int k) {
	unsigned long long i, m = 1 << n, l = 1 << (n - k);
	complexd *B = new complexd[m];
	#pragma omp parallel for shared(A, B, H, m, l) private(i)
		for (i = 0; i < m; ++i)
		if ((i & l) == 0)
		{
			B[i] =  H[0]*A[i & ~l] + H[1]*A[i | l];
		}
		else
		{
			B[i] = H[2]*A[i & ~l] + H[3]*A[i | l];
		}
	return B;
}

int main(int argc, char **argv) {

	if (argc < 3) {
		cout << argv[0] << " n k" << endl;
		return 0;
	}
	int n, k;
	n = atoi(argv[1]);
	k = atoi(argv[2]);
	double start = omp_get_wtime();
	complexd *A = init(n);
	complexd H[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	complexd *B = f(A, n, H, k);
	start = omp_get_wtime() - start;
	char* numthr = getenv("OMP_NUM_THREADS");
	cout << n << ' ' << k << ' ' << ((numthr != NULL) ? numthr : "1") << ' ' << start << endl;
	delete [] A;
	delete [] B;
	return 0;

}
