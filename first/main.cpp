#include <iostream>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <omp.h>
#include <fstream>
using namespace std;

typedef complex<double> complexd;

complexd* init(int n, int mode) {
	unsigned long long i, m = 1 << n;
	complexd *A = new complexd[m];
	double sum = 0;
	unsigned int seed = omp_get_wtime();
	#pragma omp parallel for firstprivate(seed,mode) reduction(+: sum) schedule(guided)
		for (i = 0; i < m; ++i){
		    if(!mode) {
                seed += omp_get_wtime();
                A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5);
                A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5);
            }else {
                A[i].real((228 / (float) RAND_MAX) - 0.5);
                A[i].imag((228 / (float) RAND_MAX) - 0.5);
            }

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
	#pragma omp parallel for firstprivate(m,l)
		for (i = 0; i < m; ++i){
			if ((i & l) == 0){
				B[i] =  H[0]*A[i & ~l] + H[1]*A[i | l];
			}
			else{
				B[i] = H[2]*A[i & ~l] + H[3]*A[i | l];
			}
		}
	return B;
}

int main(int argc, char **argv) {
	if (argc < 5) {
		cout << "too few args"<< endl;
		return -1;
	}
	int n, k ,i;
	n = atoi(argv[1]);
	k = atoi(argv[2]);
	i = atoi(argv[3]);
	omp_set_num_threads(i);
	double start = omp_get_wtime();
	complexd *A = init(n, atoi(argv[4]));
	complexd H[] = {1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2)};
	complexd *B = f(A, n, H, k);
	start = omp_get_wtime() - start;
	ofstream fout;
	if (atoi(argv[4]) == 1) {
        ofstream ch("B.txt", ios::out | ios::binary);
        for (int i = 0; i < (1 << n); i++) {
            ch << B[i];
        }
    }
	fout.open("result.txt",ofstream::out | ofstream::app);
	fout << n << ' ' << k << ' ' << i << ' ' << start << endl;
	fout.close();
	delete [] A;
	delete [] B;
	return 0;

}
