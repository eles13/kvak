#include <cstdlib>
#include <stdio.h>
#include <complex>
#include <mpi.h>
#include <omp.h>

typedef std::complex<double> complexd;

static int rank, size, log_size;

unsigned long long log_2(unsigned long long& m){
    unsigned long long log;
    for (log = 0; !((m >> log) & 1); ++log);
    return log;
}

complexd *generate(int n, int argc) {
    unsigned long long m = (1LLU << n) / size;
    complexd *A = new complexd[m];
    double sqr = 0, module;
    unsigned int seed;
    if (argc != 4) {
        seed = time(0) + rank;
    } else {
        seed = rank;
    }
    for (unsigned long long i = 0; i < m; ++i) {
        A[i].real((rand_r(&seed) / (float)RAND_MAX) - 0.5f);
        A[i].imag((rand_r(&seed) / (float)RAND_MAX) - 0.5f);
        sqr += abs(A[i] * A[i]);
    }
    MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        module = sqrt(module);
    MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (unsigned long long i = 0; i < m; ++i)
        A[i] /= module;
    return A;
}

complexd *read(char *f, int *n) {
    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL, &file)) {
        if (rank == 0) {
            printf("Err in opening file\n");
        }
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if (rank == 0)
        MPI_File_read(file, n, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    unsigned long long m = (1LLU << *n) / size;
    complexd *A = new complexd[m];
    double num[2];
    MPI_File_seek(file, sizeof(int) + 2 * m * rank * sizeof(double),
                  MPI_SEEK_SET);
    for (unsigned long long i = 0; i < m; ++i) {
        MPI_File_read(file, &num, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
        A[i].real(num[0]);
        A[i].imag(num[1]);
        std::cout << rank << ' ' << i << ' ' << A[i] << std::endl;
    }
    MPI_File_close(&file);
    return A;
}

void f(complexd *A, complexd *B,int n, int k, complexd *H) {
    unsigned long long m = (1LLU << n) / size;
    complexd *buf = new complexd[m];
    int rank_ = ((rank * m) ^ (1LLU << (k - 1))) / m;
    if (rank != rank_) {
        MPI_Sendrecv(A, m, MPI_DOUBLE_COMPLEX, rank_, 0, buf, m, MPI_DOUBLE_COMPLEX,
                     rank_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank < rank_) {
            for (unsigned long long i = 0; i < m; i++) {
                B[i] = H[0] * A[i] + H[1] * buf[i];
            }
        } else {
            for (unsigned long long i = 0; i < m; i++) {
                B[i] = H[2] * buf[i] + H[3] * A[i];
            }
        }
    } else {
        unsigned long long l = 1LLU << log_2(m) - k;
        for (unsigned long long i = 0; i < m; i++) {
            B[i] = ((i & l) >> (log_2(m) - k) == 0)
                   ? H[0] * A[i & ~l] + H[1] * A[i | l]
                   : H[2] * A[i & ~l] + H[3] * A[i | l];
        }
    }
    delete[] buf;
}

double normal_dis_gen(unsigned int *seed)
{
	double S = 0.;
	for (int i = 0; i<12; ++i) {
		S += (double) rand_r(seed) / RAND_MAX;
	}
	return S-6.;
}

complexd *adam(complexd *A, int n, double e)
{
    unsigned long long m = (1LLU << n) / size;
	complexd *B = new complexd[m], *C = new complexd[m],  H[4];
	#pragma omp parallel for schedule(guided)
	for (unsigned long long i = 0; i < m; ++i) {
		B[i] = A[i];
	}
	double t;
	unsigned int seed = time(0);
	for (int k = 1; k <= n; ++k) {
		if (rank == 0) {
			t = normal_dis_gen(&seed);
			H[0] = (cos(e*t) - sin(e*t)) / sqrt(2);
			H[1] = H[2] = (cos(e*t) + sin(e*t)) / sqrt(2);
			H[3] = -H[0];
		}
		MPI_Bcast(H, 4, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
		f(B, C, n, k, H);
		std::swap(B,C);
	}
	delete [] C;
	return B;
}

complexd dot(complexd *A, complexd *B, int n)
{
    unsigned long long m = (1LLU << n) / size;
	complexd x(0.0, 0.0), y(0.0, 0.0);
	#pragma omp parallel
	{
		complexd z(0.0, 0.0);
		#pragma omp for schedule(guided)
		for (unsigned long long i = 0; i < m; ++i) {
			z += conj(A[i]) * B[i];
		}
		#pragma omp critical
		y += z;
	}
	MPI_Reduce(&y, &x, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	return x;
}

void write(char *args, complexd *B, int n) {
    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, args, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &file)) {
        if (rank == 0)
            printf("Err in opening ofile\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if (rank == 0)
        MPI_File_write(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);
    unsigned long long m = (1LLU << n) / size;
    double num[2];
    MPI_File_seek(file, sizeof(int) + 2 * m * rank * sizeof(double),
                  MPI_SEEK_SET);
    for (unsigned long long i = 0; i < m; ++i) {
        num[0] = B[i].real();
        num[1] = B[i].imag();
        MPI_File_write(file, &num, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
}

int main(int argc, char **argv)
{
    int n;
    double e;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    omp_set_num_threads(size);
    if (argc < 3) {
        MPI_Finalize();
        return 1;
    }
    n = atoi(argv[1]);
    e = atof(argv[2]);
    double time[2], maxtime[2];
	MPI_Barrier(MPI_COMM_WORLD);
	complexd *A = (argc > 4) ? read(argv[3], &n) : generate(n,argc);
	MPI_Barrier(MPI_COMM_WORLD);
	time[0] = MPI_Wtime();
	complexd *B = adam(A, n, e);
	time[0] = MPI_Wtime() - time[0];
	complexd *C = adam(A, n, 0.0);
	delete [] A;
	MPI_Barrier(MPI_COMM_WORLD);
	time[1] = MPI_Wtime();
	double loss = abs(dot(B, C, n));
	time[1] = MPI_Wtime() - time[1];
	loss = 1.0 - loss * loss;
	if (loss < 0.0)
		loss = 0.0;
	if (argc > 3)
		write(argv[4], B, n);
	delete [] B;
	delete [] C;
	MPI_Reduce(time, maxtime, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank == 0)
		printf("%d\t%d\t%f\t%lf\t%lf\t%lf\n",
			size, n, e, loss, maxtime[0], maxtime[1]);
	MPI_Finalize();
	return 0;
}
