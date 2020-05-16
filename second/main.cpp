#include <cmath>
#include <complex>
#include <cstdlib>
#include <mpi.h>
#include <stdio.h>

typedef std::complex<double> complexd;

int rank, size, size_2n;

complexd *generate(int n, int argc) {
  unsigned long long m = 1LLU << (n - size_2n);
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
  unsigned long long m = 1LLU << (*n - size_2n);
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
  unsigned long long m = 1LLU << (n - size_2n);
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

complexd *f(complexd *A, int n, int k, complexd *H) {
  unsigned long long m = (1LLU << n) / size;
  complexd *B = new complexd[m];
  int rank_ = ((rank * m) ^ (1LLU << (k - 1))) / m;
  if (rank != rank_) {
    MPI_Sendrecv(A, m, MPI_DOUBLE_COMPLEX, rank_, 0, B, m, MPI_DOUBLE_COMPLEX,
                 rank_, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank < rank_) {
      for (unsigned long long i = 0; i < m; i++) {
        B[i] = H[0] * A[i] + H[1] * B[i]
      }
    } else {
      for (unsigned long long i = 0; i < m; i++) {
        B[i] = H[2] * B[i] + H[3] * A[i]
      }
    }
  } else {
    unsigned long long l = 1LLU << (int)log2(m) - k;
    for (unsigned long long i = 0; i < m; i++) {
      B[i] = ((i & l) >> ((int)log2(m) - k) == 0)
                 ? H[0] * A[i & ~l] + H[1] * A[i | l]
                 : H[2] * A[i & ~l] + H[3] * A[i | l];
    }
  }
  return B;
}

int main(int argc, char **argv) {
  int n, k;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (argc < 3) {
    MPI_Finalize();
    return 1;
  }
  n = atoi(argv[1]);
  k = atoi(argv[2]);
  for (size_2n = 0; !((size >> size_2n) & 1); ++size_2n)
    ;
  double time, maxtime;
  MPI_Barrier(MPI_COMM_WORLD);
  complexd *A = (argc > 4) ? read(argv[3], &n) : generate(n, argc);
  complexd H[] = {1 / sqrt(2), 1 / sqrt(2), 1 / sqrt(2), -1 / sqrt(2)};
  MPI_Barrier(MPI_COMM_WORLD);
  time = MPI_Wtime();
  complexd *B = f(A, n, k, H);
  time = MPI_Wtime() - time;
  if (argc > 4) {
    write(argv[4], B, n);
  } else if (argc == 4) {
    write(argv[3], B, n);
  }
  MPI_Reduce(&time, &maxtime, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printf("%d\t%d\t%d\t%lf\n", n, k, size, maxtime);
  delete[] B;
  MPI_Finalize();
  return 0;
}
