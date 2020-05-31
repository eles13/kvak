//
// Created by pe on 31.05.2020.
//

#ifndef ETC_QUANT_H
#define ETC_QUANT_H

#include <cmath>
#include <complex>
#include <cstdlib>
#include <mpi.h>
#include <stdio.h>
#include <omp.h>

typedef std::complex<double> complexd;


unsigned long long log_2(unsigned long long& m){
    unsigned long long log;
    for (log = 0; !((m >> log) & 1); ++log);
    return log;
}


complexd* gen(int n, int mode) {
    unsigned long long i, m = 1 << n;
    complexd *A = new complexd[m];
    double sum = 0;
    unsigned int seed = omp_get_wtime();
#pragma omp parallel for firstprivate(seed,mode) reduction(+: sum) schedule(guidedв)
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

complexd dot(complexd *A, complexd *B, int n) {
    unsigned long long m = (1LLU << n);
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
    return x;
}

complexd* f2(complexd *in, complexd U[4][4], int nqubits, int q1, int q2)
{
    int shift1=nqubits-q1;
    int shift2=nqubits-q2;
    int pow2q1=1<<(shift1);
    int pow2q2=1<<(shift2);
    int N=1<<nqubits;
    complexd* out = new complexd[nqubits];
#pragma omp parallel for shared(N, shift1, shift2, pow2q1, pow2q2) schedule(guided)
    for	(int i=0; i<N; i++)
    {
        int i00 = i & ~pow2q1 & ~pow2q2;
        int i01 = i & ~pow2q1 | pow2q2;
        int i10 = (i | pow2q1) & ~pow2q2;
        int i11 = i | pow2q1 | pow2q2;
        int iq1 = (i & pow2q1) >> shift1;
        int iq2 = (i & pow2q2) >> shift2;
        int iq=(iq1<<1)+iq2;
        out[i] = U[iq][(0<<1)+0] * in[i00] + U[iq][(0<<1)+1] * in[i01] + U[iq][(1<<1)+0] * in[i10] + U[iq][(1<<1)+1] * in[i11];
    }
    return out;
}

complexd* NOT(complexd *A,unsigned n, unsigned k, bool inplace = true, bool reverse = false) {
    complexd H[4];
    H[0] = 0;
    H[1] = 1;
    H[2] = 1;
    H[3] = 0;
    if(inplace) {
        A = f(A, n, H, k);
        return nullptr;
    }else{
        return f(A, n, H, k);
    }
}



complexd* ROT(complexd *A,unsigned n, unsigned k, double t, bool inplace = true, bool reverse = false) {
    complexd H[4];
    H[0] = 1;
    H[1] = 0;
    H[2] = 0;
    H[3] = exp(t);
    if (reverse){
        complexd temp(1,0);
        H[3] = temp/H[3];
    }
    if(inplace) {
        A = f(A, n, H, k);
        return nullptr;
    }else{
        return f(A, n, H, k);
    }
}

complexd* CNOT(complexd *A,unsigned n, unsigned k, unsigned p, bool inplace = true, bool reverse = false) {
    complexd U[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; i < 4; i++) {
            U[i][j] = 0;
        }
    }
    U[0][0] = 1;
    U[1][1] = 1;
    U[2][3] = 1;
    U[3][2] = 1;
    if (inplace){
        A = f2(A, U, n, k, p);
        return nullptr;
    }else{
        return f2(A, U, n, k, p);
    }
}


complexd* CROT(complexd *A,unsigned n, unsigned k, unsigned p, double t, bool inplace = true, bool reverse = false) {
    complexd U[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; i < 4; i++) {
            U[i][j] = 0;
        }
    }
    complexd a(0.0, 1.0);
    U[0][0] = 1;
    U[1][1] = 1;
    U[3][3] = exp(t);
    U[2][2] = 1;
    if (reverse){
        complexd temp(1,0);
        U[3][3] = temp/U[3][3];
    }
    if (inplace){
        A = f2(A, U, n, k, p);
        return nullptr;
    }else{
        return f2(A, U, n, k, p);
    }
}

complexd* blackbox_transform(int type, complexd* A, unsigned n, unsigned k, unsigned p = 0, double t = 0.0){
    complexd* buf;
    switch(type){
        case 0: {
            buf = NOT(A,n,k,false);
            NOT(buf,n,k,true,true);
            return buf;
        }
        case 1:{
            buf = ROT(A,n,k,t,false);
            ROT(buf,n,k,t,true, true);
            return buf;
        }
        case 2:{
            buf = CNOT(A,n,k, p,false);
            CNOT(buf,n,k, p,true,true);
            return buf;
        }
        case 3:{
            buf = CROT(A,n,k, p, t,false);
            CROT(buf,n,k,t, p,true, true);
            return buf;
        }
        default: return nullptr;
    }
}


#endif //ETC_QUANT_H
