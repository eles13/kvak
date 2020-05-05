SHELL := /bin/bash
#MAKEFLAGS += --silent
N := 25 26 27 28
K := 01 04
I := 01 02 04 08

all: main
mainp: main.cpp
	mpixlC $^ -o $@
main:
    mpic++ $^ -o $@ -lm
submit: mainp
	for n in $(N); do \
		for k in $(K) $$n; do \
			for i in $(I); do \
				mpisubmit.pl -p $$i -w 00:30 -o out_$$k-$$n-$$i -e err_$$k-$$n-$$i ./main -- $$n $$k ; \
			done \
		done \
	done
clean:
	rm main out_* err_*
test: main
    mpirun -n 1 main 10 4 out.txt
    diff -e -q out.txt eq.txt
    echo Success