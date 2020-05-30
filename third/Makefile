I := 60
N := 24 25 26 27 28
P := 08
T := 01 02 04 08
E := 0.01

all: mainp
main:
    mpic++ main.cpp -fopenmp -o main
mainp:
	mpixlC -O3 -fopenmp main.cpp -o main
submit: mainp
	for n in $(N); do \
		for p in $(P); do \
		    for t in $(T); do \
                for ((i=0; i < $(I); ++i)) ; do \
                    mpisubmit.pl -p $$p -w 1:0  \
                    --stdout out_$$n-$$p-$$t-$$i --stderr err_$$n-$$p-$$t-$$i \
                    ./main -- $$n $$e ;\
			    done \
		    done \
	    done \
	done
test: main
    mpirun -n 1 main 10 0.01 out.txt
    diff -e -q out.txt eq.txt
    echo Success
