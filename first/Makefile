all:
	g++ -fopenmp main.cpp -o main -O2
res: all
	for n in 20 24 28 30; do \
		for k in 1 4 $$n; do \
			for i in 1 2 4 8; do \
				bsub -W 15 -q normal -o out_$$k-$$n-$$i -e err_$$k-$$n-$$i ./main $$n $$k $$i 0; \
			done \
		done \
	done
test: all
	./main 20 1 1 1
	diff -e -q B.txt etal.txt
	echo Success
