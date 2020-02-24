all:
	g++ -fopenmp $^ -o $@
res: main
	for n in 8 16 32; do \
		for k in 1 4; do \
			for i in 1 2 4 8 16; do \
				bsub -W 15 -q normal -o out_$$k-$$n-$$i -e err_$$k-$$n-$$i ./main $$n $$k $$i; \
			done \
		done \
	done
