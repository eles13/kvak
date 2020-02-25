all:
	g++ -fopenmp $^ -o $@
res: all
	for n in 20 24 28 30; do \
		for k in 1 4 $$n; do \
			for i in 1 2 4 8; do \
				bsub -W 15 -q normal -o out_$$k-$$n-$$i -e err_$$k-$$n-$$i ./main $$n $$k $$i; \
			done \
		done \
	done
