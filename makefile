test:
	icc -openmp -I ../eigen cd-svm.cpp -o cd-svm
clean:
	rm -rf cd-svm