SHELL := /bin/bash

compile:
	nvcc kernel.cu -o gray

test:
	sh test.sh
