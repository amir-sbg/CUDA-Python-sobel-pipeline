.PHONY: configure build run clean

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --config Release --parallel

run: build
	./build/cuda_saxpy

clean:
	rm -rf build
