CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Iinclude -Wall -Wextra

all: libshunt.so scalability_bench

# Shared library loaded from Python via ctypes (shunt._native).
libshunt.so: capi.cpp include/shunt_core.hpp
	$(CXX) $(CXXFLAGS) -shared -fPIC capi.cpp -o $@

# Fig. 19 microbenchmark.
scalability_bench: scalability_bench.cpp include/shunt_core.hpp
	$(CXX) $(CXXFLAGS) scalability_bench.cpp -o $@

clean:
	rm -f libshunt.so scalability_bench

.PHONY: all clean
