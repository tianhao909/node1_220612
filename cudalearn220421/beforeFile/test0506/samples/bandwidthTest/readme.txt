Sample: bandwidthTest
Minimum spec: SM 3.5

This is a simple test program to measure the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e.  This test application is capable of measuring device to device copy bandwidth, host to device copy bandwidth for pageable and page-locked memory, and device to host copy bandwidth for pageable and page-locked memory.

Key concepts:
CUDA Streams and Events
Performance Strategies

//fth nvcc bandwidthTest.cu -I /usr/local/cuda-11.3/samples/common/inc/
