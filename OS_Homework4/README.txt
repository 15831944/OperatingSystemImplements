The code is formed for GPUmemory-based simple file system implement!

Please type in "make" command to compile the program

or use the command "nvcc -g -G -arch=sm_30 main.cu -o fs" to compile the code

type in "./fs" to run the program

the program will come out a output binary file named "snapshot.bin"

Methods: seperate the big volume space into several blocks

Superblocks : 1024 bytes, to remember whether each block is in use or not (1024byte per block, maximum size of a file)
FCBs * 1024: 20bytes for filename, 2bytes for size, 2bytes for block_entry, 2bytes for modified_time
Swapping and sorting space: 36864-1024-1024*26

FileSpace: 1048576 bytes
