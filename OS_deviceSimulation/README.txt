100033202 Lin Che Yu OS_HW5 device simulation
How to test:
First type 'make' in linux terminal, the makefile will come out ko file and insert into module
Then we should type in dmesg to check where the main.c register the device
Type in 'sudo mknod /dev/mydev c 250 0' ( 250 0 for example
Type in 'sudo chmod 666 /dev/mydev'
Then you can run './test' to check if the virtual device runs well
After the check, type in 'make clean' to check the dmesg as well!
Happy lunar new year!
 
