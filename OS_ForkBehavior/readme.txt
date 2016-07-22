For HW1-1:
{
To compile just type in "gcc oshw1.c" or "make"
the execution file is a1_1.out
the usage of this file is to fork a child process in user mode, while it's parent process waiting for its signal
Command for this file: ./a1_1.out "Execution file for child process"
There are 3 conditions:
1.Exited normally, output"Exited, status = "# of status""
2.Exited abnormally, output the meaning of signal returned.
3.Stopped, output "Stopped" , basically this won't happen.
}