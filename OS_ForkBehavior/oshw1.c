#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <string.h>
int main(int argc, char* argv[])
{
pid_t child,waitsig;
int status;

printf("Fork now!\n");
child = fork();

if(child == 0)
{
     printf("My parent's PID is %d\n",getppid());
     execl(argv[1],NULL);
}
else
{

          do {
               waitsig = waitpid(-1, &status, WNOHANG);
                if (waitsig == -1)
	 	{
			printf("Child Process error!!\n");
                }
                 if (WIFEXITED(status))
		   {
		   printf("--------Child Exited--------\n");
                   printf("exited, status=%d\n", WEXITSTATUS(status));
                   }
		else if (WIFSIGNALED(status))
		   {
		   printf("--------Something Wrong--------\n");
                   printf("%s\n", strsignal(WTERMSIG(status)));
                   }
		else if (WIFSTOPPED(status))
		   {
                   printf("stopped by signal %d\n", WSTOPSIG(status));
                   }
               } while (!WIFEXITED(status) && !WIFSIGNALED(status));
}

return 0;
}
