#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#define NUM_THREADS 3

int TCOUNT = 0 ;
int COUNT_LIMIT = 0;
int count = 0;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;
int primes[1000]={0};
int pflag = 0;
void *Watch_Count(void *t)
{
    int tid = (int)t;
    printf("Start Watch_Count():thread %d\n",tid);

    pthread_mutex_lock(&count_mutex);
    while(count < COUNT_LIMIT){
        printf("Watch_Count():thread %d p=%d. Going into wait...\n",tid,count);
        pthread_cond_wait(&count_threshold_cv, &count_mutex);
        printf("Watch_Count(): thread %d Condition signal received.\n", tid);
        printf("Watch_Count(): thread %d Updating the value of p...\n", tid);
        printf("The latest prime found before p = %d.\n",primes[pflag-2]);
        count += primes[pflag-2];
        printf("Watch_Count(): thread %d p now = %d.\n", tid, count);
    }
    printf("Watch_Count(): thread %d Unlocking Mutex\n",tid);
    pthread_mutex_unlock(&count_mutex);

    pthread_exit(NULL);
}

void *Prime_Count(void *t)
{
    int j,i;
    double result = 0.0;

    int tid = (int)t;

    while( count < TCOUNT )
     {
        if(count!=COUNT_LIMIT)
        {

         pthread_mutex_lock(&count_mutex);
         count++;
         printf("Prime_Count(): thread %d, p = %d\n",tid,count);
         int isprime = 0;
         int check = 0;
         for(j = 1 ; j <= count ; j++)
         {
             if(count%j==0)
                check++;
         }
         if(check == 2)
            isprime = 1;

         if(isprime)
              {
              primes[pflag]=count;
              pflag++;
              printf("Prime_Count(): thread %d, find prime = %d\n",tid,count);

             if (count == COUNT_LIMIT)
             {
              printf("Prime_Count(): thread %d, p = %d Prime Reached. ",tid,count);
              pthread_cond_signal(&count_threshold_cv);
              printf("Just sent signal.\n");
             }

              }


             pthread_mutex_unlock(&count_mutex);

             sleep(2);
        }
     }

        pthread_exit(NULL);
}


int main(int argc, char* argv[])
{
pthread_t threads[NUM_THREADS];
TCOUNT = atoi(argv[1]); COUNT_LIMIT = atoi(argv[2]);
int i,rc;
pthread_mutex_init(&count_mutex,NULL);
pthread_cond_init(&count_threshold_cv,NULL);

int t1=1,t2=2,t3=3;
pthread_create(&threads[0],NULL,Watch_Count,(void*)t1);
 pthread_create(&threads[1],NULL,Prime_Count,(void*)t2);
  pthread_create(&threads[2],NULL,Prime_Count,(void*)t3);

for(i = 0 ; i < NUM_THREADS ; i++)
{
    pthread_join(threads[i],NULL);
}
printf ("Main(): Waited on %d threads. Final value of count = %d. Done.\n",NUM_THREADS, count);

 pthread_mutex_destroy(&count_mutex);
 pthread_cond_destroy(&count_threshold_cv);
 pthread_exit (NULL);
return 0;
}
