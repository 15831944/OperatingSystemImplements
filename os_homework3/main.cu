#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PAGESIZE 32
#define PHYSICAL_MEM_SIZE 32768
#define STORAGE_SIZE 131072

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

int load_bFile(char* filename, uchar* input, int ssz)
{
	int sz=0;
	FILE *myfile;
	myfile = fopen(filename,"rb");
 while(1==fread(input+sz*sizeof(uchar),sizeof(uchar),1,myfile))
         {
             sz++;
         }

    fclose(myfile);

    return sz;
}

void write_bFile(char* out , uchar* results , int input_size)
{
    FILE *myfile;
    int i;
    myfile = fopen(out,"wb");
    
    for(i = 0 ; i < input_size ; i++)
   {
      fputc(results[i],myfile);
   }

   fclose(myfile);
}

__device__ __managed__ int PAGE_ENTRIES = 0;

__device__ __managed__ int PAGEFAULT = 0;

__device__ __managed__ uchar storage[STORAGE_SIZE];

__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

__device__ __managed__ int pt_entries = PHYSICAL_MEM_SIZE/PAGESIZE;

extern __shared__ u32 pt[];

__device__ void init_pageTable(int pt_entries)
{
	int i;
   for( i = 0; i < pt_entries ; i++)
  {
     pt[i]=5555;        //i is frame number, pt[i] is which page in this frame
     pt[i+pt_entries]='i';   // valid bit
     pt[i+pt_entries*2]=0;   //LRU counter
  }
}

__device__ void swap_block(uchar *phy, uchar *log, int are,int frame)   //swap needy page in storage and swap the victim back when pagefault occurs.
{
      int i,k;
      for(i = 0,k = are * PAGESIZE ; i < PAGESIZE ; k++ , i++)
      {
        phy[k]=log[frame*PAGESIZE+i];
        log[frame*PAGESIZE+i]='n';
      }     

}

 __device__ void swap_out(uchar *phy, uchar *log, int victim,int frame)   //swap needy page in storage and swap the victim back when pagefault occurs.
{      
        int i,k;
        for(i = 0,k = victim * PAGESIZE ; i < PAGESIZE ; k++ , i++)
        {
          log[frame*PAGESIZE+i]=phy[k];
          phy[k]='n';
       }
}

__device__ u32 paging(uchar *buffer, u32 frame_num, u32 offset)
{
   int flag = 0;
   int ad = 0;
   int i;
   for(i=0 ; i < pt_entries ; i++)
   {
      if(pt[i]==frame_num)     //Needy page is in physical memory
      {
           flag=1;
           ad=i;
           break;
      }
      else if(pt[i+pt_entries]=='i')   //This page is free for swapping in the page we need.
      {
           PAGEFAULT++;     
           flag=2;
           ad=i;
           break;
      }
   }   

   if(flag==1)
   {
      pt[ad+2*pt_entries]++;         //LRU counter +1, return the address in physical memory.
      ad=(ad*PAGESIZE)+offset;
   }
   else if(flag==2)
   {
      pt[ad+pt_entries]='v';         //swap in the page we need (in storage) and update page table
      pt[ad]=frame_num;
      swap_block(buffer,storage,ad,frame_num);   
      pt[ad+pt_entries*2]++;          //LRU counter +1, return the address in physical memory.
      ad=(ad*PAGESIZE)+offset;
   }
   else
   {
      int min=pt[pt_entries-1+pt_entries*2];    //set the last LRU counter as a start point
      int victim=0;
      for(i=pt_entries-1;i>=0;i--)        //Finding the LRU page from bottom to up according to LRU counter value.
      {
         if(pt[i+2*pt_entries]<=min)
         {  
           min=pt[i+2*pt_entries];
           victim=i;
         }
      }
      swap_out(buffer,storage,victim,pt[victim]);
      pt[victim+pt_entries]='i';   //set bit to invalid
      ad=paging(buffer,frame_num,offset);
   }
   return ad;
}

__device__ uchar Gread(uchar *buffer, u32 addr)
{
	u32 frame_num = addr/PAGESIZE;
	u32 offset    = addr%PAGESIZE;

	addr = paging(buffer, frame_num, offset);
	return buffer[addr];
}

__device__ void Gwrite(uchar *buffer, u32 addr, uchar value)
{
	u32 frame_num = addr/PAGESIZE;
	u32 offset    = addr%PAGESIZE;

	addr = paging(buffer, frame_num, offset);
	buffer[addr] = value;
}

__device__ void snapshot(uchar *results, uchar* buffer, int offset, int input_size)
{
	for(int i = 0 ; i < input_size ; i++)
		results[i] = Gread(buffer, i+offset);
}

__global__ void mykernel(int input_size)
{
	__shared__ uchar data[PHYSICAL_MEM_SIZE];


	init_pageTable(pt_entries);

	//####Gwrite/Gread code section start####
	for(int i = 0; i < input_size ; i++)
		Gwrite(data, i , input[i]);

	for(int i = input_size -1 ; i >= input_size - 10 ; i--)
		int value = Gread(data,i);

		snapshot(results, data,0, input_size);
	//####Gwrite/Gread code section end####
	printf("pagefault times = %d\n", PAGEFAULT);

}

int main()
{
	
	int input_size = load_bFile(DATAFILE, input, STORAGE_SIZE);
	
	cudaSetDevice(2);
	mykernel<<<1, 1, 16384>>>(input_size);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	write_bFile(OUTFILE, results, input_size);

	return 0;
}