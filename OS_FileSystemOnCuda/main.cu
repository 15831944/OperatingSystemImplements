#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
#define STORAGE_SIZE 1085440
#define MAX_FILE_SIZE 1048576

#define G_WRITE 991
#define G_READ  992
#define LS_S    993
#define LS_D    994
#define RM      995

typedef unsigned char uchar;
typedef uint32_t u32t;

typedef struct File_Control_Block{
 char filename[20];
 int size;
 int block_entry;
 int modified_time;
}u32;

__device__ __managed__ int nowfile = 0;
__device__ __managed__ int ticking_time = 0;
__device__ __managed__ int sz = 0,addr;
__device__ __managed__ int i,j,k,len,existed,check,valid,found,sp;
__device__ __managed__ uchar temp1,temp2,temp3,temp4,temp5,temp6;
__device__ __managed__ uchar *volume;

void load_bFile(char const* filename , uchar* input , u32t ssz)
{
	FILE *myfile;
	myfile = fopen(filename,"rb");

    fread(input,sizeof(uchar),ssz,myfile);

    fclose(myfile);
}

void write_bFile(char const* out , uchar* results , u32t input_size)
{
    FILE *myfile;

    myfile = fopen(out,"wb");

    fwrite(results,sizeof(uchar),input_size,myfile);

    fclose(myfile);
}

__device__ int strlens(char const* filename)
{
    sz = 0;
    for(i = 0 ; i < 20 ; i++)
    {
    sz++;
    if(filename[i]=='\0')
    break;
    }

    return sz;
}

__device__ u32 open(char const* filename, int ACCESS_MODE)
{
 u32 temp;
 if(ACCESS_MODE == G_WRITE)
 {
    check = 0;
    existed = -1;
    len = strlens(filename);
    for(i = 0 ; i < 1024 ; i++)
    {
       if(volume[i] == 1)
       {
        for(j = 0 ; j < len ; j++)
        {
            if(filename[j] == volume[1024 + i*26 + j])
               check++;
        }
        if(check == len)
        {
            existed = i;
        }
        check = 0;
        if(existed != -1)
          break;
       }
    }
    if(existed != -1)
    {

    len = strlens(filename);
    for( i = 0 ; i < len ; i++)
    {
    temp.filename[i] = filename[i];
    }
   

    nowfile = existed;
 
    temp.size = volume[1024 + nowfile*26 + 20]*256 + volume[1024 + nowfile*26 + 21];
    temp.block_entry = nowfile;
    temp.modified_time = volume[1024 + nowfile*26 + 24]*256 + volume[1024 + nowfile*26 + 25];


/*
    temp.size = 0;
    volume[1024 + nowfile*26 + 20] = temp.size/256;
    volume[1024 + nowfile*26 + 21] = temp.size%256;



    temp.block_entry = nowfile;
    volume[1024 + nowfile*26 + 22] = temp.block_entry/256;
    volume[1024 + nowfile*26 + 23] = temp.block_entry%256;

    for( i = 0 ; i < len ; i++)
    {
        volume[1024+nowfile*26+i] = temp.filename[i];
    }

    for(i = 0 ; i < 1024 ; i++)
    {
        volume[36864 + temp.block_entry*1024 + i] = 0;
    }

    ticking_time++;
    temp.modified_time = ticking_time;
    volume[1024 + nowfile*26 + 24] = temp.modified_time/256;
    volume[1024 + nowfile*26 + 25] = temp.modified_time%256;
*/

    return temp;
    }

    else{

    len = strlens(filename);
    for( i = 0 ; i < len ; i++)
    {
    temp.filename[i] = filename[i];
    }

   

    for(i = 0 ; i < 1024 ; i++)
    {
        if(volume[i] == 0)
        {
         nowfile = i;
         volume[i] = 1;
         break;
        }
    }
    //TODO:err_handle

    temp.size = 0;
    volume[1024 + nowfile*26 + 20] = 0;
    volume[1024 + nowfile*26 + 21] = 0;

    temp.block_entry = nowfile;
    volume[1024 + nowfile*26 + 22] = temp.block_entry/256;
    volume[1024 + nowfile*26 + 23] = temp.block_entry%256;



    for( i = 0 ; i < len ; i++)
    {
        volume[1024+nowfile*26+i] = temp.filename[i];
    }

    ticking_time++;
    temp.modified_time = ticking_time;
    volume[1024 + nowfile*26 + 24] = temp.modified_time/256;
    volume[1024 + nowfile*26 + 25] = temp.modified_time%256;
    
    return temp;
    }
 }
 else if(ACCESS_MODE == G_READ)
 {
    check = 0;
    existed = -1;
    len = strlens(filename);
    for(i = 0 ; i < 1024 ; i++)
    {
       if(volume[i] == 1)
       {
        for(j = 0 ; j < len ; j++)
        {
            if(filename[j] == volume[1024 + i*26 + j])
               check++;
        }
        if(check == len)
        {
            existed = i;
        }
        check = 0;
       }
    }
    if(existed != -1)
    {


    for( i = 0 ; i < len ; i++)
    {
    temp.filename[i] = filename[i];
    }

    nowfile = existed;

    temp.size = volume[1024 + nowfile*26 +20]*256+ volume[1024 + nowfile*26 + 21];

    temp.block_entry = nowfile;

    temp.modified_time = volume[1024 + nowfile*26 + 24]*256 + volume[1024 + nowfile*26 + 25];

    return temp;

    }

    else
    {
      //TODO:err handle
      printf("No such file can be read\n");
    }
 }

}

__device__ void write(uchar *startPlace, int data_size, u32 file_pointer)
{
  for(i = 0 ; i < file_pointer.size ; i++)
  {
    volume[36864 + file_pointer.block_entry*1024 + i] = 0;
  }

  for(i = 0 ; i < data_size ; i++)
  {
    volume[36864 + file_pointer.block_entry*1024 + i] = *(startPlace+i);
  }
    file_pointer.size = data_size;
    volume[1024 + file_pointer.block_entry*26 + 20] = file_pointer.size/256;
    volume[1024 + file_pointer.block_entry*26 + 21] = file_pointer.size%256;

    ticking_time++;
    file_pointer.modified_time = ticking_time;
    volume[1024 + file_pointer.block_entry*26 + 24] = file_pointer.modified_time/256;
    volume[1024 + file_pointer.block_entry*26 + 25] = file_pointer.modified_time%256;

}

__device__ void read(uchar *startPlace, int data_size, u32 file_pointer)
{
  for(i = 0 ; i < data_size ; i++)
  {
    *(startPlace+i) = volume[36864 + file_pointer.block_entry*1024 + i];
  }
}

__device__ void gsys(int OUTPUT_MODE)
{
    if(OUTPUT_MODE == LS_D)
    {
     printf("===sort by modified time===\n");

     for(i = ticking_time ; i > 0 ; i--)
     {
        for(j = 0 ; j < 1024 ; j++)
        {
            if(volume[1024 + j*26 + 20] != 255)
            {
                if(i == (volume[1024 + j*26 + 24]*256+ volume[1024 + j*26 + 25]))
                {
                    for(k = 0 ; k < 20 ; k++)
                    {
                        if(volume[1024 + j*26 + k] == '\0')
                            break;

                        printf("%c",volume[1024 + j*26 + k]);
                    }
                    printf("\n");
                }
            }
        }
     }

    }
    else if(OUTPUT_MODE == LS_S)
    {
     printf("===sort by file size===\n");
     valid = 0;
     for( i = 0 ; i < 1024 ; i++ )
     {
         if( volume[1024 + i*26 + 20] != 255 )
         {

                volume[27648 + valid*6 ] = volume[ 1024 + i*26 + 20];
                volume[27648 + valid*6 + 1] = volume[ 1024 + i*26 + 21];
                volume[27648 + valid*6 + 2] = volume[ 1024 + i*26 + 24];
                volume[27648 + valid*6 + 3] = volume[ 1024 + i*26 + 25];
                volume[27648 + valid*6 + 4] = i/256;
                volume[27648 + valid*6 + 5] = i%256;
                valid++;
         }
     }
        for(i = valid-1 ; i >0 ; i--)
        {
            sp=1;
           for( j = 0 ; j <= i ; j++)
           {
               if( ((volume[27648 + j*6]*256) + (volume[27648 + j*6 + 1])) < ((volume[27648 + (j+1)*6]*256) + (volume[27648 + (j+1)*6 + 1])) )
               {
                 temp1 = volume[27648 + j*6];
                 temp2 = volume[27648 + j*6 + 1 ];
                 temp3 = volume[27648 + j*6 + 2 ];
                 temp4 = volume[27648 + j*6 + 3 ];
                 temp5 = volume[27648 + j*6 + 4 ];
                 temp6 = volume[27648 + j*6 + 5 ];

                 volume[27648 + j*6] = volume[27648 + (j+1)*6];
                 volume[27648 + j*6 + 1 ] = volume[27648 + (j+1)*6 + 1];
                 volume[27648 + j*6 + 2 ] = volume[27648 + (j+1)*6 + 2];
                 volume[27648 + j*6 + 3 ] = volume[27648 + (j+1)*6 + 3];
                 volume[27648 + j*6 + 4 ] = volume[27648 + (j+1)*6 + 4];
                 volume[27648 + j*6 + 5 ] = volume[27648 + (j+1)*6 + 5];

                 volume[27648 + (j+1)*6] = temp1;
                 volume[27648 + (j+1)*6 + 1] = temp2;
                 volume[27648 + (j+1)*6 + 2] = temp3;
                 volume[27648 + (j+1)*6 + 3] = temp4;
                 volume[27648 + (j+1)*6 + 4] = temp5;
                 volume[27648 + (j+1)*6 + 5] = temp6;
                 sp=0;
               }
               else if ( ((volume[27648 + j*6]*256) + (volume[27648 + j*6 + 1])) == ((volume[27648 + (j+1)*6]*256) + (volume[27648 + (j+1)*6 + 1])) )
               {
                 if( ((volume[27648 + j*6 + 2]*256) + (volume[27648 + j*6 + 3])) < ((volume[27648 + (j+1)*6 + 2]*256) + (volume[27648 + (j+1)*6 + 3])) )
                 {
                 temp1 = volume[27648 + j*6];
                 temp2 = volume[27648 + j*6 + 1 ];
                 temp3 = volume[27648 + j*6 + 2 ];
                 temp4 = volume[27648 + j*6 + 3 ];
                 temp5 = volume[27648 + j*6 + 4 ];
                 temp6 = volume[27648 + j*6 + 5 ];

                 volume[27648 + j*6] = volume[27648 + (j+1)*6];
                 volume[27648 + j*6 + 1 ] = volume[27648 + (j+1)*6 + 1];
                 volume[27648 + j*6 + 2 ] = volume[27648 + (j+1)*6 + 2];
                 volume[27648 + j*6 + 3 ] = volume[27648 + (j+1)*6 + 3];
                 volume[27648 + j*6 + 4 ] = volume[27648 + (j+1)*6 + 4];
                 volume[27648 + j*6 + 5 ] = volume[27648 + (j+1)*6 + 5];

                 volume[27648 + (j+1)*6] = temp1;
                 volume[27648 + (j+1)*6 + 1] = temp2;
                 volume[27648 + (j+1)*6 + 2] = temp3;
                 volume[27648 + (j+1)*6 + 3] = temp4;
                 volume[27648 + (j+1)*6 + 4] = temp5;
                 volume[27648 + (j+1)*6 + 5] = temp6;
                 }
                 sp=0;
               }
           }
            if(sp==1) break;
        }

        for(i = 0 ; i < valid ; i++)
        {
        addr = volume[27648 + i*6 + 4] * 256 + volume[27648 + i*6 + 5];
        for ( j = 0 ; j < 20 ; j++ )
        {
         if(volume[1024 + addr*26 + j] != '\0')
         {
            printf("%c",volume[1024 + addr*26 + j]);
         }
        }
            printf(" ");
            sz = volume[1024 + addr*26 + 20]*256 + volume[1024 + addr*26 + 21];
            printf("%d\n",sz);
        }

    }


}

__device__ void gsys(int DEL_INFO, char const* filename)
{
    len = strlens(filename);
    check = 0;
    found = 0;
    if(DEL_INFO == RM)
    {
    for(i = 0; i < 1024 ;i++)
        {
        if(volume[1024 + i*26 + 20] != 255)
            {
            for(j = 0 ; j < len ; j++)
                {
                    if(volume[1024 + i*26 + j] == filename[j])
                       check++;
                }
                if(check == len)
                {
                    found = 1;
                    int entry = volume[1024 + i*26 + 22]*256 + volume[1024 + i*26 +23];
                    volume[entry] = 0;

                    for(k = 0 ; k < 26 ; k++)
                      volume[1024 + i*26 + k] = 0;

                      volume[1024 + i*26 + 20] = 255;

                    for(k = 0 ; k < 1024 ; k++)
                      volume[36864 + entry*1024 + i] = 0;
                }
                check = 0;
            }
            if(found)
                break;
        }
    }
    else{
        printf("No such command!\n");
    }
}
__global__ void mykernel(uchar *input, uchar *output)
{
    //####kernel start####
   
    u32 fp = open("t.txt\0", G_WRITE);
    write(input, 64, fp);
    fp = open("b.txt\0", G_WRITE);
    write(input+32, 32, fp);
    fp = open("t.txt\0", G_WRITE);
    write(input+32, 32, fp);

    fp = open("t.txt\0", G_READ);
    read(output, 32, fp);
    gsys(LS_D);
    gsys(LS_S);
    fp = open("b.txt\0", G_WRITE);
    write(input+64, 12, fp);
    gsys(LS_S);
    gsys(LS_D);
    gsys(RM, "t.txt\0");
    gsys(LS_S);
    //####kernel end####

}

void init_volume()
{
 memset( volume, 0, STORAGE_SIZE );

 for( int m = 0 ; m < 1024 ; m++)
    volume[ 1024 + m*26 + 20 ] = 255;

}



int main()
{

    cudaMallocManaged(&volume,STORAGE_SIZE);
    init_volume();

    uchar *input, *output;
    cudaMallocManaged(&input,MAX_FILE_SIZE);
    cudaMallocManaged(&output, MAX_FILE_SIZE);

    for(u32t m = 0 ; m < MAX_FILE_SIZE ; m++)
        output[m] = 0;

    load_bFile(DATAFILE, input, MAX_FILE_SIZE);

    cudaSetDevice(1);

    mykernel<<<1, 1>>>(input,output);
    cudaDeviceSynchronize();
    
    write_bFile(OUTFILE, output, MAX_FILE_SIZE);
	cudaDeviceReset();

    return 0;
}
