#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/unistd.h>
#include <linux/string.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/delay.h>
#include <linux/cdev.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/proc_fs.h>
#include <linux/fcntl.h>
#include <linux/uaccess.h>
#include "ioc_hw5.h"
#define DMA_BUFSIZE 64

#define DMASTUIDADDR 0x0
#define DMARWOKADDR 0x4
#define DMAIOCOKADDR 0x8
#define DMAIRQOKADDR 0xc
#define DMACOUNTADDR 0x10
#define DMAANSADDR 0x14
#define DMAREADABLEADDR 0x18
#define DMABLOCKADDR 0x1c
#define DMAOPCODEADDR 0x20
#define DMAOPERANDBADDR 0x21
#define DMAOPERANDCADDR 0x25

void *dma_buf;
static struct cdev *k_cdev;
static int Major;
static int Minor;

void myoutb (char* data, unsigned int port)
{
	memcpy(dma_buf+port,data, sizeof(char));
}
void myoutw (short* data, unsigned int port)
{
	memcpy(dma_buf+port,data, sizeof(short));
}
void myoutl (int* data, unsigned int port)
{
	memcpy(dma_buf+port,data, sizeof(int));
}
unsigned char myinb(unsigned int port)
{
	char *go;
	go = kzalloc(sizeof(char),GFP_KERNEL);
	memcpy(go,dma_buf+port, sizeof(char));
	return *go;
}
unsigned short myinw(unsigned int port)
{
	short *go;
	go = kzalloc(sizeof(short),GFP_KERNEL);
	memcpy(go,dma_buf+port, sizeof(short));
	return *go;
}
unsigned int myinl(unsigned int port)
{
	int *go;
	go = kzalloc(sizeof(int),GFP_KERNEL);
	memcpy(go,dma_buf+port, sizeof(int));
	return *go;
}

struct dataIn {
    char a;
    int b;
    short c;
};


int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}


void arithmetic_routine(struct work_struct *ws)
{

	char op;
	int a;
	short b;
   	int* ans;
	ans = kzalloc(sizeof(int),GFP_KERNEL);
  	int readable = 0;

printk("OS_HW5:%s(): Processing routine\n",__FUNCTION__);

    op = myinb(DMAOPCODEADDR);
    a  = myinl(DMAOPERANDBADDR);
    b  = myinw(DMAOPERANDCADDR);
    

    switch(op) {
        case '+':
            *ans = a+b;
            break;
        case '-':
            *ans = a-b;
            break;
        case '*':
            *ans = a*b;
            break;
        case '/':
            *ans = a/b;
            break;
        case 'p':
            *ans = prime(a, b);
            break;
        default:
            *ans = 0;
    }
    printk("OS_HW5:%s(): %d %c %d = %d\n",__FUNCTION__,a,op,b,*ans);
    myoutl(ans,DMAANSADDR);
    readable = 1;
    myoutl(&readable, DMAREADABLEADDR);
}



static ssize_t hwdev_read(int fops, int* ret, int n);
static ssize_t hwdev_write(int fops, struct dataIn* data, int n);
static int hwdev_ioctl(int filp, unsigned int cmd, unsigned int* args);
static int hwdev_open(void);
static int hwdev_release(void);
struct file_operations fops={
	.owner=THIS_MODULE,
	.read=hwdev_read,
	.write=hwdev_write,
	.unlocked_ioctl=hwdev_ioctl,
	.open=hwdev_open,
	.release=hwdev_release
};




static ssize_t hwdev_read(int fops, int* ret, int n)
{
int *ans;
int *init;
init = kzalloc(sizeof(int),GFP_KERNEL);
ans = kzalloc(sizeof(int),GFP_KERNEL);
*ans = myinl(DMAANSADDR);
copy_to_user(ret,ans,sizeof(int));
printk("OS_HW5:%s(): ans = %d\n",__FUNCTION__,*ans);
myoutl(init,DMAANSADDR);
myoutl(init,DMAREADABLEADDR);
return 1;
}

static ssize_t hwdev_write(int fops, struct dataIn* data, int n)
{
struct dataIn *tdata;
tdata = kzalloc(sizeof(struct dataIn),GFP_KERNEL);
struct work_struct *write_work = kzalloc(sizeof(typeof(*write_work)), GFP_KERNEL);
copy_from_user(tdata, data, sizeof(struct dataIn));
myoutb(&tdata->a, DMAOPCODEADDR);
myoutw(&tdata->c, DMAOPERANDCADDR);
myoutl(&tdata->b, DMAOPERANDBADDR);


INIT_WORK(write_work, arithmetic_routine);
schedule_work(write_work);
printk("OS_HW5:%s(): work queuing\n",__FUNCTION__);
if(myinl(DMABLOCKADDR) == 1)
{
printk("OS_HW5:%s(): block!\n",__FUNCTION__);
while(myinl(DMAREADABLEADDR)!=1)
{
	mdelay(10);
}
}
return 1;
}

static int hwdev_ioctl(int filp, unsigned int cmd, unsigned int *args)
{
	int ret = 0;
	int *temp;
	temp = kzalloc(sizeof(int),GFP_KERNEL);
	if(_IOC_TYPE(cmd)!= HW5_IOC_MAGIC)
		{
		printk("OS_HW5:%s(): Command Error\n",__FUNCTION__);
		return -ENOTTY;
		}
	if(_IOC_NR(cmd) > HW5_IOC_MAXNR)
		{
		printk("OS_HW5:%s(): MAXNR Error\n",__FUNCTION__);
		return -ENOTTY;
		}
		
		switch(cmd) {
		case HW5_IOCSETSTUID:
				ret = copy_from_user(temp,args,sizeof(int));
				if(*args == *temp)
				{
				myoutl(temp,DMASTUIDADDR);
				*temp = myinl(DMASTUIDADDR);
				printk("OS_HW5:%s(): My STUID is = %d\n",__FUNCTION__,*temp);
				}
				else
				{
					printk("OS_HW5:%s(): My STUID error\n",__FUNCTION__);
				}
				break;
		case HW5_IOCSETRWOK:
			        ret = copy_from_user(temp,args,sizeof(int));
				myoutl(temp,DMARWOKADDR);
				*temp = myinl(DMARWOKADDR);
				if(*temp==1)
				printk("OS_HW5:%s(): RW OK\n",__FUNCTION__);
				break;
		case HW5_IOCSETIOCOK:
				ret = copy_from_user(temp,args,sizeof(int));
				myoutl(temp,DMAIOCOKADDR);
				*temp = myinl(DMAIOCOKADDR);
				if(*temp==1)
				printk("OS_HW5:%s(): IOC OK\n",__FUNCTION__);
				break;
		case HW5_IOCSETIRQOK:
			        ret = copy_from_user(temp,args,sizeof(int));
				myoutl(temp,DMAIRQOKADDR);
				*temp = myinl(DMAIRQOKADDR);
				if(*temp==1)
				printk("OS_HW5:%s(): IRQ OK\n",__FUNCTION__);
				break;
		case HW5_IOCSETBLOCK:
				ret = copy_from_user(temp,args,sizeof(int));
				myoutl(temp,DMABLOCKADDR);
				*temp = myinl(DMABLOCKADDR);
				if(*temp==1)
				printk("OS_HW5:%s(): Blocking IO\n",__FUNCTION__);
		                else if(*temp == 0)
				printk("OS_HW5:%s(): Non-blocking IO\n",__FUNCTION__);
				break;
		case HW5_IOCWAITREADABLE:
				
				while(myinl(DMAREADABLEADDR)!=1)
				{
				 msleep(500);
				 printk("OS_HW5:%s(): wait readable 1\n",__FUNCTION__);
				}
				ret = myinl(DMAREADABLEADDR);
				copy_to_user(args,&ret,sizeof(args));
				break;
		
		}
	return 0;
}

static int hwdev_open(void)
{
printk("OS_HW5:%s():device open()\n",__FUNCTION__);
return 0;
}

static int hwdev_release(void)
{
printk("OS_HW5:%s():device release()\n",__FUNCTION__);
return 0;
}


static int hwdev_init(void)
{
dev_t devnum;
dev_t dev;
int result;

printk("OS_HW5:%s():........Start........\n",__FUNCTION__);
k_cdev = kzalloc(sizeof(struct cdev), GFP_KERNEL);

cdev_init(k_cdev,&fops);
k_cdev->owner = THIS_MODULE;
result = alloc_chrdev_region(&devnum,0,1,"mydev");
if(result < 0)
{
 printk("OS_HW5:%s():Major num alloc failed\n",__FUNCTION__);
 return result;
}

Major = MAJOR(devnum);
Minor = MINOR(devnum);
dev = MKDEV(Major,Minor);

result = cdev_add(k_cdev,dev,1);
if(result < 0)
{
printk(KERN_INFO "Unable to alloc cdev");
return result;
}
printk("OS_HW5:%s():register chrdev(%d,%d)\n",__FUNCTION__,Major,Minor);

dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);

if(dma_buf)
printk("OS_HW5:%s():allocate dma buffer success\n",__FUNCTION__);

else
printk("OS_HW5:%s():allocate dma buffer failed\n",__FUNCTION__);

return 0;

}

static void __exit hwdev_exit(void)
{
printk("OS_HW5:%s():device exit()\n",__FUNCTION__);
kfree(dma_buf);
printk("OS_HW5:%s():free dma buffer\n",__FUNCTION__);
cdev_del(k_cdev);
unregister_chrdev_region(Major,1);
printk("OS_HW5:%s():unregister chrdev\n",__FUNCTION__);
printk("OS_HW5:%s():........End........\n",__FUNCTION__);
}

MODULE_LICENSE("GPL");
module_init(hwdev_init);
module_exit(hwdev_exit);
