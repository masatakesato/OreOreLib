// Threadpool.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include "ThreadPool.h"

#define LOOPCOUNT 20000

class CMyWorkThread : public CWorkThread
{
public:
	unsigned virtual RunProcess(void *Param)
	{
		//printf("param = %d\n", (int)Param);

		for(int i=0;i<10;i++)
		{
			float x=4;
			float y=5;
			float z=34;
			x= x*y/z;
			z=x+34*z+y/x;
			x=y*z+y/z;
		}
		
		return(0);


		// テクスチャをロードする
		// タイルキャッシュを部分更新する
		// キャッシュステータスを更新する
		
	}
};



unsigned _stdcall ThreadProc1(void *Param)
{
	for(int i=0;i<10;i++)
	{
		float x=4;
		float y=5;
		float z=34;
		x= x*y/z;
		z=x+34*z+y/x;
		x=y*z+y/z;
	}
	//printf("end?.............\n");
	return(0);
}


int main(int argc, char* argv[])
{
	unsigned int Thrdaddr;
	int i;
	int startTime;
	int endTime;
	HANDLE hEndEvent = CreateEvent(NULL, FALSE, TRUE, L"End Event");
	HANDLE hHandleArray[LOOPCOUNT] = {0};
	HANDLE hSmallHandleArray[4] = {0};

	startTime = GetCurrentTime ();
	ThreadPool* myPool = new ThreadPool(4);

	for(i=0;i<LOOPCOUNT;i++)
	{
		CMyWorkThread* myThread = new CMyWorkThread();
		myPool->SubmitJob(myThread, (void *)i);
	}
	
	myPool->DestroyPool();
	endTime = GetCurrentTime ();
	printf( "Thread Pool Method Elapsed Time:\t%d ms\n", endTime-startTime );

	ResetEvent(hEndEvent);
	startTime = GetCurrentTime ();
	for(int i=0; i<LOOPCOUNT; i+=4)
	{
		hSmallHandleArray[0] = (HANDLE) _beginthreadex( NULL, 0, ThreadProc1, (void*)&i, 0, &Thrdaddr);
		hSmallHandleArray[1] = (HANDLE) _beginthreadex( NULL, 0, ThreadProc1, (void*)&i, 0, &Thrdaddr);
		hSmallHandleArray[2] = (HANDLE) _beginthreadex( NULL, 0, ThreadProc1, (void*)&i, 0, &Thrdaddr);
		hSmallHandleArray[3] = (HANDLE) _beginthreadex( NULL, 0, ThreadProc1, (void*)&i, 0, &Thrdaddr);
		WaitForMultipleObjects(4, hSmallHandleArray, TRUE, INFINITE);
	}
	endTime = GetCurrentTime ();
	printf( "Four Threads Method Elapsed Time:\t%d ms\n", endTime-startTime );

	//No bounds on threads
	startTime = GetCurrentTime ();
	for(i=0;i<LOOPCOUNT;i++){
		hHandleArray[i] = (HANDLE) _beginthreadex( NULL, 0, ThreadProc1, (void*)&i, 0, &Thrdaddr);
	}
	WaitForMultipleObjects(LOOPCOUNT,hHandleArray,TRUE,INFINITE);
	endTime = GetCurrentTime ();
	printf( "Unbound Method Elapsed Time:\t\t%d ms\n", endTime-startTime );
	
	return 0;
}

