#include	<crtdbg.h>

#include	<oreore/container/PriorityQueue.h>
using namespace OreOreLib;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	tcout << _T("//================ Test Priority Queue =================//\n");


	PriorityQueue<int> pqueue;

	tcout << _T("pqueue.Init(6);\n");
	pqueue.Init(6);

	pqueue.Display();
	tcout<< tendl;

	tcout << _T("Enqueue 9 and 3.\n");
	pqueue.Enqueue(9);
	pqueue.Enqueue(3);

	pqueue.Display();
	tcout<< tendl;

	tcout << _T("pqueue.Extend(3);\n");
	pqueue.Extend(3);

	pqueue.Display();
	tcout<< tendl;

	tcout << _T("pqueue.Enqueue(6);\n");
	pqueue.Enqueue(6);

	pqueue.Display();
	tcout<< tendl;

	tcout << _T("pqueue.Shrink(6);\n");
	pqueue.Shrink(6);

	pqueue.Display();
	tcout<< tendl;

	for( int i=0; i<3; ++i )
	{
		auto val = pqueue.Dequeue();
		tcout << _T("Dequeue: ") << val << tendl;
	}

	pqueue.Display();
	tcout<< tendl;

	return 0;
}