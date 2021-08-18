#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/container/RingQueue.h>
using namespace OreOreLib;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	RingQueue<int> queue;

	tcout << _T("//============== Extend/Shrink with rear < front case =============//\n");

	tcout << _T("queue.Init(8);\n");
	queue.Init(8);
	tcout << tendl;

	queue.Display();
	tcout << tendl;

	for( int i=0; i<8; ++i )
	{
		tcout << _T("Enqueue: ") << i << tendl;
		queue.Enqueue(i);
	}
	tcout << tendl;

	queue.Display();
	tcout << tendl;

	for( int i=0; i<4; ++i )
	{
		auto val = queue.Dequeue();
		tcout << _T("Dequeue: ") << val << tendl;
	}
	tcout << tendl;

	queue.Display();
	tcout << tendl;


	tcout << _T("queue.Enqueue(-2);\n");
	queue.Enqueue(-2);
	tcout << tendl;

	tcout << _T("queue.Extend(6);\n");
	queue.Extend(6);
	tcout << tendl;

	tcout << _T("queue.Enqueue(-9999);\n");
	queue.Enqueue(-9999);
	tcout << tendl;

	queue.Display();
	tcout << tendl;

	tcout << _T("queue.Shrink(2);\n");
	queue.Shrink(2);
	tcout << tendl;

	queue.Display();
	tcout << tendl;



	tcout << _T("//============== Extend/Shrink with front < rear case =============//\n");

	queue.Init(8);

	for( int i=0; i<7; ++i )
	{
		tcout << _T("Enqueue: ") << i << tendl;
		queue.Enqueue(i);
	}

	queue.Display();
	tcout << tendl;


	for( int i=0; i<3; ++i )
	{
		auto val = queue.Dequeue();
		tcout << _T("Dequeue: ") << val << tendl;
	}

	tcout << _T("queue.Extend(6);\n");
	queue.Extend(6);
	tcout << tendl;

	tcout << _T("queue.Enqueue(-9999);\n");
	queue.Enqueue(-9999);
	tcout << tendl;

	queue.Display();
	tcout << tendl;

	queue.Shrink(2);
	tcout << tendl;

	queue.Display();
	tcout << tendl;


	return 0;
}