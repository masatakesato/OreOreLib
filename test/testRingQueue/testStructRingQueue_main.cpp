#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/container/RingQueue.h>
using namespace OreOreLib;


struct Int
{
	Int()
		: pVal( new int() )
	{
		tcout << _T("Int::Int()...\n");	
	}


	Int( int i )
		: pVal( new int(i) )
	{
		tcout << _T("Int::Int( int i )...\n");	
	}


	~Int()
	{
		tcout << _T("Int::~Int()...\n");	
		SafeDelete( pVal );
	}


	Int( const Int& obj )
		: pVal( new int(*obj.pVal) )
	{
		tcout << _T("Int::Int( const Int& obj )...\n");
	}


	Int( Int&& obj )
		: pVal( obj.pVal )
	{
		tcout << _T("Int::Int( Int&& obj )...\n");
		obj.pVal = nullptr;
	}


	Int& operator=( const Int& obj )
	{
		tcout << _T("Int::Int& operator=( const Int& obj )...\n");
		if( this != &obj )
		{
			SafeDelete( pVal );

			pVal = new int( *obj.pVal );
		}

		return *this;
	}


	Int& operator=( Int&& obj )
	{
		tcout << _T("Int::Int& operator=( Int&& obj )...\n");
		if( this != &obj )
		{
			SafeDelete( pVal );

			pVal = obj.pVal;

			obj.pVal = nullptr;
		}

		return *this;
	}



	friend tostream& operator<<( tostream& stream, const Int& obj )
	{
		if( obj.pVal )	stream << *obj.pVal;
		return stream;
	}



	int* pVal;
};



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	//Int* arr = (Int*)malloc( sizeof(Int) * 8 );

	//for( int i=0; i<4; ++i )
	//	new( &arr[i])Int(i);


	//return 0;


	RingQueue<Int> queue;

	tcout << _T("//============== Extend/Shrink with rear < front case =============//\n");

	while(1)
	{
		tcout << _T("queue.Init(8);\n");
		queue.Init(8);
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		for( int i=0; i<7; ++i )
		{
			tcout << _T("Enqueue: ") << i << tendl;
			queue.Enqueue( /*Int(i)*/i );
		}
		tcout << tendl;

		auto Val = Int(7);

		queue.Enqueue( Val );


		queue.Display();
		tcout << tendl;

		for( int i=0; i<4; ++i )
		{
			//auto val = queue.Dequeue();
			Int val;
			queue.Dequeue( val );
			tcout << _T("Dequeue: ") << val << tendl;
		}
		tcout << tendl;

		queue.Display();
		tcout << tendl;


		int v= -2;
		tcout << _T("queue.Enqueue(-2);\n");
		queue.Enqueue(v);
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
	}


	tcout << _T("//============== Extend/Shrink with front < rear case =============//\n");

	while(1)
	{
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
	}

	return 0;
}