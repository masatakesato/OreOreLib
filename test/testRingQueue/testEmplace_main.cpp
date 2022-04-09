#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/container/RingQueue.h>
using namespace OreOreLib;


struct Struct
{
	Struct()
		: pVal( new int() )
	{
		tcout << _T("Struct::Struct()...\n");	
	}


	Struct( int i, const tstring& s )
		: pVal( new int(i) )
		, str( s )
	{
		tcout << _T("Struct::Struct( int i )...\n");	
	}


	~Struct()
	{
		tcout << _T("Struct::~Struct()...\n");	
		SafeDelete( pVal );
		//str.clear();
		//str.shrink_to_fit();
		str.~tstring();
	}


	Struct( const Struct& obj )
		: pVal( new int(*obj.pVal) )
		//, str( obj.str )
	{
		tcout << _T("Struct::Struct( const Struct& obj )...\n");
	}


	Struct( Struct&& obj )
		: pVal( obj.pVal )
		, str( (tstring&&)obj.str )
	{
		tcout << _T("Struct::Struct( Struct&& obj )...\n");

		obj.pVal = nullptr;
		//obj.str.~tstring();// stringクラスはムーブコンストラクタ持ってないから、明示的に削除する必要がある
	}


	Struct& operator=( const Struct& obj )
	{
		tcout << _T("Struct::Struct& operator=( const Struct& obj )...\n");
		if( this != &obj )
		{
			SafeDelete( pVal );

			pVal = new int( *obj.pVal );
			str = obj.str;
		}

		return *this;
	}


	Struct& operator=( Struct&& obj )
	{
		tcout << _T("Struct::Struct& operator=( Struct&& obj )...\n");
		if( this != &obj )
		{
			SafeDelete( pVal );

			pVal = obj.pVal;
			str = obj.str;

			obj.pVal = nullptr;
			//obj.str.~tstring();// stringクラスはムーブコンストラクタ持ってないから、明示的に削除する必要がある
		}

		return *this;
	}



	friend tostream& operator<<( tostream& stream, const Struct& obj )
	{
		if( obj.pVal )	stream << *obj.pVal;
		stream << ", " << obj.str.c_str();
		return stream;
	}



	int* pVal;
	tstring str;
};




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	RingQueue<Struct> queue;

	tcout << _T("//============== Extend/Shrink with rear < front case =============//\n");

	//while(1)
	{
		tcout << _T("queue.Init(8);\n");
		queue.Init(8);
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		for( int i=0; i<7; ++i )
		{
			tcout << _T("Enqueue: ") << i << tendl;
			queue.Emplace( /*Struct(i)*/i, to_tstring(i).c_str() );
		}
		tcout << tendl;

		auto Val = Struct( 7, _T("7") );
		queue.Enqueue( Val );

		queue.Display();
		tcout << tendl;

		for( int i=0; i<4; ++i )
		{
			//auto val = queue.Dequeue();
			Struct val;
			queue.Dequeue( val );
			tcout << _T("Dequeue: ") << val << tendl;
		}
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		int v= -2;
		tcout << _T("queue.Enqueue(-2);\n");
		queue.Emplace( v, _T("-2") );
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		tcout << _T("queue.Extend(6);\n");
		queue.Extend(6);
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		tcout << _T("queue.Enqueue(-9999);\n");
		queue.Emplace( -9999, _T("-9999") );
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
			auto str = _T("Enqueue: ") + to_tstring(i);
			queue.Emplace(i, str.c_str() );
		}
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		for( int i=0; i<3; ++i )
		{
			auto val = queue.Dequeue();
			tcout << _T("Dequeue: ") << val << tendl;
		}
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		tcout << _T("queue.Extend(6);\n");
		queue.Extend(6);
		tcout << tendl;

		queue.Display();
		tcout << tendl;

		tcout << _T("queue.Enqueue(-9999);\n");
		auto str = _T("Enqueue: ") + to_tstring(-9999);
		queue.Emplace( -9999, str.c_str() );
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