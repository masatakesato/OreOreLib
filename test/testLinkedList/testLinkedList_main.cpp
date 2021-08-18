#include	<crtdbg.h>
#include	<list>

#include	<oreore/common/TString.h>
#include	<oreore/container/StaticArray.h>
#include	<oreore/container/LinkedList.h>
using namespace OreOreLib;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	LinkedList<float> list;

	tcout << _T("//=============== PushFront test ==============//\n");
	for( int i=0; i<3; ++i )
	{
		tcout << _T("PushFront: ") << float(i) << tendl;
		list.PushFront( float(i) );
	}

	list.Display();
	tcout << tendl;


	tcout << _T("//=============== PushBack test ==============//\n");
	for( int i=0; i<5; ++i )
	{
		tcout << _T("PushBack: ") << float(i) << tendl;
		list.PushBack( float(i) );
	}

	list.Display();
	tcout << tendl;


	for( auto& elm : list )
	{
		tcout << "--" << tendl;
	}



	tcout << _T("//=============== Copy test ==============//\n");
	LinkedList<float> list2( list );//list2 = list;// copy constructor
	//LinkedList<float> list2( std::move(list) );// move constructor
	//list2 = std::move(list);// copy assignment operator
	//list2 = std::move(list);// assignment operator



	tcout << _T("//=============== PopFront test ==============//\n");
	for( int i=0; i<2; ++i )
	{
		auto front = list.Front();
		list.PopFront();
		tcout << _T("PopFront: ") << front << tendl;
	}

	list.Display();
	tcout << tendl;


	tcout << _T("//=============== PopBack test ==============//\n");
	for( int i=0; i<4; ++i )
	{
		//auto back = list.Back();
		list.PopBack();
		//tcout << _T("PopBack: ") << back << tendl;
	}

	list.Display();
	tcout << tendl;

	tcout << _T("//=============== Resize(8) test ==============//\n");
	list.Resize(8);

	list.Display();
	tcout << tendl;


	tcout << _T("//=============== Resize(0) test ==============//\n");
	list.Resize(0);

	list.Display();
	tcout << tendl;


	tcout << _T("//=============== Resize(1) test ==============//\n");
	list.Resize(1, -3333);

	list.Display();
	tcout << tendl;



	tcout << _T("//=============== View copy result ==============//\n");
	list2.Display();

	return 0;
}