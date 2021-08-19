#include	<crtdbg.h>
#include	<iostream>


#include	<oreore/common/TString.h>
#include	<oreore/container/Array.h>
#include	<oreore/memory/SharedPtr.h>
using namespace OreOreLib;

using fArray = Array<float>;


auto g_LambdaDeleter = [](int* p) { tcout << _T("Lambda deleter.\n"); delete p; };


template< typename T >
class Del
{
public:

	Del()
		: m_ID(0)
	{
	}

	Del( int i )
		: m_ID(i)
	{
	}
	
	void operator()( T* ptr ) const
	{
		tcout << _T("Del<T> [" << m_ID <<  "]\n");
		SafeDelete( ptr );
	}

private:

	int m_ID = 0;

};



template< typename T >
class Del< T[] >
{
public:

	Del()
		: m_ID(0)
	{
	}

	Del( int i )
		: m_ID(i)
	{
	}

	void operator()( T* ptr ) const
	{
		tcout << _T("Del<T[]> [" << m_ID <<  "]\n");
		SafeDeleteArray( ptr );
	}

private:

	int m_ID = 0;

};



class FClose
{
public:

	void operator()( FILE* fp ) const
	{
		if( fp )
		{
			tcout << _T("FClose\n");
			fclose( fp );
			fp = nullptr;
		}
	}

};




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	
	{
		tcout << _T( "//===================== SharedPtr< int > with custom delete functor =================//\n" );
		SharedPtr<int> sp( new int(10), Del<int>(55555)/*DefaultDeleter<int>()*/ );
		*sp = 33;
		tcout << *sp << tendl;	
	}

	tcout << tendl;	

	{
		tcout << _T( "//===================== SharedPtr< int[] > with custom delete functor =================//\n" );
		SharedPtr<int[]> sp( new int[10]/*, Del<int[]>(5)/*DefaultDeleter<int[]>()*/ );
		sp[5] = 33;
		tcout << sp[5] << tendl;	
	}	
	
	tcout << tendl;

	{
		tcout << _T( "//===================== SharedPtr< FILE > with custom delete functor =================//\n" );
		SharedPtr<FILE> sp( fopen("test.txt", "r"), FClose() );
		//*sp = 33;
		//tcout << *sp << tendl;	
	}
	
	tcout << tendl;

	{
		tcout << _T( "//===================== SharedPtr< int > with lambda deleter =================//\n" );
		SharedPtr<int> sp( new int(10), g_LambdaDeleter );
		*sp = 33;
		tcout << *sp << tendl;		
	}
	
	tcout << tendl;


	{
		tcout << _T( "//===================== Create SharedPtr<int> from SharedPtr<int> =================//\n" );
		SharedPtr<int> sp1( new int(10), Del<int>(55555) );
		*sp1 *= -10;
		tcout << *sp1 << tendl;

		SharedPtr<int> sp2(sp1);
		tcout << *sp2 << tendl;

		tcout << _T( "reset...\n" );
		sp2.Reset();
		tcout << _T( "...\n" );
	}

	tcout << tendl;

	{
		tcout << _T( "//===================== move SharedPtr<int> =================//\n" );
		SharedPtr<int> sp1( new int(10), Del<int>(55555) );
		*sp1 *= -10;
		tcout << *sp1 << tendl;

		SharedPtr<int> sp2(std::move(sp1) );
		tcout << *sp2 << tendl;

		tcout << _T( "reset...\n" );
		sp2.Reset();
		tcout << _T( "...\n" );
	}

	tcout << tendl;

	{
		tcout << _T( "//===================== Create SharedPtr<int> from UniquePtr<int> =================//\n" );
		UniquePtr<int, Del<int>> up1( new int(10) );
		*up1 *= -10;
		tcout << *up1 << tendl;

		SharedPtr<int> sp2(std::move(up1) );
		tcout << *sp2 << tendl;

		tcout << _T( "reset...\n" );
		sp2.Reset();
		tcout << _T( "...\n" );
	}

	

//	const SharedPtr< int[] > vvv( new int[10] );

//	vvv.UseCount();

//	SharedPtr< int[] > vvv2( vvv );

//	tcout << vvv2[0] << tendl;
//	tcout << vvv[0] << tendl;


//	vvv2.Reset();
	/*
	int *a = new int(6);
	SharedPtr< int > vvv2( a );
//	std::shared_ptr<int> vvv2(a);
//	*vvv2 = -666;

//	vvv2.Reset();

	*vvv2 = -666;
	*/
	return 0;
}




/*
int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	//return 0;


	//SmartPtr<int> aptr( new int );

	//*aptr = 122;

	//auto p = aptr.Get();

	//*p = -9956666;

	//tcout << *aptr << tendl;
	////aptr = 122;


	SharedPtr<fArray> arrptr( new fArray(3) );

	(*arrptr)[0] = 5555;			// Call T& operator*()
	arrptr->begin()[1] = -5869486;	// Call T* operator->()
	
	tcout << (*arrptr)[0] << tendl;

	tcout << arrptr.UseCount() << tendl;

	const SharedPtr<fArray> const_arrptr = arrptr;

	(*const_arrptr)[0];		// Call const T& operator*() const
	const_arrptr->begin();	// const T* operator->() const

	//tcout << arrptr.UseCount() << tendl;
	//tcout << (const_arrptr==const_arrptr) << tendl;
	//tcout << (const_arrptr==arrptr) << tendl;

	//{
	//	SmartPtr<fArray> ptr2 = aptr;

	//	(*ptr2)[0] = 6666.333f;
	//	ptr2->Init(10);
	//}


	{
		UniquePtr<int> uptr( new int(999) );

		tcout << *uptr << tendl;

		SharedPtr<int> sptr( uptr );

		tcout << *sptr << tendl;

		//tcout << *uptr << tendl; // Cannot Do. sptr has taken Ownership.
	
	}


	//{
	//	UniquePtr<int[]> uptr( new int[30] );
	//	uptr[28] = -5531355;

	//	tcout << uptr[28] << tendl;

	//	SharedPtr<int[]> sptr( uptr );

	//	tcout << sptr[28] << tendl;

	//	tcout << sptr[28] << tendl; // Cannot Do. sptr has taken Ownership.
	//
	//}


	{
		int *aaa = new int[30];
		UniquePtr<int[]> uptr( aaa );
		uptr[28] = -5531355;

		tcout << uptr[28] << tendl;

		SharedPtr<int[]> sptr( uptr );

		tcout << sptr[28] << tendl;

		tcout << sptr[28] << tendl; // Cannot Do. sptr has taken Ownership.
	
	}


	//int iarray[5] = {1, 2, 3, 4, 5};



	const SharedPtr< int[] > vvv( new int[10] );

	vvv.UseCount();



	SharedPtr< int[] > vvv2( vvv );

//	auto &&a = SharedPtr<int[]>( new int[3] )[0];// forbidden
//	int b = SharedPtr<int[]>( new int[3] )[0];// forbidden

//	a = 3;

	tcout << vvv2[0] << tendl;
	tcout << vvv[0] << tendl;


	vvv2.Reset();

//	int b=3;
	return 0;
}
*/