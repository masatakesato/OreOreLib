#include	<oreore/memory/MemoryOperations.h>
using namespace OreOreLib;


struct Val
{
	Val() : value( new float(0) )
	{
		tcout << "Val()\n";
	}

	Val( float val )
		: value( new float(val) )
	{
		tcout << "Val( float val )\n";
	}

	~Val()
	{
		tcout << "~Val(): " << tendl;
		SafeDelete( value );
	}

	Val( const Val& obj )
		: value( new float(*obj.value)  )
	{
		tcout << "Val( const Val& obj ) : " << *value << tendl;
	}

	Val( Val&& obj )
		: value( obj.value )
	{
		tcout << "Val( Val&& obj )\n";
		 obj.value = nullptr;
	}


	float* value;

};



//int main()
//{
//	int* a = new int[5];// = {1, 2, 3, 4, 5};
//	int* b = new int[5];//{};
//
//	MemMove( &b[0], &a[0], (sizeType)5 );
//
//
//	std::string va[4], vb[4];
//	Uninitialized_MemMove( &vb[0], &va[0], (sizeType)5 );
//
//}





template < class F, class SrcIter, class DstIter >
void ForwardMemScanProcess_( F&& func, DstIter* pDst, const SrcIter* pSrc, sizeType size )
{
	SrcIter* begin = (SrcIter*)pSrc;
	const SrcIter* end = pSrc + size;
	DstIter* out = pDst;

	while( begin != end )
	{
		func( out, begin );
		++begin; ++out;
	}
}



template < class F, class SrcIter, class DstIter >
void BackwardMemScanProcess_( F&& func, DstIter* pDst, const SrcIter* pSrc, sizeType size )
{
	SrcIter* begin = (SrcIter*)pSrc + size - 1;
	const SrcIter* end = pSrc - 1;
	DstIter* out = pDst + size - 1;

	while( begin != end )
	{
		func( out, begin );
		--begin; --out;
	}
}




//
//template < class F, class Iter >
//void BackwardMemScanProcess_( F&& f, Iter* pDst, const Iter* pSrc, sizeType size )
//{
//	Iter* begin = (Iter*)pSrc + size - 1;
//	const Iter* end = pSrc - 1;
//	Iter* out = pDst + size - 1;
//
//	while( begin != end )
//	{
//		// DoSomething
//		f( begin, out );
//		
//		--begin; --out;
//	}
//}



template < class SrcIter, class DstIter >
void CopyOperation( DstIter* dst, SrcIter* src )
{
	tcout << "CopyOperation()...\n";
	dst->~DstIter();// Destruct existing data from destination memory
	new ( dst ) DstIter( *(DstIter*)src );// Call copy constructor
}



template < class SrcIter, class DstIter >
void MigrateOperation( DstIter* dst, SrcIter* src )
{
	tcout << "MigrateOperation()...\n";
	dst->~DstIter();// Destruct existing data
	new ( dst ) DstIter( (DstIter&&)( *src ) );// Overwite existing memory with placement new
}





template < class SrcIter, class DstIter >
void Migrate_Uninitialized( DstIter* pDst, const SrcIter* pSrc, sizeType size ) 
{
	//BackwardMemScanProcess_( CopyOperation<Iter>, pDst, pSrc, size );

	ForwardMemScanProcess_(
		[]( DstIter* dst, SrcIter* src )
		{
			tcout << "Func!!!\n";
			//	dst->~DstIter();// Destruct existing data
			new ( dst ) DstIter( (DstIter&&)( *src ) );// Overwite existing memory with placement new
		},
		pDst, pSrc, size );
}


template < class SrcIter, class DstIter >
void Copy_Uninitialized( DstIter* pDst, const SrcIter* pSrc, sizeType size ) 
{
	//BackwardMemScanProcess_( CopyOperation<Iter>, pDst, pSrc, size );

	ForwardMemScanProcess_(
		[]( DstIter* dst, SrcIter* src )
		{
			tcout << "Func!!!\n";
			//	dst->~DstIter();// Destruct existing data
			new ( dst ) DstIter( *(DstIter*)src );// Overwite existing memory with placement new
		},
		pDst, pSrc, size );
}






template<typename F>
int function(F foo, int a)
{
    return foo(a);
}

int test(int a)
{
    return a;
}

int main()
{
    // function will work out the template types
    // based on the parameters.
    function(test, 1);
    function([](int a) -> int { return a; }, 1);


	std::string src[4], dst[4];

	src[0] = "0";
	src[1] = "1";
	src[2] = "2";
	src[3] = "3";


//	BackwardMemScanProcess_( Func<std::string>, &va[0], &vb[0], 4 );

	Copy_Uninitialized( &dst[0], &src[0], 4 );
}