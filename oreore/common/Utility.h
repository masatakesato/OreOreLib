#ifndef	UTILITY_H
#define	UTILITY_H


#include	<assert.h>
#include	<limits>



#if _WIN32 || _WIN64// Check windows

	#if _WIN64
		#define ENVIRONMENT64
	#else
		#define ENVIRONMENT32
	#endif

#elif __GNUC__// Check GCC

	#if __x86_64__ || __ppc64__
		#define ENVIRONMENT64
	#else
		#define ENVIRONMENT32
	#endif

#endif




#ifndef NULL
#define	NULL 0
#endif



//using byte		= char;// typedef char			byte;
//using ubyte		= unsigned char;// typedef unsigned char	ubyte;
//using ushort	= unsigned short;// typedef unsigned short	ushort;

//using ulong		= unsigned long;//typedef unsigned long	ulong;




// Platform-independent types

using int8		= signed char;
using uint8		= unsigned char;

using int16		= signed short;
using uint16	= unsigned short;

//using uint		= unsigned int;//typedef	unsigned int	uint;
using int32		= int;
using uint32	= unsigned int;

using int64		= long long;
using uint64	= unsigned long long;

using float32	= float;
using float64	= double;
using float128	= long double;


#ifdef ENVIRONMENT64
using uintptr	= uint64;
#elif ENVIRONMENT32
using uintptr	= uint32;
#endif




namespace BitSize
{
	const size_t uInt8		= std::numeric_limits<uint8>::digits;
	const size_t Int8		= std::numeric_limits<int8>::digits;

	const size_t uInt16		= std::numeric_limits<uint16>::digits;
	const size_t Int16		= std::numeric_limits<int16>::digits;

	const size_t uInt32		= std::numeric_limits<uint32>::digits;
	const size_t Int32		= std::numeric_limits<int32>::digits;

	const size_t uInt64		= std::numeric_limits<uint64>::digits;
	const size_t Int64		= std::numeric_limits<int64>::digits;

	const size_t Float32	= std::numeric_limits<float32>::digits;

	const size_t Float64	= std::numeric_limits<float64>::digits;

	const size_t Float128	= std::numeric_limits<float128>::digits;

	const size_t uIntPtr	= std::numeric_limits<uintptr>::digits;
};




namespace ByteSize
{
	const size_t uInt8		= sizeof(uint8);// size_t
	const size_t Int8		= sizeof(int8);

	const size_t uInt16		= sizeof(uint16);
	const size_t Int16		= sizeof(int16);

	const size_t uInt32		= sizeof(uint32);
	const size_t Int32		= sizeof(int32);

	const size_t uInt64		= sizeof(uint64);
	const size_t Int64		= sizeof(int64);

	const size_t Float32	= sizeof(float32);

	const size_t Float64	= sizeof(float64);

	const size_t Float128	= sizeof(float128);

	const size_t uIntPtr	= sizeof(uintptr);


	const size_t DefaultAlignment = 8;

};



namespace MinPrecision
{
	const auto uInt8	= (std::numeric_limits<uint8>::min)();
	const auto Int8		= (std::numeric_limits<int8>::min)();

	const auto uInt16	= (std::numeric_limits<uint16>::min)();
	const auto Int16	= (std::numeric_limits<int16>::min)();

	const auto uInt32	= (std::numeric_limits<uint32>::min)();
	const auto Int32	= (std::numeric_limits<int32>::min)();

	const auto uInt64	= (std::numeric_limits<uint64>::min)();
	const auto Int64	= (std::numeric_limits<int64>::min)();

	const auto Float32	= (std::numeric_limits<float32>::min)();

	const auto Float64	= (std::numeric_limits<float64>::min)();

	const auto Float128	= (std::numeric_limits<float128>::min)();
}



namespace MaxPrecision
{
	const auto uInt8	= (std::numeric_limits<uint8>::max)();
	const auto Int8		= (std::numeric_limits<int8>::max)();

	const auto uInt16	= (std::numeric_limits<uint16>::max)();
	const auto Int16	= (std::numeric_limits<int16>::max)();

	const auto uInt32	= (std::numeric_limits<uint32>::max)();
	const auto Int32	= (std::numeric_limits<int32>::max)();

	const auto uInt64	= (std::numeric_limits<uint64>::max)();
	const auto Int64	= (std::numeric_limits<int64>::max)();

	const auto Float32	= (std::numeric_limits<float32>::max)();

	const auto Float64	= (std::numeric_limits<float64>::max)();

	const auto Float128	= (std::numeric_limits<float128>::max)();
};





// multithread dll
#if defined(NDEBUG) && defined(_DLL)
#define IS_MD
#endif


// multi thread debug dll
#if defined(_DEBUG) && defined(_DLL)
#define	IS_MDD
#endif


// multi thread
#if defined(NDEBUG) && !defined(_DLL)
#define	IS_MT
#endif


// multi thread debug
#if defined(_DEBUG) && !defined(_DLL)
#define	IS_MTD
#endif


#if defined(UNICODE) || defined(_UNICODE)
#define	IS_UNICODE
#endif



#ifdef _DEBUG


	#define HANDLE_EXCEPTION() \
		try { \
			throw; \
		} \
		catch( const std::exception &e ) { \
			tcout << e.what() << _T( "\n" ); \
		} \
		catch( const int i ) { \
			tcout << i << _T( "\n" ); \
		} \
		catch( const long l ) { \
			tcout << l << _T( "\n" ); \
		} \
		catch( const char *p ) { \
			tcout << p << _T( "\n" ); \
		} \
		catch( ... ) { \
			tcout << _T( "unknown exception occured...\n" ); \
		}


	namespace internal
	{
		static int abort()
		{
			::abort();
			return 1;
		}
	}

	#define ASSERT( expression ) \
		!(expression) && tcerr << _T( "Assertion Failed: "#expression", " ) << __FILE__ << _T("(") << __LINE__ << _T(")\n") && internal::abort()
		//(!(expr) && printf("Assertion Failed: "#expr", %s(%d)\n", __FILE__, __LINE__, __VA_ARGS__) /*&& internal::abort()*/)



#else

	#define HANDLE_EXCEPTION()


	#define ASSERT( ... )



#endif
// https://stackoverflow.com/questions/3641737/c-get-description-of-an-exception-caught-in-catch-block





template< typename T >
inline static void SafeDelete( T*& p )
{
	if( p )
	{
		delete p;
		p = nullptr;//NULL;
	}
}


template< typename T >
inline static void SafeDeleteArray( T*& p )
{
	if( p )
	{
		delete[] p;
		p = nullptr;//NULL;
	}
}


template< typename T >
inline static void SafeRelease( T*& p )
{
	if( p )
	{
		p->Release();
		p = nullptr;//NULL;
	}
}


template< typename T, size_t SIZE >
inline static size_t ArraySize( const T (&)[SIZE] )
{   
    return SIZE;
}






#endif	// UTILITY_H //