#ifndef MEMORY_OPERATIONS_H
#define	MEMORY_OPERATIONS_H

//#include	<algorithm>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"
#include	"../algorithm/Algorithm.h"



namespace OreOreLib
{

	//##############################################################################################//
	//																								//
	//											MemCopy												//
	//																								//
	//##############################################################################################//

#if __cplusplus >= 201703L	// Above C++17

	// If using Visual c++, folowing command must be added for __cplusplus macro activation.
	//   /Zc:__cplusplus

	// Memory Copy
	template < class SrcIter, class DstIter >
	DstIter* MemCopy( DstIter* pDst, SrcIter* pSrc, sizeType size )
	{
		if constexpr ( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
		{
			return (DstIter*)memcpy( pDst, pSrc, sizeof DstIter * size );
		}
		else
		{
			SrcIter* begin = pSrc;
			SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while( begin != end )
			{
				// Placement new version
				out->~DstIter();// Desctuct existing data from destination memory
				new ( out ) DstIter( *(DstIter*)begin );// Call copy constructor

				// Copy assignment operator version
				//*out = *(DstIter*)begin;

				++begin; ++out;
			}
			
			return out;
		}
	}

	// Uninitialized Memory Copy
	template < class SrcIter, class DstIter >
	DstIter* Uninitialized_MemCopy( DstIter* pDst, SrcIter* pSrc, sizeType size )
	{
		if constexpr ( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
		{
			return (DstIter*)memcpy( pDst, pSrc, sizeof DstIter * size );
		}
		else
		{
			SrcIter* begin = pSrc;
			SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while( begin != end )
			{
				// Placement new version
				new ( out ) DstIter( *(DstIter*)begin );// Call copy constructor

				// Copy assignment operator version
				//*out = *(DstIter*)begin;

				++begin; ++out;
			}
			
			return out;
		}
	}



#else	// Below C++14

	// Trivial Memcpy
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemCopy( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
	}

	// Uninitialized Trivial Memcpy( same as MemCpy )
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	Uninitialized_MemCopy( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
	}



	// Non-Trivial Memcpy
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	MemCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		SrcIter* begin = (SrcIter*)pSrc;
		const SrcIter* end = pSrc + size;
		DstIter* out = pDst;

		while( begin != end )
		{
			// Placement new version
			out->~DstIter();// Destruct existing data from destination memory
			new ( out ) DstIter( *(DstIter*)begin );// Call copy constructor

			// Copy assignment operator version
			//*out = *(DstIter*)begin;

			++begin; ++out;// expecting copy assignment operator implementation
		}
		
		return out;
	}

	// Non-Trivial Uninitialized Memcpy
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	Uninitialized_MemCopy( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		SrcIter* begin = (SrcIter*)pSrc;
		const SrcIter* end = pSrc + size;
		DstIter* out = pDst;

		while( begin != end )
		{
			// Placement new version
			new ( out ) DstIter( *(DstIter*)begin );// Call copy constructor

			// Copy assignment operator version
			//*out = *(DstIter*)begin;

			++begin; ++out;// expecting copy assignment operator implementation
		}
		
		return out;
	}




#endif




	//##############################################################################################//
	//																								//
	//								MemMove / Uninitialized_MemMove									//
	//																								//
	//##############################################################################################//

#if __cplusplus >= 201703L	// Above C++17

	// Memory Move
	template < class SrcIter, class DstIter >
	DstIter* MemMove( DstIter* pDst, SrcIter* pSrc, sizeType size )
	{
		if constexpr ( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
		{
			return (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
		}
		else
		{
			SrcIter* begin = pSrc;
			SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while(begin != end)
			{
				// Placement new version
				out->~DstIter();// Desctuct existing data from destination memory
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Call move constructor
				
				// Copy assignment operator version
				//*out = *(DstIter*)begin;
				
				++begin; ++out;
			}

			return out;
		}
	}

	// Uninitialized Memory Move
	template < class SrcIter, class DstIter >
	DstIter* Uninitialized_MemMove( DstIter* pDst, SrcIter* pSrc, sizeType size )
	{
		if constexpr ( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
		{
			return (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
		}
		else
		{
			SrcIter* begin = pSrc;
			SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while(begin != end)
			{
				// Placement new version
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Call move constructor
				
				// Copy assignment operator version
				//*out = *(DstIter*)begin;
				
				++begin; ++out;
			}

			return out;
		}
	}



#else	// Below C++14

	// Trivial MemMove
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemMove( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
	}

	// Trivial Uninitialized MemMove(same as MemMove)
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	Uninitialized_MemMove( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
	}



template < class SrcIter, class DstIter >
void ForwardMemScanProcess( DstIter* pDst, const SrcIter* pSrc, sizeType size )
{
	SrcIter* begin = (SrcIter*)pSrc;
	const SrcIter* end = pSrc + size;
	DstIter* out = pDst;

	while( begin != end )
	{
		// DoSomething

		++begin; ++out;
	}
}


template < class SrcIter, class DstIter >
void BackwardMemScanProcess( DstIter* pDst, const SrcIter* pSrc, sizeType size )
{
	SrcIter* begin = (SrcIter*)pSrc + size - 1;
	const SrcIter* end = pSrc - 1;
	DstIter* out = pDst + size - 1;

	while( begin != end )
	{
		// DoSomething

		--begin; --out;
	}
}

//TODO: テスト
/*
template<typename F>
int function(F foo, int a) {
    return foo(a);
}

int test(int a) {
    return a;
}

int main()
{
    // function will work out the template types
    // based on the parameters.
    function(test, 1);
    function([](int a) -> int { return a; }, 1);
}


*/



	// Non-Trivial MemMove
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	MemMove( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		if( pSrc < pDst )// Copy from the last element
		{
			SrcIter* begin = (SrcIter*)pSrc + size - 1;
			const SrcIter* end = pSrc - 1;
			DstIter* out = pDst + size - 1;

			while( begin != end )
			{
				// Placement new version
				out->~DstIter();// Destruct existing data
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Overwite existing memory with placement new

				// Copy assignment operator version. cannot deal with dynamic memory object( e.g., string )
				//*out = *(DstIter*)begin;

				--begin; --out;
			}
		}
		else if( pSrc > pDst )// Copy from the first element
		{
			SrcIter* begin = (SrcIter*)pSrc;
			const SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while( begin != end )
			{
				// Placement new version
				out->~DstIter();// Destruct existing data
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Overwite existing memory with placement new

				// Copy assignment operator version. cannot deal with dynamic memory object( e.g., string )
				//*out = *(DstIter*)begin;

				++begin; ++out;
			}
		}
		
		return out;
	}

	// Non-Trivial Uninitialized MemMove
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	Uninitialized_MemMove( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		SrcIter* begin = (SrcIter*)pSrc;
		const SrcIter* end = pSrc + size;
		DstIter* out = pDst;

		while( begin != end )
		{
			// Placement new version
			new ( out ) DstIter( (DstIter&&)( *begin ) );// Overwite existing memory with placement new

			// Copy assignment operator version. cannot deal with dynamic memory object( e.g., string )
			//*out = *(DstIter*)begin;

			++begin; ++out;
		}
		
		return out;
	}



#endif




	//##############################################################################################//
	//																								//
	//											MemMigrate											//
	//																								//
	//##############################################################################################//

#if __cplusplus >= 201703L	// Above C++17

	// Memory Migrate
	template < class SrcIter, class DstIter >
	DstIter* MemMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
	{
		if constexpr ( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
		{
			auto result = (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
			memset( pSrc, 0, sizeof SrcIter * size );
			return result;
		}
		else
		{
			SrcIter* begin = pSrc;
			SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while(begin != end)
			{
				// Placement new version
				out->~DstIter();// Desctuct existing data from destination memory
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Call move constructor
				begin->~SrcIter();// Cleanup source

				// Copy assignment operator version
				//*out = *(DstIter*)begin;
				
				++begin; ++out;
			}

			return out;
		}
	}

	// Uninitialized Memory Migrate
	template < class SrcIter, class DstIter >
	DstIter* Uninitialized_MemMigrate( DstIter* pDst, SrcIter* pSrc, sizeType size )
	{
		if constexpr ( std::is_same_v<SrcIter, DstIter> && std::is_trivially_copyable_v<SrcIter> )
		{
			auto result = (DstIter*)memmove( pDst, pSrc, sizeof DstIter * size );
			memset( pSrc, 0, sizeof SrcIter * size );
			return result;
		}
		else
		{
			SrcIter* begin = pSrc;
			SrcIter* end = pSrc + size;
			DstIter* out = pDst;

			while(begin != end)
			{
				// Placement new version
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Call move constructor
				begin->~SrcIter();// Cleanup source

				// Copy assignment operator version
				//*out = *(DstIter*)begin;
				
				++begin; ++out;
			}

			return out;
		}
	}


#else	// Below C++14
	
	// Trivial MemMigrate
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemMigrate( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		auto result = (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
		memset( pSrc, 0, sizeof Iter * size );
		return result;
	}

	// Trivial Uninitialized MemMigrate( same as MemMigrate )
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	Uninitialized_MemMigrate( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		auto result = (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
		memset( pSrc, 0, sizeof Iter * size );
		return result;
	}



	// Non-Trivial MemMigrate
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	MemMigrate( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		SrcIter* begin = (SrcIter*)pSrc;
		const SrcIter* end = pSrc + size;
		DstIter* out = pDst;

		while( begin != end )
		{
			// Placement new version
			out->~DstIter();// Destruct existing data
			new ( out ) DstIter( (DstIter&&)( *begin ) );// Overwite existing memory with placement new
			begin->~SrcIter();// Cleanup source

			// Copy assignment operator version. cannot deal with dynamic memory object( e.g., string )
			//*out = *(DstIter*)begin;

			++begin; ++out;
		}
		
		return out;
	}
	
	// Non-Trivial Uninitialized MemMigrate
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	Uninitialized_MemMigrate( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		SrcIter* begin = (SrcIter*)pSrc;
		const SrcIter* end = pSrc + size;
		DstIter* out = pDst;

		while( begin != end )
		{
			// Placement new version
			new ( out ) DstIter( (DstIter&&)( *begin ) );// Overwite existing memory with placement new
			begin->~SrcIter();// Cleanup source

			// Copy assignment operator version. cannot deal with dynamic memory object( e.g., string )
			//*out = *(DstIter*)begin;

			++begin; ++out;
		}
		
		return out;
	}



#endif




	//##############################################################################################//
	//																								//
	//											MemClear											//
	//																								//
	//##############################################################################################//

#if __cplusplus >= 201703L	// Above C++17

	// Memory Clear
	template < class Iter >
	Iter* MemClear( Iter* pDst, sizeType size )
	{
		if constexpr ( std::is_trivially_copyable_v<Iter> )
		{
			return (Iter*)memset( pDst, 0, sizeof Iter * size );
		}
		else
		{
			Iter* begin = pDst;
			const Iter* end = pDst + size;

			while( begin != end )
			{
				begin->~Iter();// Desctuct existing data
				++begin;
			}
		
			return pDst;
		}
	}



#else	// Below C++14

	// Trivial MemClear
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemClear( Iter* pDst, sizeType size )
	{
		return (Iter*)memset( pDst, 0, sizeof Iter * size );
	}



	// Non-Trivial MemClear
	template < class Iter >
	std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	MemClear( Iter* pDst, sizeType size )
	{
		Iter* begin = pDst;
		const Iter* end = pDst + size;

		while( begin != end )
		{
			begin->~Iter();// Desctuct existing data
			++begin;
		}
	
		return pDst;
	}



#endif




}// end of namespace


#endif // !MEMORY_OPERATIONS_H
