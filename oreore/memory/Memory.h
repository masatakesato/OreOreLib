#ifndef MEMORY_H
#define	MEMORY_H

//#include	<algorithm>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"
#include	"../algorithm/Algorithm.h"

//TODO: Replace SizeType with MemSizeType


namespace OreOreLib
{

	//##############################################################################################################//
	//																												//
	//										Memory size_type definitions											//
	//																												//
	//##############################################################################################################//


	#if defined( MEM_64 )

		using MemSizeType = typedef uint64;// 64bit

	#elif defined( MEM_86 )

		using MemSizeType = typedef uint32;// 32bit

	#elif defined( MEM_ENVIRONMENT )
	
		using MemSizeType = typename sizeType;// platform dependent

	#else

		using MemSizeType = typename uint32;// default configuration

	#endif





	#if __cplusplus >= 201703L
	// If using Visual c++, folowing command must be added for __cplusplus macro activation.
	//   /Zc:__cplusplus

	//##############################################################################################################//
	//																												//
	//											MemCopy (above C++17)												//
	//																												//
	//##############################################################################################################//

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
				//out->~DstIter();// Desctuct existing data from destination memory
				new ( out ) DstIter( *(DstIter*)begin );// Call copy constructor

				// Copy assignment operator version
				//*out = *(DstIter*)begin;

				++begin; ++out;
			}
			
			return out;
		}
	}

	//// Memory Copy
	//template < class Iter >
	//Iter* MemCopy( Iter* pDst, Iter* pSrc, sizeType size )
	//{
	//	if constexpr ( std::is_trivially_copyable_v<Iter> )
	//	{
	//		return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
	//	}
	//	else
	//	{
	//		Iter* begin = pSrc;
	//		Iter* end = pSrc + size;
	//		Iter* out = pDst;

	//		while( begin != end )
	//		{
	//			// Placement new version
	//			out->~Iter();// Desctuct existing data from destination memory
	//			new ( out ) Iter( *begin );// Call copy constructor
	//
	//			// Copy assignment operator version
	//			*out = *begin;
	//			++begin; ++out;
	//		}
	//
	//		return out;
	//	}
	//}



	//##############################################################################################################//
	//																												//
	//											MemMove (above C++17)												//
	//																												//
	//##############################################################################################################//

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
				//out->~DstIter();// Desctuct existing data from destination memory
				new ( out ) DstIter( (DstIter&&)( *begin ) );// Call move constructor
				
				// Copy assignment operator version
				//*out = *(DstIter*)begin;
				
				++begin; ++out;
			}

			return out;
		}
	}

	//// Memory Move
	//template < class Iter >
	//Iter* MemMove( Iter* pDst, Iter* pSrc, sizeType size )
	//{
	//	if constexpr ( std::is_trivially_copyable_v<Iter> )
	//	{
	//		return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
	//	}
	//	else
	//	{
	//		Iter* begin = pSrc;
	//		Iter* end = pSrc + size;
	//		Iter* out = pDst;

	//		while(begin != end)
	//		{
	//			// Placement new version
	//			//out->~Iter();// Desctuct existing data from destination memory
	//			new ( out ) Iter( (Iter&&)( *begin ) );// Call move constructor
	//
	//			// Copy assignment operator version
	//			*out = *begin;
	//
	//			++begin; ++out;
	//		}

	//		return out;
	//	}
	//}


	//##############################################################################################################//
	//																												//
	//											MemClear (above C++17)												//
	//																												//
	//##############################################################################################################//

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



	#else
	

	//##############################################################################################################//
	//																												//
	//											MemCopy (below C++14)												//
	//																												//
	//##############################################################################################################//

	// Trivial Memcpy
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemCopy( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
	}

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
			//out->~DstIter();// Desctuct existing data from destination memory
			new ( out ) DstIter( *(DstIter*)begin );// Call copy constructor

			// Copy assignment operator version
			//*out = *(DstIter*)begin;

			++begin; ++out;// expecting copy assignment operator implementation
		}
		
		return out;
	}

	//// Single type Non-Trivial Memcpy( single template type )
	//template < class Iter >
	//std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	//MemCopy( Iter* pDst, const Iter* pSrc, sizeType size )
	//{
	//	Iter* begin = (Iter*)pSrc;
	//	const Iter* end = pSrc + size;
	//	Iter* out = pDst;

	//	while( begin != end )
	//	{
	//		// Placement new version
	//		//out->~Iter();// Desctuct existing data from destination memory
	//		new ( out ) Iter( *begin );// Call copy constructor
	//
	//		// Copy assignment operator version
	//		//*out = *begin;
	//
	//		++begin; ++out;
	//	}
	//
	//	return out;
	//}



	//##############################################################################################################//
	//																												//
	//											MemMove (below C++14)												//
	//																												//
	//##############################################################################################################//

	// Trivial MemMove
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemMove( Iter* pDst, const Iter* pSrc, sizeType size )
	{
		return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
	}

	// Non-Trivial MemMove
	template < class SrcIter, class DstIter >
	std::enable_if_t< (!std::is_same_v<SrcIter, DstIter> && std::is_convertible_v<SrcIter, DstIter>) || !std::is_trivially_copyable_v<SrcIter> || !std::is_trivially_copyable_v<DstIter>, DstIter* >
	MemMove( DstIter* pDst, const SrcIter* pSrc, sizeType size )
	{
		SrcIter* begin = (SrcIter*)pSrc;
		const SrcIter* end = pSrc + size;
		DstIter* out = pDst;

		while( begin != end )
		{
			// Placement new version
			//out->~DstIter();// Desctuct existing data
			new ( out ) DstIter( (DstIter&&)( *begin ) );// Desctuct existing data from destination memory

			// Copy assignment operator version
			//*out = *(DstIter*)begin;

			++begin; ++out;
		}
		
		return out;
	}

	//// Non-Trivial MemMove( single template type )
	//template < class Iter >
	//std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	//MemMove( Iter* pDst, const Iter* pSrc, sizeType size )
	//{
	//	Iter* begin = (Iter*)pSrc;
	//	const Iter* end = pSrc + size;
	//	Iter* out = pDst;

	//	while( begin != end )
	//	{
	//		// Placement new version
	//		//out->~Iter();// Desctuct existing data from destination memory
	//		new ( out ) Iter( (Iter&&)( *begin ) );// Call move constructor
	//
	//		// Copy assignment operator version
	//		*out = *begin;
	//
	//		++begin; ++out;
	//	}
	//
	//	return out;
	//}



	//##############################################################################################################//
	//																												//
	//											MemClear (below C++14)												//
	//																												//
	//##############################################################################################################//

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



	#endif//__cplusplus





	//##############################################################################################################//
	//																												//
	//											Memory class implementation											//
	//																												//
	//##############################################################################################################//

	template< typename T >
	struct Memory
	{
		using SizeType = typename MemSizeType;//typename uint32;//sizeType;//int32;//uint64;//

	public:

		// Default constructor
		Memory()
			: m_Length( 0 )
			, m_AllocSize( 0 )
			, m_Capacity( 0 )
			, m_pData( nullptr )
		{
			//tcout << _T("Memory default constructor...\n");
		}


		// Constructor
		Memory( SizeType len )
			: m_Length( len )
			, m_AllocSize( len * c_ElementSize )
			, m_Capacity( len )
			, m_pData( AllocateBuffer(len, true) )
		{
			//tcout << _T("Memory constructor(dynamic allocation)...\n");
		}

		
		// Constructor
		Memory( SizeType len, const T& fill )
			: m_Length( len )
			, m_AllocSize( len * c_ElementSize )
			, m_Capacity( len )
			, m_pData( AllocateBuffer(len) )
		{
			for( auto iter=m_pData; iter !=m_pData+len; ++iter )
				new ( iter ) T( fill );
		}


		// Constructor
		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		Memory( Args const & ... args )
			: m_Length( sizeof ...(Args) )
			, m_AllocSize( sizeof ...(Args) * sizeof(T) )
			, m_Capacity( sizeof ...(Args) )
			, m_pData( AllocateBuffer( SizeType( sizeof ...(Args) ) )  )
		{
			auto p = m_pData;
			for( const auto& val : std::initializer_list<T>{args...} )
				*(p++) = val;	
		}

		// Constructor with initializer_list
		Memory( std::initializer_list<T> ilist )
			: m_Length( SizeType( ilist.size() ) )
			, m_AllocSize( SizeType( ilist.size() * sizeof(T) ) )
			, m_Capacity( SizeType( ilist.size() ) )
			, m_pData( AllocateBuffer( SizeType( ilist.size() ) ) )
		{
			auto p = m_pData;
			for( const auto& val : ilist )
				*(p++) = val;
		}


		// Constructor with iterator
		template < class Iter >
		Memory( Iter first, Iter last )
			: m_Length( SizeType( last - first ) )
			, m_AllocSize( m_Length * sizeof(T) )
			, m_Capacity( SizeType(last - first) )
			, m_pData( AllocateBuffer( SizeType(last - first) ) )
		{
			MemCopy( begin(), first, m_Length );
		}


		// Destructor
		virtual ~Memory()
		{
			//tcout << _T("Memory destructor...\n");
			Release();
		}


		// Copy constructor
		Memory( const Memory& obj )
			: m_Length( obj.m_Length )
			, m_AllocSize( obj.m_AllocSize )
			, m_Capacity( obj.m_Capacity )
			, m_pData( nullptr )
		{
			//tcout << _T("Memory copy constructor...\n");
			if( obj.m_pData )
			{
				AllocateBuffer( m_Capacity );
				MemCopy( m_pData, obj.m_pData, Min(m_Length, obj.m_Length) );
			}
		}


		// Move constructor
		Memory( Memory&& obj )
			: m_Length( obj.m_Length )
			, m_AllocSize( obj.m_AllocSize )
			, m_Capacity( obj.m_Capacity )
			, m_pData( obj.m_pData )

		{
			//tcout << _T("Memory move constructor...\n");

			obj.m_Length	= 0;
			obj.m_AllocSize	= 0;
			obj.m_Capacity	= 0;
			obj.m_pData		= nullptr;// clear reference from obj
		}


		// Copy Assignment operator =
		inline Memory& operator=( const Memory& obj )
		{
			if( this != &obj )
			{
				//tcout << _T("Memory copy assignment operator...\n");
				m_Length	= obj.m_Length;
				m_AllocSize	= obj.m_AllocSize;
				m_Capacity	= obj.m_Capacity;
				DeallocateBuffer();
				
				if( obj.m_pData )
				{
					AllocateBuffer( m_Capacity );
					MemCopy( m_pData, obj.m_pData, Min(m_Length, obj.m_Length) );
				}
			}

			return *this;
		}


		// Move assignment operator =
		inline Memory& operator=( Memory&& obj )
		{
			if( this != &obj )
			{
				//tcout << _T("Memory move assignment operator...\n");

				// free current m_pData first.
				DeallocateBuffer();

				// copy data to *this
				m_Length		= obj.m_Length;
				m_AllocSize		= obj.m_AllocSize;
				m_Capacity		= obj.m_Capacity;
				m_pData			= obj.m_pData;

				// clear reference from obj
				obj.m_Length	= 0;
				obj.m_AllocSize	= 0;
				obj.m_Capacity	= 0;
				obj.m_pData		= nullptr;
			}

			return *this;
		}


		// Subscript operator for read only.( called if Memory is const )
		inline const T& operator[]( SizeType n ) const&
		{
			return m_pData[n];
		}


		// Subscript operator for read-write.( called if Memory is non-const )
		inline T& operator[]( SizeType n ) &
		{
			return m_pData[n];
		}


		// Subscript operator. ( called by following cases: "T a = Memory<T>(10)[n]", "auto&& a = Memory<T>(20)[n]" )
		inline T operator[]( SizeType n ) const&&
		{
			return std::move(m_pData[n]);// return object
		}


		inline operator bool() const
		{
			return m_pData != nullptr;
		}


		inline bool operator==( const Memory& rhs ) const
		{
			return m_pData == rhs.m_pData;
		}


		inline bool operator !=( const Memory& rhs ) const
		{
			return m_pData != rhs.m_pData;
		}



		void Init( SizeType len )
		{
			if( len > m_Capacity )
			{
				DeallocateBuffer();
				m_Capacity	= len;
				AllocateBuffer( m_Capacity, true );
			}

			m_Length = len;
			m_AllocSize	= c_ElementSize * len;
		}


		void Init( SizeType len, const T& fill )
		{
			if( len > m_Capacity )
			{
				DeallocateBuffer();
				m_Capacity	= len;
				AllocateBuffer( m_Capacity, false );
			}

			m_Length = len;
			m_AllocSize	= c_ElementSize * len;

			for( auto iter=m_pData; iter !=m_pData+len; ++iter )
				new ( iter ) T( fill );
		}


		void Init( std::initializer_list<T> ilist )
		{
			SizeType len = static_cast<SizeType>( ilist.size() );

			if( len > m_Capacity )
			{
				DeallocateBuffer();
				m_Capacity	= len;
				AllocateBuffer( m_Capacity, false );
			}

			m_Length = len;
			m_AllocSize	= c_ElementSize * len;

			MemCopy( begin(), ilist.begin(), ilist.size() );
		}


		// Constructor with iterator
		template < class Iter >
		void Init( Iter first, Iter last )
		{
			SizeType len = static_cast<SizeType>( last - first );

			if( len > m_Capacity )
			{
				DeallocateBuffer();
				m_Capacity	= len;
				AllocateBuffer( m_Capacity, false );
			}

			m_Length = len;
			m_AllocSize	= c_ElementSize * len;

			MemCopy( begin(), first, m_Length );
		}


		void Release()
		{
			DeallocateBuffer();
			m_Length	= 0;
			m_AllocSize	= 0;
			m_Capacity	= 0;
		}


		void Clear()
		{
			MemClear( m_pData, m_Length );
		}


		void SetValues( uint8* pdata, SizeType len )
		{
			ASSERT( pdata );
			MemCopy( m_pData, (T*)pdata, Min( m_Length, len ) );
		}


		template < typename ... Args >
		std::enable_if_t< TypeTraits::all_convertible<T, Args...>::value, void >
		SetValues( const Args& ... args )
		{
			auto values = { (T)args... };
			MemCopy( m_pData, values.begin(), Min( m_Length, (SizeType)values.size() ) );
		}


		template < typename Type >
		std::enable_if_t< std::is_convertible_v<Type, T>/*  std::is_same_v<Type, T>*/, void >
		SetValues( std::initializer_list<Type> ilist )
		{
			MemCopy( m_pData, ilist.begin(), Min( m_Length, (SizeType)ilist.size() ) );
		}


		inline bool Reserve( SizeType newlen )
		{
			if( newlen <= m_Capacity )
				return false;

			m_Capacity	= newlen;
			T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * m_Capacity ) );

			if( m_pData )
			{
				MemMove( newdata, m_pData, m_Length );
				DeallocateBuffer();
			}

			m_pData = newdata;

			return true;
		}


		inline bool Resize( SizeType newlen )
		{
			// Reallocate memory
			auto oldlen = m_Length;
			ReallocateBuffer( newlen );

			// Initialize allocated elements using placement new default constructor
			for( auto i=oldlen; i<newlen; ++i )	new ( m_pData + i ) T();

			return true;
		}


		inline bool Resize( SizeType newlen, const T& fill )
		{
			// Reallocate memory
			auto oldlen = m_Length;
			ReallocateBuffer( newlen );

			// Initialize allocated elements using placement new copy constructor
			for( auto i=oldlen; i<newlen; ++i )	new ( m_pData + i ) T( fill );

			return true;
		}


		inline bool Extend( SizeType numelms )
		{
			if( numelms==0 || numelms==~0u )	return false;
			return Resize( m_Length + numelms );
		}


		inline bool Extend( SizeType numelms, const T& fill )
		{
			if( numelms==0 || numelms==~0u )	return false;
			return Resize( m_Length + numelms, fill );
		}


		inline bool Shrink( SizeType numelms )
		{
			if( m_Length > numelms )	return ReallocateBuffer( m_Length - numelms );
			return false;
		}


		inline void CopyFrom( const Memory& src )
		{
			MemCopy( m_pData, src.m_pData, Min(m_Length, src.m_Length) );
		}


		inline void CopyTo( Memory& dst ) const
		{
			MemCopy( dst.m_pData, m_pData, Min(m_Length, dst.m_Length) );
		}


		SizeType ElementSize() const
		{
			return c_ElementSize;
		}


		template < typename Type=SizeType >
		Type Length() const
		{
			return static_cast<Type>( m_Length );
		}


		SizeType Capacity() const
		{
			return m_Capacity;
		}


		SizeType AllocatedSize() const
		{
			return m_AllocSize;
		}


		bool Empty() const
		{
			return (!m_pData) | (m_Length==0);
		}


		T& Front() const
		{
			return *m_pData;
		}


		T& Back() const
		{
			auto tmp = end();
			--tmp;

			return *tmp;
		}




		// https://stackoverflow.com/questions/31581880/overloading-cbegin-cend
		// begin / end overload for "range-based for loop"
		inline T* begin()
		{
			return m_pData;
		}


		inline const T* begin() const
		{
			return m_pData;
		}


		inline T* end()
		{
			return begin() + m_Length;
		}


		inline const T* end() const
		{
			return begin() + m_Length;
		}



	protected:

		const SizeType c_ElementSize = (SizeType)sizeof(T);

		SizeType	m_Length;
		SizeType	m_AllocSize;
		T*			m_pData;

		SizeType	m_Capacity;


		// new delete memory without constructor
		// https://stackoverflow.com/questions/4576307/c-allocate-memory-without-activating-constructors/4576402
		inline T* AllocateBuffer( SizeType len, bool init=false )
		{
			// Allocate memory
			/*T* buffer*/m_pData = static_cast<T*>( ::operator new( c_ElementSize * len ) );
			
			// Call default constructor
			if( init )
			{
				for( auto iter=m_pData; iter !=m_pData+len; ++iter )
					new ( iter ) T();
			}

			return m_pData;
		}


		inline void DeallocateBuffer()
		{
			if( m_pData==nullptr )
				return;

			for( auto iter=m_pData; iter!=m_pData+m_Length; ++iter )	iter->~T();
			::operator delete( m_pData );

			m_pData = nullptr;
		}


		inline bool ReallocateBuffer( SizeType newlen )
		{
			if( newlen < m_Length )
			{
				for( auto iter=m_pData+newlen; iter!=m_pData+m_Capacity; ++iter )
					iter->~T();
			}
			else if( newlen > m_Capacity )
			{
				T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * newlen ) );

				if( m_pData )
				{
					MemMove( newdata, m_pData, Min(m_Length, newlen) );
					DeallocateBuffer();
				}
				m_Capacity	= newlen;
				m_pData		= newdata;
			}

			//for( SizeType i=m_Length; i<newlen; ++i )
			//	new ( m_pData+i ) T();

			m_Length	= newlen;
			m_AllocSize	= c_ElementSize * m_Length;

			return true;
		}



	};




	//##############################################################################################################//
	//																												//
	//												Helper funcitions												//
	//																												//
	//##############################################################################################################//


	template < typename T >
	inline sizeType Find( const Memory<T>& arr, const T& item )
	{
		for( const auto& elm : arr )
		{
			if( elm == item )
				return &elm - &arr[0];
		}

		return -1;
	}



	template < typename T >
	inline bool Exists( const Memory<T>& arr, const T& item )
	{
		for( const auto& elm : arr )
		{
			if( elm == item )
				return true;
		}

		return false;
	}



	template < typename T >
	inline bool Exists( sizeType numelms, const T data[], const T& item )
	{
		for( sizeType i=0; i<numelms; ++i )
		{
			if( data[i] == item )
				return true;
		}

		return false;
	}



	template < typename T, typename Predicate >
	inline int64 FindIf( const Memory<T>& arr, Predicate pred )
	{
		auto first = arr.begin();
		const auto last = arr.end();

		for(; first != last; ++first )
		{
			if( pred(*first) )
				return first - arr.begin();
		}

		return -1;
	}



}// end of namespace


#endif // !MEMORY_H
