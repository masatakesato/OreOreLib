#ifndef MEMORY_H
#define	MEMORY_H

//#include	<algorithm>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"
#include	"../algorithm/Algorithm.h"



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




	//##############################################################################################################//
	//																												//
	//											MemoryBase class declaration											//
	//																												//
	//##############################################################################################################//

	template< typename T, typename IndexType >	struct MemoryBase;




	//##############################################################################################################//
	//																												//
	//											Memory partial specialization										//
	//																												//
	//##############################################################################################################//

	template < typename T >
	using Memory = MemoryBase< T, MemSizeType >;





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
			//out->~DstIter();// Destruct existing data from destination memory
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
	//		//out->~Iter();// Destruct existing data from destination memory
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
// 未初期化知領域にデータを新規移動する場合がある
// 既存領域データを削除した上書きする場合もある
			//out->~DstIter();// Destruct existing data
			new ( out ) DstIter( (DstIter&&)( *begin ) );// Overwite existing memory with placement new

			// Copy assignment operator version. cannot deal with dynamic memory object( e.g., string )
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
	//										MemoryBase class implementation											//
	//																												//
	//##############################################################################################################//

	template< typename T, typename IndexType >
	struct MemoryBase
	{
	public:

		// Default constructor
		MemoryBase()
			: m_Length( 0 )
			, m_AllocSize( 0 )
			, m_Capacity( 0 )
			, m_pData( nullptr )
		{
			//tcout << _T("MemoryBase default constructor...\n");
		}


		// Constructor
		MemoryBase( IndexType len )
			: m_Length( len )
			, m_AllocSize( len * c_ElementSize )
			, m_Capacity( len )
			, m_pData( AllocateBuffer(len, true) )
		{
			//tcout << _T("MemoryBase constructor(dynamic allocation)...\n");
		}

		
		// Constructor
		MemoryBase( IndexType len, const T& fill )
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
		MemoryBase( Args const & ... args )
			: m_Length( sizeof ...(Args) )
			, m_AllocSize( sizeof ...(Args) * sizeof(T) )
			, m_Capacity( sizeof ...(Args) )
			, m_pData( AllocateBuffer( static_cast<IndexType>( sizeof ...(Args) ) )  )
		{
			auto ilist = std::initializer_list<T>{ args... };
			MemCopy( begin(), ilist.begin(), m_Length );

			//auto p = m_pData;
			//for( const auto& val : std::initializer_list<T>{args...} )
			//	*(p++) = val;
		}

		// Constructor with initializer_list
		MemoryBase( std::initializer_list<T> ilist )
			: m_Length( static_cast<IndexType>( ilist.size() ) )
			, m_AllocSize( static_cast<IndexType>( ilist.size() * sizeof(T) ) )
			, m_Capacity( m_Length )
			, m_pData( AllocateBuffer( m_Length ) )
		{
			MemCopy( begin(), ilist.begin(), m_Length );
			
			//auto p = m_pData;
			//for( const auto& val : ilist )
			//	*(p++) = val;
		}


		// Constructor with iterator
		template < class Iter >
		MemoryBase( Iter first, Iter last )
			: m_Length( static_cast<IndexType>( last - first ) )
			, m_AllocSize( m_Length * sizeof(T) )
			, m_Capacity( m_Length )
			, m_pData( AllocateBuffer( m_Length ) )
		{
			MemCopy( begin(), first, m_Length );
		}


		// Destructor
		virtual ~MemoryBase()
		{
			//tcout << _T("MemoryBase destructor...\n");
			Release();
		}


		// Copy constructor
		MemoryBase( const MemoryBase& obj )
			: m_Length( obj.m_Length )
			, m_AllocSize( obj.m_AllocSize )
			, m_Capacity( obj.m_Capacity )
			, m_pData( nullptr )
		{
			//tcout << _T("MemoryBase copy constructor...\n");
			if( obj.m_pData )
			{
				AllocateBuffer( m_Capacity );
				MemCopy( m_pData, obj.m_pData, Min(m_Length, obj.m_Length) );
			}
		}


		// Move constructor
		MemoryBase( MemoryBase&& obj )
			: m_Length( obj.m_Length )
			, m_AllocSize( obj.m_AllocSize )
			, m_Capacity( obj.m_Capacity )
			, m_pData( obj.m_pData )

		{
			//tcout << _T("MemoryBase move constructor...\n");

			obj.m_Length	= 0;
			obj.m_AllocSize	= 0;
			obj.m_Capacity	= 0;
			obj.m_pData		= nullptr;// clear reference from obj
		}


		// Copy Assignment operator =
		inline MemoryBase& operator=( const MemoryBase& obj )
		{
			if( this != &obj )
			{
				//tcout << _T("MemoryBase copy assignment operator...\n");
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
		inline MemoryBase& operator=( MemoryBase&& obj )
		{
			if( this != &obj )
			{
				//tcout << _T("MemoryBase move assignment operator...\n");

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


		// Subscript operator for read only.( called if MemoryBase is const )
		inline const T& operator[]( IndexType n ) const&
		{
			return m_pData[n];
		}


		// Subscript operator for read-write.( called if MemoryBase is non-const )
		inline T& operator[]( IndexType n ) &
		{
			return m_pData[n];
		}


		// Subscript operator. ( called by following cases: "T a = MemoryBase<T,IndexType>(10)[n]", "auto&& a = MemoryBase<T,IndexType>(20)[n]" )
		inline T operator[]( IndexType n ) const&&
		{
			return std::move(m_pData[n]);// return object
		}


		inline operator bool() const
		{
			return m_pData != nullptr;
		}


		inline bool operator==( const MemoryBase& rhs ) const
		{
			return m_pData == rhs.m_pData;
		}


		inline bool operator !=( const MemoryBase& rhs ) const
		{
			return m_pData != rhs.m_pData;
		}



		void Init( IndexType len )
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


		void Init( IndexType len, const T& fill )
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
			IndexType len = static_cast<IndexType>( ilist.size() );

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
			IndexType len = static_cast<IndexType>( last - first );

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


		void SetValues( uint8* pdata, IndexType len )
		{
			ASSERT( pdata );
			MemCopy( m_pData, (T*)pdata, Min( m_Length, len ) );
		}


		template < typename ... Args >
		std::enable_if_t< TypeTraits::all_convertible<T, Args...>::value, void >
		SetValues( const Args& ... args )
		{
			auto values = { (T)args... };
			MemCopy( m_pData, values.begin(), Min( m_Length, (IndexType)values.size() ) );
		}


		template < typename Type >
		std::enable_if_t< std::is_convertible_v<Type, T>/*  std::is_same_v<Type, T>*/, void >
		SetValues( std::initializer_list<Type> ilist )
		{
			MemCopy( m_pData, ilist.begin(), Min( m_Length, (IndexType)ilist.size() ) );
		}


		inline bool Reserve( IndexType newlen )
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


		inline bool Resize( IndexType newlen )
		{
			if( newlen==0 || newlen==~0u )	return false;
			return ReallocateBuffer( newlen );
		}


		inline bool Resize( IndexType newlen, const T& fill )
		{
			if( newlen==0 || newlen==~0u )	return false;
			return ReallocateBuffer( newlen, &fill );
		}


		inline IndexType InsertBefore( IndexType elm )
		{
			AllocateInsertionSpace( elm );

			// Init newdata[ elm ]
			T* val = new ( &m_pData[elm] ) T();//newdata[elm] = T();//

			return elm;
		}


		inline IndexType InsertBefore( IndexType elm, const T& src )
		{
			AllocateInsertionSpace( elm );

			// Copy src to newdata[ elm ]
			T* val = new ( &m_pData[elm] ) T(src);//newdata[elm] = src;//

			return elm;
		}


		inline IndexType InsertBefore( IndexType elm, T&& src )
		{
			AllocateInsertionSpace( elm );

			// Move src to newdata[ elm ]
			T* val = new ( &m_pData[elm] ) T( (T&&)src );//newdata[elm] = (T&&)src;//

			return elm;
		}

		
		inline IndexType InsertAfter( IndexType elm )
		{
			return InsertBefore( elm + 1 );
		}


		inline IndexType InsertAfter( IndexType elm, const T& src )
		{
			return InsertBefore( elm+1, src );
		}


		inline IndexType InsertAfter( IndexType elm, T&& src )
		{
			return InsertBefore( elm+1, (T&&)src );
		}


		inline void CopyFrom( const MemoryBase& src )
		{
			MemCopy( m_pData, src.m_pData, Min(m_Length, src.m_Length) );
		}


		inline void CopyTo( MemoryBase& dst ) const
		{
			MemCopy( dst.m_pData, m_pData, Min(m_Length, dst.m_Length) );
		}


		inline IndexType ElementSize() const
		{
			return c_ElementSize;
		}


		template < typename Type=IndexType >
		inline Type Length() const
		{
			return static_cast<Type>( m_Length );
		}


		inline IndexType Capacity() const
		{
			return m_Capacity;
		}


		inline IndexType AllocatedSize() const
		{
			return m_AllocSize;
		}


		inline bool Empty() const
		{
			return (!m_pData) | (m_Length==0);
		}


		inline T& Front()
		{
			return *m_pData;
		}


		inline const T& Front() const
		{
			return *m_pData;
		}


		inline T& Back()
		{
			auto tmp = end();
			--tmp;

			return *tmp;
		}


		inline const T& Back() const
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

		const IndexType c_ElementSize = (IndexType)sizeof(T);

		IndexType	m_Length;
		IndexType	m_AllocSize;
		T*			m_pData;

		IndexType	m_Capacity;


		// new delete memory without constructor
		// https://stackoverflow.com/questions/4576307/c-allocate-memory-without-activating-constructors/4576402
		inline T* AllocateBuffer( IndexType len, bool init=false )
		{
			// Allocate memory
			/*T* buffer*/m_pData = static_cast<T*>( ::operator new( c_ElementSize * len ) );
			//memset( m_pData, 0,  c_ElementSize * len );
			
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

			for( auto iter=m_pData; iter!=m_pData+m_Length; ++iter )
				iter->~T();
			::operator delete( m_pData );

			m_pData = nullptr;
		}


		inline bool ReallocateBuffer( IndexType newlen, const T* fill=nullptr )
		{
			//   m_Length    m_Capacity   newlen
			// -----*------------|----------x
			if( newlen > m_Capacity )
			{
				T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
				//memset( newdata, 0,  c_ElementSize * newlen );
				for( auto i=m_Length; i<newlen; ++i )
					fill ?
					new ( newdata + i ) T( *fill ) :
					new ( newdata + i ) T();

				if( m_pData )
				{
					MemMove( newdata, m_pData, Min(m_Length, newlen) );
					DeallocateBuffer();
				}
				m_Capacity	= newlen;
				m_pData		= newdata;
			}

			//   m_Length    newlen     m_Capacity
			// -----*-----------x------------|
			else if( newlen > m_Length )
			{
				for( auto iter=m_pData+m_Length; iter!=m_pData+newlen; ++iter )
					fill ?
					new ( iter ) T( *fill ) :
					new ( iter ) T();
			}

			//    newlen    m_Length    m_Capacity
			// -----x-----------*------------|
			else if( newlen < m_Length )
			{
				for( auto iter=m_pData+newlen; iter!=m_pData+m_Length; ++iter )
					iter->~T();
			}

			m_Length	= newlen;
			m_AllocSize	= c_ElementSize * m_Length;

			return true;
		}


		// Create new memory space in the middle of m_pData. Allocated new space is UNINITIALIZED.
		inline bool AllocateInsertionSpace( IndexType elm, IndexType numelms=1 )
		{
			IndexType newlen = m_Length + numelms;
			ASSERT( elm < newlen );

			T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
			if( !newdata )
				return false;

			if( m_pData )
			{
				// Move m_pData[ 0 : elm-1 ] to newdata[ 0 : elm-1 ]
				MemMove( &newdata[ 0 ], &m_pData[ 0 ], elm );
				// Move m_pData[ elm : m_Length-1 ] to newdata[ elm + numelms : ... ]
				MemMove( &newdata[ elm + numelms ], &m_pData[ elm ], m_Length - elm );

				DeallocateBuffer();
			}

			m_Capacity	= newlen;
			m_pData		= newdata;
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


	template < typename T, typename IndexType >
	inline IndexType Find( const MemoryBase<T, IndexType>& arr, const T& item )
	{
		for( const auto& elm : arr )
		{
			if( elm == item )
				return static_cast<IndexType>( &elm - &arr[0] );
		}

		return -1;
	}



	template < typename T, typename IndexType >
	inline bool Exists( const MemoryBase<T, IndexType>& arr, const T& item )
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



	template < typename T, typename IndexType, typename Predicate >
	inline IndexType FindIf( const MemoryBase<T, IndexType>& arr, Predicate pred )
	{
		auto first = arr.begin();
		const auto last = arr.end();

		for(; first != last; ++first )
		{
			if( pred(*first) )
				return static_cast<IndexType>( first - arr.begin() );
		}

		return -1;
	}



}// end of namespace


#endif // !MEMORY_H
