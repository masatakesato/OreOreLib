#ifndef MEMORY_H
#define	MEMORY_H

//#include	<algorithm>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"
#include	"../algorithm/Algorithm.h"
#include	"MemoryOperations.h"



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
	//											MemoryBase class declaration										//
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
			, m_pData( AllocateBuffer( len ) )
		{
			//tcout << _T("MemoryBase constructor(dynamic allocation)...\n");
			for( auto iter=m_pData; iter !=m_pData+len; ++iter )
				new ( iter ) T();
		}

		
		// Constructor
		MemoryBase( IndexType len, const T& fill )
			: m_Length( len )
			, m_AllocSize( len * c_ElementSize )
			, m_Capacity( len )
			, m_pData( AllocateBuffer( len ) )
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
			Mem::UninitializedCopy( begin(), ilist.begin(), m_Length );
		}


		// Constructor with initializer_list
		MemoryBase( std::initializer_list<T> ilist )
			: m_Length( static_cast<IndexType>( ilist.size() ) )
			, m_AllocSize( static_cast<IndexType>( ilist.size() * sizeof(T) ) )
			, m_Capacity( m_Length )
			, m_pData( AllocateBuffer( m_Length ) )
		{
			Mem::UninitializedCopy( begin(), ilist.begin(), m_Length );
		}


		// Constructor with iterator
		template < class Iter >
		MemoryBase( Iter first, Iter last )
			: m_Length( static_cast<IndexType>( last - first ) )
			, m_AllocSize( m_Length * sizeof(T) )
			, m_Capacity( m_Length )
			, m_pData( AllocateBuffer( m_Length ) )
		{
			Mem::UninitializedCopy( begin(), first, m_Length );
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
				m_pData = AllocateBuffer( m_Capacity );
				Mem::UninitializedCopy( m_pData, obj.m_pData, Min(m_Length, obj.m_Length) );
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
					m_pData = AllocateBuffer( m_Capacity );
					Mem::UninitializedCopy( m_pData, obj.m_pData, Min(m_Length, obj.m_Length) );
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
			// Reallocate buffer
			ReallocateBuffer( len );

			// Initialize
			Mem::UninitializedInit( m_pData, len );
		}


		void Init( IndexType len, const T& fill )
		{
			// Reallocate buffer
			ReallocateBuffer( len );

			// Fill
			for( auto iter=m_pData; iter !=m_pData+len; ++iter )
				new ( iter ) T( fill );
		}


		void Init( std::initializer_list<T> ilist )
		{
			// Reallocate buffer
			ReallocateBuffer( static_cast<IndexType>( ilist.size() ) );

			// Copy data
			Mem::UninitializedCopy( begin(), ilist.begin(), ilist.size() );
		}


		// Constructor with iterator
		template < class Iter >
		void Init( Iter first, Iter last )
		{
			// Reallocate buffer
			ReallocateBuffer( static_cast<IndexType>( last - first ) );

			// Copy data
			Mem::UninitializedCopy( begin(), first, m_Length );
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
			Mem::Clear( m_pData, m_Length );
		}


		void SetValues( uint8* pdata, IndexType len )
		{
			ASSERT( pdata );
			Mem::Copy( m_pData, (T*)pdata, Min( m_Length, len ) );
		}


		template < typename ... Args >
		std::enable_if_t< TypeTraits::all_convertible<T, Args...>::value, void >
		SetValues( const Args& ... args )
		{
			auto values = { (T)args... };
			Mem::Copy( m_pData, values.begin(), Min( m_Length, (IndexType)values.size() ) );
		}


		template < typename Type >
		std::enable_if_t< std::is_convertible_v<Type, T>/*  std::is_same_v<Type, T>*/, void >
		SetValues( std::initializer_list<Type> ilist )
		{
			Mem::Copy( m_pData, ilist.begin(), Min( m_Length, (IndexType)ilist.size() ) );
		}


		inline bool Reserve( IndexType newlen )
		{
			if( newlen <= m_Capacity )
				return false;

			m_Capacity	= newlen;
			T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * m_Capacity ) );

			if( m_pData )
			{
				Mem::UninitializedMigrate( newdata, m_pData, m_Length );
				DeallocateBuffer();
			}

			m_pData = newdata;

			return true;
		}


		inline bool Resize( IndexType newlen )
		{
			if( newlen==0 || newlen==~0u )	return false;

			auto oldlen = m_Length;
			if( !MigrateBuffer( newlen ) )
				return false;

			// placement new uninitialized elements
			for( auto i=oldlen; i<newlen; ++i )
				new ( m_pData + i ) T();

			return true;
		}


		inline bool Resize( IndexType newlen, const T& fill )
		{
			if( newlen==0 || newlen==~0u )	return false;

			auto oldlen = m_Length;
			if( !MigrateBuffer( newlen ) )
				return false;

			// Fill uninitialized elements using placement new copy constructor
			for( auto i=oldlen; i<newlen; ++i )
				new ( m_pData + i ) T( fill );

			return true;
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


		inline void LeftShiftElements( IndexType idx, IndexType numelms, IndexType shift )
		{
			if( idx < shift || shift == 0 )
				return;

			T* pSrc = m_pData + idx;// 移動元要素イテレータ.
			T* pDst = pSrc - shift;// 移動先要素イテレータ

			while( pSrc <= m_pData + Min( idx + numelms - 1 + shift, m_Length - 1 ) )//end() )
			{
				pDst->~T();// destruct dst data first 
				new ( pDst ) T( (T&&)( *pSrc ) );// then move src data

				++pDst;
				++pSrc;
			}

			// destruct empty elements
			for( IndexType i=0; i<shift; ++i )
				(pDst++)->~T();
		}


		inline void RightShiftElements( IndexType idx, IndexType numelms, IndexType shift )
		{
			if( m_Length <= ( idx + shift ) || shift == 0 )
				return;

			T* pDst = m_pData + Min( idx + numelms - 1 + shift, m_Length - 1 );// 移動先要素イテレータ. m_Lengthからはみ出ないようにする
			T* pSrc = pDst - shift;// 移動元イテレータ

			while( pSrc >= m_pData + idx )// Start shift from the last element
			{
				pDst->~T();// destruct dst data first 
				new ( pDst ) T( (T&&)( *pSrc ) );// then move src data

				--pDst;
				--pSrc;
			}

			//tcout << pDst << tendl;
			//tcout << m_pData + idx + shift - 1 << tendl;

			// destruct empty elements
			for( IndexType i=0; i<shift; ++i )
				(pDst--)->~T();//(m_pData + idx + i)->~T();
		}


		inline void CopyFrom( const MemoryBase& src )
		{
			Mem::Copy( m_pData, src.m_pData, Min(m_Length, src.m_Length) );
		}


		inline void CopyTo( MemoryBase& dst ) const
		{
			Mem::Copy( dst.m_pData, m_pData, Min(m_Length, dst.m_Length) );
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
		inline T* AllocateBuffer( IndexType newlen )
		{
			return static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
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


		inline bool ReallocateBuffer( IndexType newlen )
		{	
			if( newlen > m_Capacity )// Allocate new buffer if capacity is short
			{
				DeallocateBuffer();

				m_pData = static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
				if( !m_pData )
				{
					m_Length	= 0;
					m_Capacity	= 0;
					m_AllocSize	= 0;
					return false;
				}

				m_Capacity	= newlen;
			}
			else// Clear existing buffer
			{
				Mem::Clear( m_pData, m_Length );
			}

			// Update m_Length and m_AllocSize
			m_Length	= newlen;
			m_AllocSize	= c_ElementSize * newlen;

			return true;
		}


		inline bool MigrateBuffer( IndexType newlen )
		{
			// Reallocate buffer if capacity is short
			//   m_Length    m_Capacity   newlen
			// -----*------------|----------x
			if( newlen > m_Capacity )
			{
				T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
				if( !newdata )	return false;
				
				if( m_pData )
				{
					Mem::UninitializedMigrate( newdata, m_pData, Min(m_Length, newlen) );
					DeallocateBuffer();
				}
				m_Capacity	= newlen;
				m_pData		= newdata;
			}

			// Delete out-of-bound elements ( if newlen is smaller than m_Length )
			//    newlen    m_Length    m_Capacity
			// -----x-----------*------------|
			for( auto iter=m_pData+newlen; iter<m_pData+m_Length; ++iter )
				iter->~T();

			// Update m_Length and m_AllocSize
			m_Length	= newlen;
			m_AllocSize	= c_ElementSize * newlen;

			return true;
		}


		
		//inline bool MigrateBuffer( IndexType newlen, bool init, const T* fill=nullptr )
		//{
		//	//   m_Length    m_Capacity   newlen
		//	// -----*------------|----------x
		//	if( newlen > m_Capacity )
		//	{
		//		T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
		//		
		//		if( init )
		//		for( auto i=m_Length; i<newlen; ++i )
		//			fill ?
		//			new ( newdata + i ) T( *fill ) :
		//			new ( newdata + i ) T();

		//		if( m_pData )
		//		{
		//			Mem::UninitializedMigrate( newdata, m_pData, Min(m_Length, newlen) );
		//			DeallocateBuffer();
		//		}
		//		m_Capacity	= newlen;
		//		m_pData		= newdata;
		//	}

		//	//   m_Length    newlen     m_Capacity
		//	// -----*-----------x------------|
		//	else if( newlen > m_Length )
		//	{
		//		if( init )
		//		for( auto iter=m_pData+m_Length; iter!=m_pData+newlen; ++iter )
		//			fill ?
		//			new ( iter ) T( *fill ) :
		//			new ( iter ) T();
		//	}

		//	//    newlen    m_Length    m_Capacity
		//	// -----x-----------*------------|
		//	else if( newlen < m_Length )
		//	{
		//		for( auto iter=m_pData+newlen; iter!=m_pData+m_Length; ++iter )
		//			iter->~T();
		//	}

		//	m_Length	= newlen;
		//	m_AllocSize	= c_ElementSize * m_Length;

		//	return true;
		//}


		// Create new memory space in the middle of m_pData. Allocated new space is UNINITIALIZED.
		inline bool AllocateInsertionSpace( IndexType elm, IndexType numelms=1 )
		{
			IndexType newlen = m_Length + numelms;
			ASSERT( elm < newlen );

			//=============== Consume reserved area if available =============//
			if( newlen <= m_Capacity )
			{
				auto nummoveelms = m_Length - elm;

				// m_pData[ m_Length, newlen-1] の、シフト足のはみ出た要素だけ初期化する
				if( nummoveelms > 0 )
					for( auto iter=m_pData+m_Length; iter!=m_pData+newlen; ++iter )	new ( iter ) T();

				// Update m_Length/m_AllocSize
				m_Length	= newlen;
				m_AllocSize	= c_ElementSize * m_Length;

				// Finally right shift elements and create insertion space.
				RightShiftElements( elm, nummoveelms, numelms );

				return true;
			}


			//=============== Allocate new memory otherwise =================//
			T* newdata	= static_cast<T*>( ::operator new( c_ElementSize * newlen ) );
			if( !newdata )
				return false;

			if( m_pData )
			{
				// Move m_pData[ 0 : elm-1 ] to newdata[ 0 : elm-1 ]
				Mem::UninitializedMigrate( &newdata[ 0 ], &m_pData[ 0 ], elm );
				// Move m_pData[ elm : m_Length-1 ] to newdata[ elm + numelms : ... ]
				Mem::UninitializedMigrate( &newdata[ elm + numelms ], &m_pData[ elm ], m_Length - elm );

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
