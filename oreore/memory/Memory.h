﻿#ifndef MEMORY_H
#define	MEMORY_H

//#include	<algorithm>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"

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
	//										MemCopy / MemMove (above C++17)											//
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
				*out = *(DstIter*)begin;
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
	//			*out = *begin;
	//			++begin; ++out;
	//		}
	//
	//		return out;
	//	}
	//}



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
				*out = *(DstIter*)begin;
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
	//			*out = *begin;
	//			++begin; ++out;
	//		}

	//		return out;
	//	}
	//}



	#else
	

	//##############################################################################################################//
	//																												//
	//										MemCopy / MemMove (below C++14)											//
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
			*out = *(DstIter*)begin;
			++begin; ++out;
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
	//		*out = *begin;
	//		++begin; ++out;
	//	}
	//
	//	return out;
	//}



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
			*out = *(DstIter*)begin;
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
	//		*out = *begin;
	//		++begin; ++out;
	//	}
	//
	//	return out;
	//}



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
		Memory( SizeType len, T* pdata=nullptr )
			: m_Length( len )
			, m_AllocSize( len * sizeof(T) )
			, m_Capacity( len )
			, m_pData( new T[len]() )
		{
			//tcout << _T("Memory constructor(dynamic allocation)...\n");

			if( pdata )
				MemCopy( m_pData, pdata, m_Length );
		}

		
		// Constructor
		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		Memory( Args const & ... args )
			: m_Length( sizeof ...(Args) )
			, m_AllocSize( sizeof ...(Args) * sizeof(T) )
			, m_Capacity( sizeof ...(Args) )
			, m_pData( new T[ sizeof ...(Args) ]{args...} )
		{
			
		}

		// Constructor with initializer_list
		Memory( std::initializer_list<T> ilist )
			: m_Length( SizeType( ilist.size() ) )
			, m_AllocSize( SizeType( ilist.size() * sizeof(T) ) )
			, m_Capacity( SizeType( ilist.size() ) )
			, m_pData( new T[ ilist.size() ] )
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
			, m_pData( new T[ last - first ]() )
		{
			auto p = m_pData;
			for(; first != last; ++first )
				*(p++) = T(*first);
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
				m_pData = new T[ m_Capacity ];
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
				SafeDeleteArray( m_pData );
				
				if( obj.m_pData )
				{
					m_pData = new T[ m_Capacity ];
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
				SafeDeleteArray( m_pData );

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



		void Init( SizeType len, T* pdata=nullptr )
		{
			m_Length	= len;
			m_AllocSize	= c_ElementSize * len;

			if( m_Length > m_Capacity )
			{
				SafeDeleteArray( m_pData );
				m_Capacity	= m_Length;
				m_pData		= new T[ m_Capacity ]();
			}

			if( pdata )
			{
				MemCopy( m_pData, pdata, m_Length );
			}


			//SafeDeleteArray( m_pData );

			//m_Length	= len;
			//m_AllocSize	= c_ElementSize * len;
			//m_Capacity	= len;
			//m_pData		= new T[ m_Capacity ]();

			//if( pdata )
			//	MemCopy( m_pData, pdata, m_Length );
		}


		void Init( std::initializer_list<T> ilist )
		{
			m_Length	= static_cast<SizeType>( ilist.size() );
			m_AllocSize	= c_ElementSize * m_Length;

			if( m_Length > m_Capacity )
			{
				SafeDeleteArray( m_pData );
				m_Capacity	= m_Length;
				m_pData		= new T[ m_Capacity ]();
			}

			MemCopy( begin(), ilist.begin(), ilist.size() );



			//SafeDeleteArray( m_pData );

			//m_Length	= ilist.size();
			//m_AllocSize	= m_Length * sizeof(T);
			//m_Capacity	= ilist.size();
			//m_pData		= new T[ m_Capacity ];	

			//MemCopy( begin(), ilist.begin(), ilist.size() );
		}


		void Init( SizeType len, const T& fill )
		{
			m_Length	= len;
			m_AllocSize	= c_ElementSize * len;

			if( m_Length > m_Capacity )
			{
				SafeDeleteArray( m_pData );
				m_Capacity	= m_Length;
				m_pData		= new T[ m_Capacity ]();
			}

			for( auto& data : m_pData )
				data = fill;
		}



		void Release()
		{
			m_Length	= 0;
			m_AllocSize	= 0;
			m_Capacity	= 0;
			SafeDeleteArray( m_pData );
		}


		void Clear()
		{
			if( m_Length > 0 )
				memset( m_pData, 0, m_AllocSize );
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



		inline bool Resize( SizeType newlen )
		{
			if( newlen < m_Length )
			{
				for( auto iter=m_pData+m_Length; iter !=m_pData+m_Capacity; ++iter )
					iter->~T();
				//for( SizeType i=m_Length; i<newlen; ++i )	m_pData[i].~T();
			}
			else if( newlen > m_Capacity )
			{
				T *newdata	= new T[ newlen ]();
				SizeType newallocsize = c_ElementSize * newlen;

				if( m_pData )
				{
					MemCopy( newdata, m_pData, Min(m_Length, newlen) );
					SafeDeleteArray( m_pData );
				}
				m_Capacity	= newlen;
				m_pData		= newdata;
			}

			m_Length	= newlen;
			m_AllocSize	= c_ElementSize * m_Length;

			return true;


			//if( newlen <= m_Capacity )
			//{
			//	for( auto& iter=m_pData+m_Length; iter !=m_pData+m_Capacity; ++iter )	iter.~T();//for( SizeType i=m_Length; i<newlen; ++i )	m_pData[i].~T();

			//	m_Length	= newlen;
			//	m_AllocSize	= c_ElementSize * newlen;
			//}
			//else
			//{
			//	T *newdata	= new T[ newlen ]();
			//	SizeType newallocsize = c_ElementSize * newlen;

			//	if( m_pData )
			//	{
			//		MemCopy( newdata, m_pData, Min(m_Length, newlen) );
			//		SafeDeleteArray( m_pData );
			//	}

			//	m_pData		= newdata;
			//	m_Length	= newlen;
			//	m_Capacity	= newlen;
			//	m_AllocSize	= newallocsize;
			//}

			//return true;
		}


		inline bool Resize( SizeType newlen, const T& fill )
		{
			if( newlen < m_Length )
			{
				for( auto iter=m_pData+m_Length; iter !=m_pData+m_Capacity; ++iter )
					iter->~T();
				//for( SizeType i=m_Length; i<newlen; ++i )	m_pData[i].~T();
			}
			else if( newlen > m_Capacity )
			{
				T *newdata	= new T[ newlen ]();
				SizeType newallocsize = c_ElementSize * newlen;

				if( m_pData )
				{
					MemCopy( newdata, m_pData, Min(m_Length, newlen) );
					SafeDeleteArray( m_pData );
				}
				m_Capacity	= newlen;
				m_pData		= newdata;
			}

			for( SizeType i=m_Length; i<newlen; ++i )
				m_pData[i] = fill;

			m_Length	= newlen;
			m_AllocSize	= c_ElementSize * m_Length;

			return true;



			//if( newlen <= m_Capacity )
			//{
			//	for( SizeType i=m_Length; i<newlen; ++i )
			//		m_pData[i] = fill;

			//	m_Length	= newlen;
			//	m_AllocSize	= c_ElementSize * newlen;
			//}
			//else
			//{
			//	T *newdata	= new T[ newlen ]();
			//	SizeType newallocsize = c_ElementSize * newlen;

			//	if( m_pData )
			//	{
			//		MemCopy( newdata, m_pData, Min(m_Length, newlen) );
			//		SafeDeleteArray( m_pData );
			//	}

			//	for( SizeType i=m_Length; i<newlen; ++i )
			//		newdata[i] = fill;

			//	m_pData		= newdata;
			//	m_Length	= newlen;
			//	m_AllocSize	= newallocsize;
			//}

			//return true;
		}


		inline bool Reserve( SizeType newlen )
		{
			if( newlen <= m_Capacity )
				return false;

			m_Capacity	= newlen;
			T* newdata	= new T[ m_Capacity ]();

			if( m_pData )
			{
				MemCopy( newdata, m_pData, m_Length );
				SafeDeleteArray( m_pData );
			}

			m_pData = newdata;

			return true;
		}


		inline bool Extend( SizeType numelms )
		{
			return Resize( m_Length + numelms );
		}


		inline bool Shrink( SizeType numelms )
		{
			if( m_Length > numelms )	return Resize( m_Length - numelms );
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


		SizeType Length() const
		{
			return m_Length;
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



	template < class ForwardIter, class T >
	inline void Fill( ForwardIter first, ForwardIter last, const T& value )
	{
		while( first != last )
			*first++ = value;
	}





}// end of namespace


#endif // !MEMORY_H
