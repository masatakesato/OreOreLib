#ifndef MEMORY_H
#define	MEMORY_H

#include	<algorithm>

#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"



namespace OreOreLib
{


	#if __cplusplus >= 201703L
	// If using Visual c++, folowing command must be added for __cplusplus macro activation.
	//   /Zc:__cplusplus

	//##############################################################################################################//
	//																												//
	//										MemCopy / MemMove (above C++17)											//
	//																												//
	//##############################################################################################################//

	// Memory Copy
	template < class Iter >
	Iter* MemCopy( Iter* pDst, Iter* pSrc, size_t size )
	{
		if constexpr ( std::is_trivially_copyable_v<Iter> )
		{
			return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
		}
		else
		{
			Iter* begin = pSrc;
			Iter* end = pSrc + size;
			Iter* out = pDst;

			while( begin != end )
			{
				*out = *begin;
				++begin; ++out;
			}
    
			return out;
		}
	}



	// Memory Move
	template < class Iter >
	Iter* MemMove( Iter* pDst, Iter* pSrc, size_t size )
	{
		if constexpr ( std::is_trivially_copyable_v<Iter> )
		{
			return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
		}
		else
		{
			Iter* begin = pSrc;
			Iter* end = pSrc + size;
			Iter* out = pDst;

			while(begin != end)
			{
				*out = *begin;
				++begin; ++out;
			}

			return out;
		}
	}


	#else
	

	//##############################################################################################################//
	//																												//
	//										MemCopy / MemMove (below C++14)											//
	//																												//
	//##############################################################################################################//

	// Trivial Memcpy
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemCopy( Iter* pDst, const Iter* pSrc, size_t size )
	{
		return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
	}

	// Non-Trivial Memcpy
	template < class Iter >
	std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	MemCopy( Iter* pDst, const Iter* pSrc, size_t size )
	{
		Iter* begin = (Iter*)pSrc;
		const Iter* end = pSrc + size;
		Iter* out = pDst;

		while( begin != end )
		{
			*out = *begin;
			++begin; ++out;
		}
    
		return out;
	}



	// Trivial MemMove
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemMove( Iter* pDst, const Iter* pSrc, size_t size )
	{
		return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
	}

	// Non-Trivial MemMove
	template < class Iter >
	std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	MemMove( Iter* pDst, const Iter* pSrc, size_t size )
	{
		Iter* begin = (Iter*)pSrc;
		const Iter* end = pSrc + size;
		Iter* out = pDst;

		while( begin != end )
		{
			*out = *begin;
			++begin; ++out;
		}
    
		return out;
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
	public:

		// Default constructor
		Memory()
			: m_Length( 0 )
			, m_AllocSize( 0 )
			, m_pData( nullptr )
		{
			//tcout << _T("Memory default constructor...\n");
		}


		// Constructor
		Memory( int len, T* pdata=nullptr )
			: m_Length( len )
			, m_AllocSize( len * sizeof(T) )
			, m_pData( new T[len]() )
		{
			//tcout << _T("Memory constructor(dynamic allocation)...\n");
			assert( len > 0 );

			if( pdata )
				MemCopy( m_pData, pdata, m_Length );
		}

		
		// Constructor
		template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		Memory( Args const & ... args )
			: m_pData( new T[ sizeof ...(Args) ]{args...} )
			, m_Length( sizeof ...(Args) )
			, m_AllocSize( sizeof ...(Args) * sizeof(T) )
		{
			
		}

		// Constructor with initializer_list
		Memory( std::initializer_list<T> ilist )
			: m_pData( new T[ ilist.size() ] )
			, m_Length( int(ilist.size()) )
			, m_AllocSize( int(ilist.size()) * sizeof(T) )
		{
			auto p = m_pData;
			for( const auto& val : ilist )
				*(p++) = val;
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
			, m_pData( nullptr )
		{
			//tcout << _T("Memory copy constructor...\n");
			if( obj.m_pData )
			{
				m_pData = new T[ m_Length ];
				MemCopy( m_pData, obj.m_pData, Min(m_Length, obj.m_Length) );
			}
		}


		// Move constructor
		Memory( Memory&& obj )
			: m_Length( obj.m_Length )
			, m_AllocSize( obj.m_AllocSize )
			, m_pData( obj.m_pData )
		{
			//tcout << _T("Memory move constructor...\n");

			obj.m_Length = 0;
			obj.m_AllocSize = 0;
			obj.m_pData	= nullptr;// clear reference from obj
		}


		// Copy Assignment operator =
		inline Memory& operator=( const Memory& obj )
		{
			if( this != &obj )
			{
				//tcout << _T("Memory copy assignment operator...\n");
				m_Length		= obj.m_Length;
				m_AllocSize		= obj.m_AllocSize;
				SafeDeleteArray( m_pData );
				
				if( obj.m_pData )
				{
					m_pData = new T[ m_Length ];
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
				m_pData			= obj.m_pData;

				// clear reference from obj
				obj.m_Length	= 0;
				obj.m_AllocSize	= 0;
				obj.m_pData		= nullptr;
			}

			return *this;
		}


		// Subscript operator for read only.( called if Memory is const )
		inline const T& operator[]( std::size_t n ) const&
		{
			return m_pData[n];
		}


		// Subscript operator for read-write.( called if Memory is non-const )
		inline T& operator[]( std::size_t n ) &
		{
			return m_pData[n];
		}


		// Subscript operator. ( called by following cases: "T a = Memory<T>(10)[n]", "auto&& a = Memory<T>(20)[n]" )
		inline T operator[]( std::size_t n ) const&&
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



		void Init( int len, T* pdata=nullptr )
		{
			assert( len>0 );
	
			SafeDeleteArray( m_pData );

			m_Length	= len;
			m_AllocSize	= c_ElementSize * len;
			m_pData		= new T[len]();

			if( pdata )
				MemCopy( m_pData, pdata, m_Length );
		}


		//template < typename ... Args, std::enable_if_t< TypeTraits::all_same<T, Args...>::value>* = nullptr >
		//void Init( Args const & ... args )
		//{
		//	SafeDeleteArray( m_pData );

		//	m_Length	= sizeof ...(Args);
		//	m_AllocSize	= sizeof ...(Args) * sizeof(T);
		//	m_pData		= new T[ sizeof ...(Args) ]{args...};
		//	
		//}


		void Init( std::initializer_list<T> ilist )
		{
			SafeDeleteArray( m_pData );

			m_Length	= int( ilist.size() );
			m_AllocSize	= m_Length * sizeof(T);
			m_pData		= new T[ m_Length ];	

			//auto p = m_pData;
			//for( const auto& val : ilist )
			//	*(p++) = val;
			MemCopy( begin(), ilist.begin(), ilist.size() );
		}


		void Release()
		{
			m_Length	= 0;
			m_AllocSize	= 0;
			SafeDeleteArray( m_pData );
		}


		void Clear()
		{
			if( m_Length>0 )
				memset( m_pData, 0, m_AllocSize );
		}


		void SetValues( uint8* pdata, int len )
		{
			assert( len>0 && pdata );
			MemCopy( m_pData, (T*)pdata, Min( m_Length, len ) );
		}


		template < typename ... Args >
		std::enable_if_t< TypeTraits::all_convertible<T, Args...>::value, void >
		SetValues( const Args& ... args )
		{
			auto values = { (T)args... };
			MemCopy( m_pData, values.begin(), Min( (size_t)m_Length, values.size() ) );

			//T values[]{ (T)args... };//
			//MemCopy( m_pData, values, Min( (size_t)m_Length, sizeof...(Args) ) );

			//SetValues( { (T)args... } );


			//int64 count = (int64)Min( sizeof...(Args), (size_t)m_Length ) - 1;
			//auto src = std::begin( { (T)args... } );
			//auto dst = begin();
			//while( count-->=0 )
			//	*dst++ = *src++;
		}


		template < typename Type >
		std::enable_if_t< std::is_same_v<Type, T>, void >
		SetValues( std::initializer_list<Type> ilist )
		{
			MemCopy( m_pData, ilist.begin(), Min( (size_t)m_Length, ilist.size() ) );

			//int64 count = (int64)Min( ilist.size(), (size_t)m_Length ) - 1;
			//auto src = ilist.begin();
			//auto dst = begin();
			//while( count-->=0 )
			//	*dst++ = (T)*src++;
		}


		inline bool Resize( int newlen )
		{
			assert( newlen > 0 );

			T *newdata	= new T[ newlen ]();
			int newallocsize = c_ElementSize * newlen;

			MemCopy( newdata, m_pData, Min(m_Length, newlen) );

			SafeDeleteArray( m_pData );
			m_pData		= newdata;
			m_Length	= newlen;
			m_AllocSize	= newallocsize;
			

			return true;
		}


		inline bool Extend( int numelms )
		{
			return Resize( m_Length + numelms );
		}


		inline bool Shrink( int numelms )
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


		int ElementSize() const
		{
			return c_ElementSize;
		}


		int Length() const
		{
			return m_Length;
		}


		int AllocatedSize() const
		{
			return m_AllocSize;
		}


		bool Empty() const
		{
			return (!m_pData) | (m_Length==0);
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

		const int c_ElementSize = sizeof(T);

		int	m_Length;
		int m_AllocSize;
		T*	m_pData;


	};




	//##############################################################################################################//
	//																												//
	//												Helper funcitions												//
	//																												//
	//##############################################################################################################//


	template < typename T >
	inline int64 Find( const Memory<T>& arr, const T& item )
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
	inline bool Exists( int numelms, const T data[], const T& item )
	{
		for( int i=0; i<numelms; ++i )
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
