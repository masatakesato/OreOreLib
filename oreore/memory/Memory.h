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
	MemCopy( Iter* pDst, Iter* pSrc, size_t size )
	{
		return (Iter*)memcpy( pDst, pSrc, sizeof Iter * size );
	}

	// Non-Trivial Memcpy
	template < class Iter >
	std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	MemCopy( Iter* pDst, Iter* pSrc, size_t size )
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



	// Trivial MemMove
	template < class Iter >
	std::enable_if_t< std::is_trivially_copyable_v<Iter>, Iter* >
	MemMove( Iter* pDst, Iter* pSrc, size_t size )
	{
		return (Iter*)memmove( pDst, pSrc, sizeof Iter * size );
	}

	// Non-Trivial MemMove
	template < class Iter >
	std::enable_if_t< !std::is_trivially_copyable_v<Iter>, Iter* >
	MemMove( Iter* pDst, Iter* pSrc, size_t size )
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


	#endif//__cplusplus







	//##############################################################################################################//
	//																												//
	//										Memory class implementation												//
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


		// Subscription operator for read only.( called if Memory is const )
		inline const T& operator[]( std::size_t n ) const&
		{
			return m_pData[n];
		}


		// Subscription operator for read-write.( called if Memory is non-const )
		inline T& operator[]( std::size_t n ) &
		{
			return m_pData[n];
		}


		// Subscription operator. ( called by following cases: "T a = Memory<T>(10)[n]", "auto&& a = Memory<T>(20)[n]" )
		inline T operator[]( std::size_t n ) const&&
		{
			return std::move(m_pData[n]);// return object
		}


		operator bool() const
		{
			return m_pData != nullptr;
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







}// end of namespace


#endif // !MEMORY_H
