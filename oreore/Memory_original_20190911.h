#ifndef MEMORY_H
#define	MEMORY_H

#include	<algorithm>

#include	"./common/Utility.h"



namespace OreOreLib
{

	template< typename T >
	struct Memory
	{
	public:

		// default constructor
		Memory()
			: data_size( ( int )sizeof( T ) )
			, length( 0 )
			, pData( nullptr )
			, pDeleteFunc( &Memory::release_memory )//, pDeleteFunc( &Memory::release_reference )//
			, pCopyFunc( &Memory::copy_memory )		//, pCopyFunc( &Memory::copy_reference )//
			, pMoveFunc( &Memory::move_memory )		//, pMoveFunc( &Memory::move_reference )//
		{
			//		std::cout << "Memory default constructor...\n";
		}

		// constructor
		Memory( int len )
			: data_size( ( int )sizeof( T ) )
			, length( len )
			, pData( new T[len] )
			, pDeleteFunc( &Memory::release_memory )
			, pCopyFunc( &Memory::copy_memory )
			, pMoveFunc( &Memory::move_memory )
		{
			assert( len > 0 );
			//memset( pData, 0, data_size * length );
			//		std::cout << "Memory constructor(dynamic allocation)...\n";
		}

		// constructor with external buffer
		Memory( int len, T* pdata )
			: data_size( ( int )sizeof( T ) )
			, length( len )
			, pData( new(pdata) T[len] )
			, pDeleteFunc( &Memory::release_reference )
			, pCopyFunc( &Memory::copy_reference )
			, pMoveFunc( &Memory::move_reference )
		{
			assert( len > 0 && pdata != nullptr );
			//		std::cout << "Memory constructor(reference)...\n";
		}


		// destructor
		~Memory()
		{
			//		std::cout << "Memory destructor...\n";
			data_size = 0;
			length  = 0;
			( this->*pDeleteFunc )( );
		}

		// Copy constructor
		Memory( const Memory& obj )
		{
			//		std::cout << "Memory copy constructor...\n";
			( const_cast<Memory&>( obj ).*obj.pCopyFunc )( *this );
		}

		// Move constructor
		Memory( Memory&& obj )
		{
			//		std::cout << "Memory move constructor...\n";
			( const_cast<Memory&>( obj ).*obj.pMoveFunc )( *this );
		}


		// Copy Assignment opertor =
		inline Memory& operator=( const Memory& obj )
		{
			if( this != &obj )
			{
				//			std::cout << "Memory copy assignment operator...\n";
				( this->*pDeleteFunc )( );// left hand side already has pData. need to call pDeleteFunc.
										  //			tcout << "delete pData\n";
				( const_cast<Memory&>( obj ).*obj.pCopyFunc )( *this );
			}
			return *this;
		}


		// Move assignment opertor =
		inline Memory& operator=( Memory&& obj )
		{
			if( this != &obj )
			{
				//			std::cout << "Memory move assignment operator...\n";
				( this->*pDeleteFunc )( );// left hand side already has pData. need to call pDeleteFunc.
										  //			tcout << "delete pData\n";
				( const_cast<Memory&>( obj ).*obj.pMoveFunc )( *this );
			}
			return *this;
		}


		// Subscription operator
		inline const T& operator[]( std::size_t n ) const&
		{
			return pData[n];
		}


		// Subscription operator
		inline T& operator[]( std::size_t n ) &
		{
			return pData[n];
		}


		// Subscription operator
		inline T operator[]( std::size_t n ) const&&
		{
			return pData[n];
		}

		void Clear()
		{
			memset( pData, 0, data_size * length );
		}

		void Release()
		{
			( this->*pDeleteFunc )( );

			length		= 0;
			pDeleteFunc	= &Memory::release_memory;
			pCopyFunc	= &Memory::copy_memory;
			pMoveFunc	= &Memory::move_memory;
		}

		void Init( int len )
		{
			assert( len>0 );

			( this->*pDeleteFunc )( );

			data_size	= ( int )sizeof( T );
			length		= len;
			pData		= new T[len];
			//memset( pData, 0, data_size * length );

			pDeleteFunc	= &Memory::release_memory;
			pCopyFunc	= &Memory::copy_memory;
			pMoveFunc	= &Memory::move_memory;
		}

		void Init( int len, T* pdata )
		{
			assert( len>0 );
			assert( pdata );

			( this->*pDeleteFunc )( );

			data_size	= ( int )sizeof( T );
			length		= len;
			pData		= new( pdata ) T[len];

			pDeleteFunc	= &Memory::release_reference;
			pCopyFunc	= &Memory::copy_reference;
			pMoveFunc	= &Memory::move_reference;
		}

		inline bool Resize( int newlen )
		{
			assert( newlen > 0 );

			if( pCopyFunc == &Memory::copy_reference ) return false;// cannot resize reference array.

			T *newdata	= new T[newlen];
			int copy_size = data_size * (std::min)( length, newlen );
			memcpy_s( newdata, copy_size, pData, copy_size );

			SafeDeleteArray( pData );
			pData	= newdata;
			length	= newlen;

			return true;
		}

		inline bool Extend( int numelms )
		{
			return Resize( length + numelms );
		}

		inline bool Shrink( int numelms )
		{
			if( length > numelms )	return Resize( length - numelms );
			return false;
		}


		inline void CopyFrom( const Memory& src )
		{
			int copy_size = (std::min)( data_size * length, src.data_size * src.length );
			memcpy_s( pData, copy_size, src.pData, copy_size );
		}

		inline void CopyTo( Memory& dst ) const
		{
			int copy_size = (std::min)( data_size * length, dst.data_size * dst.length );
			memcpy_s( dst.pData, copy_size, pData, copy_size );
		}

		int Length() const
		{
			return length;
		}

		// https://stackoverflow.com/questions/31581880/overloading-cbegin-cend
		// begin / end overload for "range-based for loop"
		inline T* begin()
		{
			return pData;
		}

		inline const T* begin() const
		{
			return pData;
		}

		inline T* end()
		{
			return begin() + length;
		}

		inline const T* end() const
		{
			return begin() + length;
		}


	protected:

		int data_size;
		int length;
		T*	pData;

		void ( Memory::*pDeleteFunc )( );
		void ( Memory::*pCopyFunc )( Memory& );
		void ( Memory::*pMoveFunc )( Memory& );

		inline void release_reference()
		{
			pData = nullptr;
		}

		inline void release_memory()
		{
			SafeDeleteArray( pData );
		}

		inline void copy_reference( Memory& dest )
		{
			dest.data_size	= data_size;
			dest.length		= length;
			dest.pData		= pData;

			dest.pDeleteFunc	= &Memory::release_reference;
			dest.pCopyFunc		= &Memory::copy_reference;
			dest.pMoveFunc		= &Memory::move_reference;
		}

		inline void copy_memory( Memory& dest )
		{
			//		tcout << "copy_memory\n";
			dest.data_size	= data_size;
			dest.length		= length;
			if( pData )
			{
				dest.pData = new T[dest.length];
				memcpy_s( dest.pData, dest.data_size*dest.length, pData, data_size*length );
			}
			else// avoid zero-element dynamic allocation.
			{
				dest.pData = nullptr;
			}

			dest.pDeleteFunc	= &Memory::release_memory;
			dest.pCopyFunc		= &Memory::copy_memory;
			dest.pMoveFunc		= &Memory::move_memory;
		}

		inline void move_reference( Memory& dest )
		{
			// copy data to dest
			dest.data_size	= data_size;
			dest.length		= length;
			dest.pData		= pData;

			// assign dest's delete/copy/move functions
			dest.pDeleteFunc	= &Memory::release_reference;
			dest.pCopyFunc		= &Memory::copy_reference;
			dest.pMoveFunc		= &Memory::move_reference;

			// clear reference from this
			pData	= nullptr;
		}

		inline void move_memory( Memory& dest )
		{
			//		tcout << "move_memory\n";
			// copy data to dest
			dest.data_size	= data_size;
			dest.length		= length;
			dest.pData		= pData;

			// assign dest's delete/copy/move functions
			dest.pDeleteFunc	= &Memory::release_memory;
			dest.pCopyFunc		= &Memory::copy_memory;
			dest.pMoveFunc		= &Memory::move_memory;

			// clear reference from this
			pData	= nullptr;
		}

	};


}// end of namespace


#endif // !MEMORY_H
