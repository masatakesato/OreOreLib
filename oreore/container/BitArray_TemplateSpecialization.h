#ifndef BIT_ARRAY_H
#define	BIT_ARRAY_H


#include	"../common/BitOperations.h"

#include	"../mathlib/MathLib.h"
#include	"../mathlib/MersenneTwister.h"



//##################################################################//
//																	//
//						Class declaration							//
//																	//
//##################################################################//

class IBitArray;


// BitArray class implementation
template< int N > class BitArray_Impl;

// Dynamic bitarray
using DynamicBitArray = BitArray_Impl< -1 >;

// Static bitarray
template< int N >
using StaticBitArray = BitArray_Impl<N>;


// 同じ名前でテンプレート引数の省略



//##################################################################//
//																	//
//						IBitArray Implementation					//
//																	//
//##################################################################//

class IBitArray
{
public:

	// Constructor
	IBitArray( int bitlength=0 )
		: m_BitLength( bitlength )
		, m_ByteSize( bitlength>0 ? DivUp( bitlength, (int)BitSize::uInt8 ) : 0 )
	{

	}


	// Destructor
	virtual ~IBitArray()
	{
	
	}


	void Set( int b )
	{
		Ptr()[ byteOffset( b )] |= 1 << bitOffset( b );
	}


	void Unset( int b )
	{
		Ptr()[ byteOffset( b ) ] &= ~( 1 << bitOffset( b ) );
	}


	void Flip( int b )
	{
		Ptr()[ byteOffset( b ) ] ^= 1 << bitOffset( b );
	}


	void SetAll()
	{
		const uint8 nonzero	= ~0;
		memset( Ptr(), nonzero, ByteSize() );
	}


	void UnsetAll()
	{
		const uint8 zero	= 0;
		memset( Ptr(), zero, ByteSize() );
	}


	void Randomize( int start, int length )// randomize length bits from start potition.
	{
		int numFlips = int( OreOreLib::genrand_real1() * length );
		for( int b=0; b<numFlips; ++b )
		{
			int idx	= int( OreOreLib::genrand_real2() * length ) + start;
			Flip( idx );
		}
	}


	void RandomizeAll()
	{
		UnsetAll();

		int numFlips = int( OreOreLib::genrand_real1() * m_BitLength );
		for( int b=0; b<numFlips; ++b )
		{
			int idx	= int( OreOreLib::genrand_real2() * m_BitLength );
			Flip( idx );
		}
	}


	int BitLength() const
	{
		return m_BitLength;
	}


	int ByteSize() const
	{
		return m_ByteSize;
	}


	int GetBit( int b ) const
	{
		return int( ( Ptr()[ byteOffset( b ) ] & 1 << bitOffset( b ) ) > 0 );
	}


	void SetBit( int b, int val )
	{
		auto byteoffset = byteOffset( b );
		auto bitoffset	= bitOffset( b );
		Ptr()[ byteoffset ]	= Ptr()[ byteoffset ] & ~( 1 << bitoffset ) | val << bitoffset;
	}


	int GetMSB() const
	{
		return Max( testMSB( Ptr(), ByteSize() ), -1 );
	}


	int GetLSB() const
	{
		return Max( testLSB( Ptr(), ByteSize() ), -1 );
	}


	void CopyFrom( const IBitArray* pSrc )
	{
		auto copylength = Min( ByteSize(), pSrc->ByteSize() );
		memcpy( Ptr(), pSrc->Ptr(), ByteSize() );
	}


	void CopyFrom( int dst_start, const IBitArray* pSrc, int src_start, int length )
	{
		for( int dst=dst_start, src=src_start; dst<Min( dst_start+length, m_BitLength ); ++dst, ++src )
			SetBit( dst, pSrc->GetBit( src ) );
	}


	void Display()
	{
		DisplayBitArray( Ptr(), m_BitLength );
	}



protected:

	int	m_BitLength = 0;
	int m_ByteSize = 0;

	virtual uint8* Ptr() const = 0;


};





//##################################################################//
//																	//
//						 Dynamic BitArray_Impl						//
//																	//
//##################################################################//

template<>
class BitArray_Impl< -1 > : public IBitArray
{
public:

	// Default constructor
	BitArray_Impl()
		: IBitArray()
		, m_pWords( nullptr )
	{

	}


	// Constructor
	BitArray_Impl( int bitlength )
		: IBitArray( bitlength )
	{
		Init( bitlength );
	}


	// Constructor
	template< int N_ >
	BitArray_Impl( const BitArray_Impl<N_>& obj )
	{
		m_BitLength	= obj.BitLength();
		m_ByteSize	= obj.ByteSize();
		if( obj.ConstPtr() )
		{
			m_pWords	= new uint8[ m_ByteSize ];
			memcpy( m_pWords, obj.ConstPtr(), m_ByteSize );
		}
		else
		{
			m_pWords	= nullptr;
		}
	}



	// Copy constructor
	BitArray_Impl( const BitArray_Impl& obj )
	{
		m_BitLength	= obj.m_BitLength;
		m_ByteSize	= obj.m_ByteSize;
		if( obj.ConstPtr() )
		{
			m_pWords	= new uint8[ m_ByteSize ];
			memcpy( m_pWords, obj.m_pWords, m_ByteSize );
		}
		else
		{
			m_pWords	= nullptr;
		}
	}


	// Move constructor
	BitArray_Impl( BitArray_Impl&& obj )
	{
		m_BitLength	= obj.m_BitLength;
		m_ByteSize	= obj.m_ByteSize;
		m_pWords	= obj.m_pWords;

		obj.m_pWords = nullptr;
	}


	// Destructor
	~BitArray_Impl()
	{
		Release();
	}


	// Copy assignment operator
	BitArray_Impl& operator=( const BitArray_Impl& obj )
	{
		if( this != &obj )
		{
			Release();

			m_BitLength	= obj.m_BitLength;
			m_ByteSize	= obj.m_ByteSize;
			if( obj.m_pWords )
			{
				m_pWords	= new uint8[ m_ByteSize ];
				memcpy( m_pWords, obj.m_pWords, m_ByteSize );
			}
			else
			{
				m_pWords	= nullptr;
			}
		}

		return *this;
	}


	// Move assignment operator
	BitArray_Impl& operator=( BitArray_Impl&& obj )
	{
		if( this != &obj )
		{
			m_BitLength	= obj.m_BitLength;
			m_ByteSize	= obj.m_ByteSize;
			m_pWords	= obj.m_pWords;

			obj.m_pWords = nullptr;
		}
		
		return *this;
	}


	void Init( int bitlength )
	{
		Release();

		m_BitLength	= bitlength;
		m_ByteSize	= DivUp( m_BitLength, (int)BitSize::uInt8 );

		m_pWords	= new uint8[ m_ByteSize ];
		UnsetAll();
	}


	void Release()
	{
		m_BitLength	= 0;
		m_ByteSize	= 0;
		SafeDeleteArray( m_pWords );
	}


	const uint8* ConstPtr() const
	{
		return m_pWords;
	}



protected:

	uint8*	m_pWords;

	virtual uint8* Ptr() const
	{
		return m_pWords;
	}

};




//##################################################################//
//																	//
//						 Static BitArray_Impl						//
//																	//
//##################################################################//
 
template < int N >
class BitArray_Impl : public IBitArray
{
public:

	// Default constructor
	BitArray_Impl()
		: IBitArray(N)
	{
		memset( m_Words, 0, m_ByteSize );
	}


	// Copy constructor
	template< int N_ >
	BitArray_Impl( const BitArray_Impl<N_>& obj )
	{
		m_BitLength = N;
		memset( m_Words, 0, m_ByteSize );

		auto copylength = Min( m_ByteSize, obj.ByteSize() );
		memcpy( m_Words, obj.ConstPtr(), copylength );
	}


	// Destructor
	virtual ~BitArray_Impl()
	{
		
	}


	// Copy assignment operator
	BitArray_Impl& operator=( const BitArray_Impl& obj )
	{
		if( this != &obj )
		{
			auto copylength = Min( m_ByteSize, obj.m_ByteSize );
			memcpy( m_Words, obj.m_Words, copylength );
		}

		return *this;
	}


	const uint8* ConstPtr() const
	{
		return (uint8*)&m_Words;
	}


	void Display()
	{
		DisplayBitArray( (uint8*)m_Words, N );
	}



private:

	uint8	m_Words[ DivUp( N, (int)BitSize::uInt8 ) ];

	virtual uint8* Ptr() const
	{
		return (uint8*)&m_Words;
	}

};




#endif // !BIT_ARRAY_H
