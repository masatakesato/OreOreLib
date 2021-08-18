#ifndef BIT_ARRAY_H
#define	BIT_ARRAY_H


#include	"../common/BitOperations.h"

#include	"../mathlib/MathLib.h"
#include	"../mathlib/MersenneTwister.h"


//##################################################################//
//																	//
//								BitArray							//
//																	//
//##################################################################//


class BitArray
{
public:

	// Default constructor
	BitArray()
		: m_BitLength( 0 )
		, m_ByteSize( 0 )
		, m_pWords( nullptr )
	{
		//tcout << _T( "BitArray::BitArray()...\n" );
	}


	// Constructor
	BitArray( int bitlength )
		: m_pWords( nullptr )
	{
		//tcout << _T( "BitArray::BitArray( int bitlength )...\n" );
		Init( bitlength );
	}


	// Constructor
	BitArray( int bitlength, const tstring& tstr )
		: m_BitLength( bitlength )
		, m_ByteSize( DivUp( m_BitLength, (int)BitSize::uInt8 ) )
		, m_pWords( /*new uint8[ m_ByteSize ]()*/nullptr )
	{
		TCharToChar( tstr.c_str(), Min(tstr.length(), size_t(m_ByteSize) ), (char*&)m_pWords );
	}


	// Copy constructor
	BitArray( const BitArray& obj )
	{
		//tcout << _T( "BitArray::BitArray( const BitArray& obj )...\n" );

		m_BitLength	= obj.m_BitLength;
		m_ByteSize	= obj.m_ByteSize;
		if( obj.m_pWords )
		{
			m_pWords = new uint8[ m_ByteSize ];
			memcpy( m_pWords, obj.m_pWords, m_ByteSize );
		}
		else
		{
			m_pWords	= nullptr;
		}
	}


	// Move constructor
	BitArray( BitArray&& obj )
	{
		m_BitLength	= obj.m_BitLength;
		m_ByteSize	= obj.m_ByteSize;
		m_pWords	= obj.m_pWords;

		obj.m_pWords = nullptr;
	}


	// Destructor
	~BitArray()
	{
		//tcout << _T( "BitArray::~BitArray()...\n" );
		Release();
	}


	// Copy assignment operator
	BitArray& operator=( const BitArray& obj )
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
	BitArray& operator=( BitArray&& obj )
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


	void Init( int bitlength, const tstring& tstr )
	{
		Release();

		m_BitLength	= bitlength;
		m_ByteSize	= DivUp( m_BitLength, (int)BitSize::uInt8 );

		TCharToChar( tstr.c_str(), Min(tstr.length(), size_t(m_ByteSize) ), (char*&)m_pWords );
	}


	void Release()
	{
		m_BitLength	= 0;
		m_ByteSize	= 0;
		SafeDeleteArray( m_pWords );
	}


	void Set( int b )
	{
		m_pWords[ byteOffset( b )] |= 1 << bitOffset( b );
	}


	void Unset( int b )
	{
		m_pWords[ byteOffset( b ) ] &= ~( 1 << bitOffset( b ) );
	}


	void Flip( int b )
	{
		m_pWords[ byteOffset( b ) ] ^= 1 << bitOffset( b );
	}


	void SetAll()
	{
		const uint8 nonzero	= ~0;
		memset( m_pWords, nonzero, m_ByteSize );
	}


	void UnsetAll()
	{
		const uint8 zero	= 0;
		memset( m_pWords, zero, m_ByteSize );
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
		return int( ( m_pWords[ byteOffset( b ) ] & 1 << bitOffset( b ) ) > 0 );
	}


	void SetBit( int b, int val )
	{
		auto byteoffset = byteOffset( b );
		auto bitoffset	= bitOffset( b );
		m_pWords[ byteoffset ]	= m_pWords[ byteoffset ] & ~( 1 << bitoffset ) | val << bitoffset;
	}


	int GetMSB() const
	{
		return Max( testMSB( m_pWords, m_ByteSize ), -1 );
	}


	int GetLSB() const
	{
		return Max( testLSB( m_pWords, m_ByteSize ), -1 );
	}


	void CopyFrom( const BitArray* pSrc )
	{
		auto copylength = Min( m_ByteSize, pSrc->m_ByteSize );
		memcpy( m_pWords, pSrc->m_pWords, m_ByteSize );
	}


	void CopyFrom( int dst_start, const BitArray* pSrc, int src_start, int length )
	{
		for( int dst=dst_start, src=src_start; dst<Min( dst_start+length, m_BitLength ); ++dst, ++src )
			SetBit( dst, pSrc->GetBit( src ) );
	}


	tstring ToTString() const
	{
		return CharToTString( (char*)m_pWords, m_ByteSize );
	}



//uint8* Ptr() const { return m_pWords; }jyhdjhgfjhd
//
//
//void SetPtr( uint8* src ) gragfagsdg
//{
//	memcpy( m_pWords, src, m_ByteSize );
//}


	virtual void Display()
	{
		tcout << _T( "//======= ") << typeid(*this).name() << _T( " =======//\n");
		DisplayBitArray( (uint8*)m_pWords, m_BitLength );
		tcout << tendl;
	}



protected:

	int		m_BitLength = 0;
	int		m_ByteSize = 0;
	uint8*	m_pWords;


	template< int N >
	friend class StaticBitArray;

};




//##################################################################//
//																	//
//							 StaticBitArray							//
//																	//
//##################################################################//
 
template < int N >
class StaticBitArray : public BitArray
{
public:

	// Default constructor
	StaticBitArray()
	{
		//tcout << _T( "StaticBitArray::StaticBitArray()...\n" );

		m_BitLength	= N;
		m_ByteSize	= DivUp( m_BitLength, (int)BitSize::uInt8 );
		m_pWords	= &m_Words[0];

		memset( m_Words, 0, m_ByteSize );
	}


	// Constructor
	StaticBitArray( const BitArray& obj )
	{
		//tcout << _T( "StaticBitArray::StaticBitArray( const BitArray& obj )...\n" );

		m_BitLength = N;
		m_ByteSize	= DivUp( m_BitLength, (int)BitSize::uInt8 );
		m_pWords	= &m_Words[0];

		memset( m_Words, 0, m_ByteSize );

		auto copylength = Min( m_ByteSize, obj.m_ByteSize );
		memcpy( m_Words, obj.m_pWords, copylength );
	}


	// Copy constructor
	StaticBitArray( const StaticBitArray& obj )
	{
		//tcout << _T( "StaticBitArray( const StaticBitArray& obj )...\n" );

		m_BitLength = N;
		m_ByteSize	= DivUp( m_BitLength, (int)BitSize::uInt8 );
		m_pWords	= &m_Words[0];

		memset( m_Words, 0, m_ByteSize );

		auto copylength = Min( m_ByteSize, obj.m_ByteSize );
		memcpy( m_Words, obj.m_pWords, copylength );
	}


	// Destructor
	~StaticBitArray()
	{
		//tcout << _T( "StaticBitArray::~StaticBitArray()...\n" );
		m_pWords = nullptr;
	}


	// Copy assignment operator
	StaticBitArray& operator=( const StaticBitArray& obj )
	{
		if( this != &obj )
		{
			auto copylength = Min( m_ByteSize, obj.m_ByteSize );
			memcpy( m_Words, obj.m_Words, copylength );
		}

		return *this;
	}


	void Display()
	{
		tcout << _T( "//======= ") << typeid(*this).name() << _T( " =======//\n");
		DisplayBitArray( (uint8*)m_Words, N );
		tcout << tendl;
	}

	void Init( int bitlength ) = delete;
	void Release() = delete;


private:

	uint8	m_Words[ DivUp( N, (int)BitSize::uInt8 ) ];


	//using BitArray::Init;
	//using BitArray::Release;

};




#endif // !BIT_ARRAY_H
