#ifndef BIT_OPERATIONS_H
#define	BIT_OPERATIONS_H


#include	"Utility.h"
#include	"TString.h"


//###################################################################################################//
//										Helper functions										     //
//###################################################################################################//

inline static sizeType byteOffset( sizeType b )
{
	return (sizeType)b / BitSize::uInt8;
}



inline static sizeType bitOffset( sizeType b )
{
	return (sizeType)b % BitSize::uInt8;
}




//###################################################################################################//
//											Bit Operations										     //
//###################################################################################################//

inline static void SetBit( sizeType b, uint8 bytes[] )
{
	bytes[ byteOffset( b ) ] |= 1 << bitOffset( b );
}



inline static void SetBit( sizeType b, bool val, uint8 bytes[] )
{
	sizeType boffset	= bitOffset( b );
	sizeType byteoffset = byteOffset( b );
	bytes[ byteoffset ] = bytes[ byteoffset ] & ~( 1 << boffset ) | static_cast<sizeType>(val) << boffset;
}



inline static void UnsetBit( sizeType b, uint8 bytes[] )
{
	bytes[byteOffset( b )] &= ~( 1 << bitOffset( b ) );
}



inline static void FlipBit( sizeType b, uint8 bytes[] )
{
	bytes[byteOffset( b )] ^= 1 << bitOffset( b );
}



inline static bool GetBit( sizeType b, uint8 bytes[] )
{
	return ( bytes[byteOffset( b )] & 1 << bitOffset( b ) ) > 0;
}



static void DisplayBitArray( uint8 bytes[], sizeType bitlength )
{
	tcout << "Bit Array[" << bitlength << "]:\n";
	for( sizeType i=bitlength; i-->0; )//for( sizeType i=bitlength-1; i>=0; --i )
	{
		tcout << GetBit(i, bytes);
		if( i % BitSize::uInt8 == 0 ) tcout << " ";
	}
	tcout << tendl;
}




//###################################################################################################//
//											Bit count 											     //
//###################################################################################################//


inline static int BitCount( uint32 i )
{
     i -= i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}



inline static int BitCount( uint64 x )
{
    x -= (x >> 1) & 0x5555555555555555;             //put count of each 2 bits into those 2 bits
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333); //put count of each 4 bits into those 4 bits 
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;        //put count of each 8 bits into those 8 bits 
    return (x * 0x0101010101010101) >> 56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}





//##################################################################################################//
//	Most Significant Bit finding operation by matteconte. . ( https://github.com/mattconte/tlsf)	//
//##################################################################################################//

// Most Significant Bit index.( 8 bit version. returns [1, 8] )
inline static uint8 _MSB( uint8 val )
{
	uint8 bit = 8;

	if( !val ) --bit;
	if( !( val & 0xf0 ) ) { val <<= 4; bit -= 4; }
	if( !( val & 0xc0 ) ) { val <<= 2; bit -= 2; }
	if( !( val & 0x80 ) ) { val <<= 1; --bit; }

	return bit;
}



// Most Significant Bit index.( 16 bit version. returns [1, 16] )
inline static uint16 _MSB( uint16 val )
{
	uint16 bit = 16;

	if( !val ) --bit;
	if( !( val & 0xff00 ) ) { val <<= 8; bit -= 8; }
	if( !( val & 0xf000 ) ) { val <<= 4; bit -= 4; }
	if( !( val & 0xc000 ) ) { val <<= 2; bit -= 2; }
	if( !( val & 0x8000 ) ) { val <<= 1; --bit; }

	return bit;
}



// Most Significant Bit index.( 32 bit version. returns [1, 32] )
inline static uint32 _MSB( uint32 val )
{
	uint32 bit = 32;

	if( !val ) --bit;
	if( !( val & 0xffff0000 ) ) { val <<= 16; bit -= 16; }
	if( !( val & 0xff000000 ) ) { val <<= 8; bit -= 8; }
	if( !( val & 0xf0000000 ) ) { val <<= 4; bit -= 4; }
	if( !( val & 0xc0000000 ) ) { val <<= 2; bit -= 2; }
	if( !( val & 0x80000000 ) ) { val <<= 1; --bit; }

	return bit;
}



// Most Significant Bit index.( 64 bit version. returns [1, 64] )
inline static uint64 _MSB( uint64 val )
{
	uint64 bit = 64;

	if( !val ) --bit;
	if( !( val & 0xffffffff00000000 ) ) { val <<= 32; bit -= 32; }
	if( !( val & 0xffff000000000000 ) ) { val <<= 16; bit -= 16; }
	if( !( val & 0xff00000000000000 ) ) { val <<= 8; bit -= 8; }
	if( !( val & 0xf000000000000000 ) ) { val <<= 4; bit -= 4; }
	if( !( val & 0xc000000000000000 ) ) { val <<= 2; bit -= 2; }
	if( !( val & 0x8000000000000000 ) ) { val <<= 1; --bit; }

	return bit;
}



// Most Significant Bit index.( 8 bit version. returns [0, 7] )
inline static uint32 GetMSB( uint8 val )
{
	return (uint32)_MSB( val ) - 1;
}



// Most Significant Bit index.( 16 bit version. returns [0, 15] )
inline static uint32 GetMSB( uint16 val )
{
	return (uint32)_MSB( val ) - 1;
}


// Most Significant Bit index.( 32 bit version. returns [0, 31] )
inline static uint32 GetMSB( uint32 val )
{
	return (uint32)_MSB( val ) - 1;
}


// Most Significant Bit index.( 64 bit version. returns [0, 63] )
inline static uint32 GetMSB( uint64 val )
{
	return (uint32)_MSB( val ) - 1;
}


template < typename IntType = int32 >
inline typename std::enable_if_t< std::is_signed_v<IntType> && std::is_integral_v<IntType>, IntType >
testMSB( uint8 bytes[], sizeType size )
{
	IntType byteoffset = static_cast<IntType>( size >= ByteSize::uInt64 ? size - ByteSize::uInt64 : size );
	IntType msb = 0;

	//========== 64 bit iterative check ===========//
	for( sizeType i=0; i<size/ByteSize::uInt64; ++i, byteoffset-=(IntType)ByteSize::uInt64 )
	{
		//tcout << "64 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt64 - 1 << "]" << tendl;
		msb = static_cast<IntType>( _MSB( (uint64&)bytes[ byteoffset ] ) );
		if( msb > 0 )
			return msb + (IntType)BitSize::uInt8 * byteoffset - 1;
	}

	//=============== 32 bit check ================//
	if( byteoffset >= (IntType)ByteSize::uInt32 )
	{
		byteoffset -= (IntType)ByteSize::uInt32;
		//tcout << "32 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt32 - 1 << "]" << tendl;
		msb = static_cast<IntType>( _MSB( (uint32&)bytes[byteoffset] ) );
		if( msb > 0 || byteoffset <= 0 )// return if msb found or all bits are checked
			return msb + (IntType)BitSize::uInt8 * byteoffset - 1;
	}

	//=============== 16 bit check ================//
	if( byteoffset >= (IntType)ByteSize::uInt16 )
	{
		byteoffset -= (IntType)ByteSize::uInt16;
		//tcout << "16 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt16 - 1 << "]" << tendl;
		msb = static_cast<IntType>( _MSB( (uint16&)bytes[byteoffset] ) );
		if( msb > 0 || byteoffset <= 0 )// return if msb found or all bits are checked
			return msb + (IntType)BitSize::uInt8 * byteoffset - 1;
	}

	//=============== 8 bit check ================//
	if( byteoffset >= (IntType)ByteSize::uInt8 )
	{
		byteoffset -= (IntType)ByteSize::uInt8;
		//tcout << "8 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt8 - 1 << "]" << tendl;
		msb = static_cast<IntType>( _MSB( (uint8&)bytes[byteoffset] ) );
		if( msb > 0 || byteoffset <= 0 )// return if msb found or all bits are checked
			return msb + (IntType)BitSize::uInt8 * byteoffset - 1;
	}

	return -1;
}




//##################################################################################################//
//	Least Significant Bit finding operation by matteconte. . ( https://github.com/mattconte/tlsf)	//
//##################################################################################################//

// Least Significant Bit index.( 8 bit version. returns [1, 8] )
inline static uint8 _LSB( uint8 val )
{
	uint8 bit = 0;

	if( val )
	{
		val &= ~val + 1;
		if( val & 0xf0 ) bit += 4;
		if( val & 0xcc ) bit += 2;
		if( val & 0xaa ) ++bit;
		++bit;
	}

	return bit;
}



// Least Significant Bit index.( 16 bit version. returns [1, 16] )
inline static uint16 _LSB( uint16 val )
{
	uint16 bit = 0;

	if( val )
	{
		val &= ~val + 1;
		if( val & 0xff00 ) bit += 8;
		if( val & 0xf0f0 ) bit += 4;
		if( val & 0xcccc ) bit += 2;
		if( val & 0xaaaa ) ++bit;
		++bit;
	}

	return bit;
}



// Least Significant Bit index.( 32 bit version. returns [1, 32] )
inline static uint32 _LSB( uint32 val )
{
	uint32 bit = 0;

	if( val )
	{
		val &= ~val + 1;
		if( val & 0xffff0000 ) bit += 16;
		if( val & 0xff00ff00 ) bit += 8;
		if( val & 0xf0f0f0f0 ) bit += 4;
		if( val & 0xcccccccc ) bit += 2;
		if( val & 0xaaaaaaaa ) ++bit;
		++bit;
	}

	return bit;
}



// Least Significant Bit index.( 64 bit version. returns [1, 64] )
inline static uint64 _LSB( uint64 val )
{
	uint64 bit = 0;

	if( val )
	{
		val &= ~val + 1;
		if( val & 0xffffffff00000000 ) bit += 32;
		if( val & 0xffff0000ffff0000 ) bit += 16;
		if( val & 0xff00ff00ff00ff00 ) bit += 8;
		if( val & 0xf0f0f0f0f0f0f0f0 ) bit += 4;
		if( val & 0xcccccccccccccccc ) bit += 2;
		if( val & 0xaaaaaaaaaaaaaaaa ) ++bit;
		++bit;
	}

	return bit;
}



// Least Significant Bit index.( 8 bit version. returns [0, 7] )
inline static uint32 GetLSB( uint8 val )
{
	return (uint32)_LSB( val ) - 1;
}



// Least Significant Bit index.( 16 bit version. returns [0, 15] )
inline static uint32 GetLSB( uint16 val )
{
	return (uint32)_LSB( val ) - 1;
}



// Least Significant Bit index.( 32 bit version. returns [0, 31] )
inline static uint32 GetLSB( uint32 val )
{
	return (uint32)_LSB( val ) - 1;
}



// Least Significant Bit index.( 64 bit version. returns [0, 63] )
inline static uint32 GetLSB( uint64 val )
{
	return (uint32)_LSB( val ) - 1;
}



template < typename IntType = int32 >
inline typename std::enable_if_t< std::is_signed_v<IntType> && std::is_integral_v<IntType>, IntType >
testLSB( uint8 bytes[], sizeType size )
{
	IntType byteoffset = 0;
	IntType lsb;

	//========== 64 bit iterative check ===========//
	for( sizeType i=0; i<size/ByteSize::uInt64; ++i, byteoffset+=ByteSize::uInt64 )
	{
		//tcout << "64 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt64 - 1 << "]" << tendl;
		lsb = static_cast<IntType>( _LSB( (uint64&)bytes[ byteoffset ] ) );
		if( lsb > 0 )
			return lsb + BitSize::uInt8 * byteoffset - 1;
	}

	//=============== 32 bit check ================//
	if( size - byteoffset >= ByteSize::uInt32 )
	{
		//tcout << "32 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt32 - 1 << "]" << tendl;
		lsb = static_cast<IntType>( _LSB( (uint32&)bytes[byteoffset] ) );
		if( lsb > 0 || size - byteoffset == ByteSize::uInt32 )
			return lsb + BitSize::uInt8 * byteoffset - 1;

		byteoffset += ByteSize::uInt32;
	}

	//=============== 16 bit check ================//
	if( size - byteoffset >= ByteSize::uInt16 )
	{
		//tcout << "16 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt16 - 1 << "]" << tendl;
		lsb = static_cast<IntType>( _LSB( (uint16&)bytes[byteoffset] ) );
		if( lsb > 0 || size - byteoffset == ByteSize::uInt16 )
			return lsb + BitSize::uInt8 * byteoffset - 1;

		byteoffset += ByteSize::uInt16;
	}

	//=============== 8 bit check ================//
	if( size - byteoffset >= ByteSize::uInt8 )
	{
		//tcout << "8 bit MSB check: [" << BitSize::uInt8 * byteoffset << ", " <<  BitSize::uInt8 * byteoffset + BitSize::uInt8 - 1 << "]" << tendl;
		lsb = static_cast<IntType>( _LSB( (uint8&)bytes[byteoffset] ) );
		if( lsb > 0 )
			return lsb + BitSize::uInt8 * byteoffset - 1;
	}

	return -1;
}





#endif // !BIT_OPERATIONS_H