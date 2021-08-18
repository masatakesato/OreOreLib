#include	<crtdbg.h>

#include	<bitset>
#include	<iostream>
using namespace std;

#include	<oreore/common/TString.h>
#include	<oreore/common/BitOperations.h>


#include	<oreore/mathlib/MathLib.h>





int main()
{

	{
		tcout << "//===================== MSB/LSB test using uint8[9] =========================//\n";

		uint8 bitstring8[] = {
			0x00, 0x00, 0x00, 0x00, 
			0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00 };

		// Initialize array
		for( int i=1; i<8; ++i )
			SetBit(i*6, bitstring8 );

	//	SetBit( 16, bitstring8 );
	//	SetBit( 46, bitstring8 );
		SetBit( 78, bitstring8 );
	//	SetBit( 118, bitstring8 );

		DisplayBitArray( bitstring8, ArraySize( bitstring8 ) * BitSize::uInt8 );

	//	tcout << bitset<64>( (uint64&)bitstring8[8] ) << tendl;
	//	tcout << bitset<64>( (uint64&)bitstring8[0] ) << tendl;

		tcout << "Most Significant Bit: ";
		tcout << testMSB( (uint8*)bitstring8, ArraySize(bitstring8) ) << tendl;
		tcout << "Least Significant Bit: ";
		tcout << testLSB( (uint8*)bitstring8, ArraySize(bitstring8) ) << tendl;
	}

	tcout << tendl;

	{
		tcout << "//===================== MSB/LSB test using uint64[2] =========================//\n";

		uint64 bitstring64[] = { 0x0, 0x0 };
		SetBit( 13, (uint8*)bitstring64 );
		SetBit( 42, (uint8*)bitstring64 );
		SetBit( 53, (uint8*)bitstring64 );

		DisplayBitArray( (uint8*)bitstring64, ArraySize( bitstring64 ) * BitSize::uInt64/*ByteSize::uInt64*/ );
		//tcout << bitset<128>( (uint64*)&bitstring64[0] ) << tendl;

		tcout << "Most Significant Bit: ";
		tcout << testMSB( (uint8*)bitstring64, ArraySize(bitstring64) * ByteSize::uInt64 ) << tendl;
		tcout << "Least Significant Bit: ";
		tcout << testLSB( (uint8*)bitstring64, ArraySize(bitstring64) * ByteSize::uInt64 ) << tendl;

	}
	
	return 0;
}