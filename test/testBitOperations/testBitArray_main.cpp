#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/container/BitArray.h>




int main()
{

	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	tcout << "//=============== BinaryString test ==================//" << tendl;

	BitArray	bitArray1;
	tcout << "Create bitstring (length=65)..." << tendl;
	bitArray1.Init( 65 );
	bitArray1.Display();

	// set all 1
	tcout << " Set all bits to 1..." << tendl;
	bitArray1.SetAll();
	bitArray1.Display();

	// set all 0
	tcout << " Set all bits to 0..." << tendl;
	bitArray1.UnsetAll();
	bitArray1.Display();

	tcout << "LSB: " << bitArray1.GetLSB() << tendl;
	tcout << "MSB: " << bitArray1.GetMSB() << tendl;


	// flip all
	tcout << " Flip all bits..." << tendl;
	for( int i=0; i<bitArray1.BitLength(); ++i )	bitArray1.Flip( i );
	bitArray1.Display();

	tcout << "LSB: " << bitArray1.GetLSB() << tendl;
	tcout << "MSB: " << bitArray1.GetMSB() << tendl;


	// randomize
	tcout << " Randomize[32, 63]..." << tendl;
	bitArray1.UnsetAll();
	bitArray1.Randomize( 32, 32 );
	bitArray1.Display();

	// randomize all
	tcout << " Randomize all bits..." << tendl;
	bitArray1.UnsetAll();
	bitArray1.RandomizeAll();
	bitArray1.Display();

	for( int i=0; i<bitArray1.BitLength(); ++i )
		bitArray1.SetBit(i, i%2);


	tcout << " Copy to another bitstring..." << tendl;
	BitArray	bitArray2( bitArray1 );

	bitArray2.Display();

	tcout << " Copy partial bitstring..." << tendl;
	bitArray2.UnsetAll();
	bitArray2.CopyFrom( 0, &bitArray1, 32, 32 );
	bitArray2.Display();

	int bitVal = bitArray2.GetBit( 31 );


	tcout << "LSB: " << bitArray2.GetLSB() << tendl;
	tcout << "MSB: " << bitArray2.GetMSB() << tendl;




	StaticBitArray<47> bitArray3;

	bitArray3.Set(13);
	bitArray3.Set(15);
	bitArray3.Set(46);

	bitArray3.Display();
	tcout << "LSB: " << bitArray3.GetLSB() << tendl;
	tcout << "MSB: " << bitArray3.GetMSB() << tendl;


	//StaticBitArray<47> bitArray4( bitArray3 );// OK
	//StaticBitArray<88> bitArray4( bitArray1 );// OK
	StaticBitArray<88> bitArray4 = bitArray1;// OK

	bitArray4.CopyFrom( 16, &bitArray1, 5, 16 );//bitArray4.CopyFrom( &bitArray1 );// OK

	bitArray4.Display();

	//StaticBitArray<88> bitArray5 = bitArray4;// OK
	StaticBitArray<88> bitArray5( bitArray4 );
	bitArray5.Display();

	
	tcout << bitArray3 << tendl;
	tcout << bitArray5 << tendl;
	tcout << (bitArray3 & bitArray5) << tendl;
	tcout << (bitArray5 & bitArray3) << tendl;

	BitArray array6( bitArray3 );// OK
	array6.Display();


	return 0;
}