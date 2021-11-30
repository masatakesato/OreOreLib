#include	<chrono>
#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/common/Utility.h>




//######################################################################################################//
//     MSB/MLB implementatin by marupeke (http://marupeke296.com/ALG_No2_TLSFMemoryAllocator.html )     //
//######################################################################################################//


// 0101... (=0x55...): 隣接ビット同士の加算用ビットマスク
// 00110011... (=0x33...): 2ビット区間同士の加算用ビットマスク
// 00001111... (=0x0f0f...): 4ビット区間同士の加算用ビットマスク
// 0000000011111111... (=0x00ff...): 8ビット区間同士の加算用ビットマスク
// 00000000000000001111111111111111... (=0x0000ffff...): 16ビット区間同士の加算用ビットマスク

// PrefixSumと同じ要領. 隣接するビット区間同士を再帰的に加算してビットフラグの総数をカウントする.
inline static uint32 GetBitCount( const uint32& val )
{
	unsigned __int32 count = ( val & 0x55555555 ) + ( ( val>>1 ) & 0x55555555 );// sum up single bits.
	count	= ( count & 0x33333333 ) + ( ( count>>2 ) & 0x33333333 );// sum up 2 bit chunks
	count	= ( count & 0x0f0f0f0f ) + ( ( count>>4 ) & 0x0f0f0f0f );// sum up 4 bit chunks
	count	= ( count & 0x00ff00ff ) + ( ( count>>8 ) & 0x00ff00ff );// sum up 8 bit chunks
	return ( count & 0x0000ffff ) + ( ( count>>16 ) & 0x0000ffff );// sum up 16 bit chunks
}

inline static uint64 GetBitCount( const uint64& val )
{
	unsigned __int64 count = ( val & 0x5555555555555555 ) + ( ( val>>1 ) & 0x5555555555555555 );// sum up single bits.
	count	= ( count & 0x3333333333333333 ) + ( ( count>>2 ) & 0x3333333333333333 );// sum up 2 bit chunks
	count	= ( count & 0x0f0f0f0f0f0f0f0f ) + ( ( count>>4 ) & 0x0f0f0f0f0f0f0f0f );// sum up 4 bit chunks
	count	= ( count & 0x00ff00ff00ff00ff ) + ( ( count>>8 ) & 0x00ff00ff00ff00ff );// sum up 8 bit chunks
	count	= ( count & 0x0000ffff0000ffff ) + ( ( count>>16 ) & 0x0000ffff0000ffff );// sum up 16 bit chunks
	return ( count & 0x00000000ffffffff ) + ( ( count>>32 ) & 0x00000000ffffffff );// sum up 32 bit chunks
}

// Get Most Significant bit( 32bit version ).
inline static uint32 GetMSB( uint32 val )
{
	if( val==0 ) return -1;

	val |= ( val >> 1 );
	val |= ( val >> 2 );
	val |= ( val >> 4 );
	val |= ( val >> 8 );
	val |= ( val >> 16 );

	return GetBitCount( val ) - 1;
}

// Get Most Significant bit( 64bit version ).
inline static uint64 GetMSB( uint64 val )
{
	if( val==0 ) return -1;

	val |= ( val >> 1 );
	val |= ( val >> 2 );
	val |= ( val >> 4 );
	val |= ( val >> 8 );
	val |= ( val >> 16 );
	val |= ( val >> 32 );

	return GetBitCount( val ) - 1;
}

// Get Least Significant bit( 32 bit version ).
inline static uint32 GetLSB( uint32 val )
{
	if( val==0 )	return -1;

	val |= ( val << 1 );
	val |= ( val << 2 );
	val |= ( val << 4 );
	val |= ( val << 8 );
	val |= ( val << 16 );

	return 32 - GetBitCount( val );
}

// Get Least Significant bit( 64 bit version ).
inline static uint64 GetLSB( uint64 val )
{
	if( val==0 )	return -1;

	val |= ( val << 1 );
	val |= ( val << 2 );
	val |= ( val << 4 );
	val |= ( val << 8 );
	val |= ( val << 16 );
	val |= ( val << 32 );

	return 64 - GetBitCount( val );
}



//######################################################################################################//
//          Faster MSB/MLB implementation by matteconte. . ( https://github.com/mattconte/tlsf)         //
//######################################################################################################//

// Most Significant Bit index.( 32 bit version. returns [1, 32] )
inline static uint32 _MSB( uint32 val )
{
	__int32 bit = 32;

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
	__int64 bit = 64;

	if( !val ) --bit;
	if( !( val & 0xffffffff00000000 ) ) { val <<= 32; bit -= 32; }
	if( !( val & 0xffff0000ffff0000 ) ) { val <<= 16; bit -= 16; }
	if( !( val & 0xff000000ff000000 ) ) { val <<= 8; bit -= 8; }
	if( !( val & 0xf0000000f0000000 ) ) { val <<= 4; bit -= 4; }
	if( !( val & 0xc0000000c0000000 ) ) { val <<= 2; bit -= 2; }
	if( !( val & 0x8000000080000000 ) ) { val <<= 1; --bit; }

	return bit;
}

// Least Significant Bit index.( 32 bit version. returns [1, 32] )
inline static uint32 _LSB( uint32 val )
{
	__int32 bit = 0;

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
	__int64 bit = 0;

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

// Most Significant Bit index.( 32 bit version. returns [0, 31] )
inline static uint32 GetMSB2( uint32 val )
{
	return _MSB( val ) - 1;
}

// Most Significant Bit index.( 64 bit version. returns [0, 63] )
inline static uint64 GetMSB2( uint64 val )
{
	return _MSB( val ) - 1;
}

// Least Significant Bit index.( 32 bit version. returns [0, 31] )
inline static uint32 GetLSB2( uint32 val )
{
	return _LSB( val ) - 1;
}

// Least Significant Bit index.( 64 bit version. returns [0, 63] )
inline static uint64 GetLSB2( uint64 val )
{
	return _LSB( val ) - 1;
}




//######################################################################################################//
//                                            FLI/SLI functions                                         //
//######################################################################################################//


// Get Second Level Index( 32 bit version ). N: 分割数 = 2^N
static int GetSLI( const uint32& val, int fli, const int N )
{
	//=================== Step by Step Calculation ===================//
	//// 最上位ビット未満のビット列だけを取り出すマスクを計算する.
	//unsigned __int32 mask = ~( 0xffffffff << fli );

	//// valの最上位ビットをゼロにした値を取得する
	//unsigned __int32 val1 = val & mask;

	//// val1のSLI該当ビット列だけ抽出するための右シフト量を計算する. FLIのアドレス範囲を2^N個に分割する場合、val1の上位Nビットだけを使う
	//const unsigned right_shit = fli - N;

	//return val1 >> right_shit;

	//=================== All in one expression ======================//
	return ( val & ~( 0xffffffff << fli ) ) >> ( fli - N );
}


// Get Second Level Index( 64 bit version ).
static int GetSLI( const uint64& val, int fli, int N )
{
	return ( val & ~( 0xffffffffffffffff << fli ) ) >> ( fli - N );
}


// Get Free List Index
static int GetFreeListIndex( int fli, int sli, int N )
{
	return fli * (int)pow( 2, N ) + sli;// - 1;
	// TODO: -1すれば要素1個分だけメモリ節約できるが、インデックス処理で混乱しやすいで一通り実装終わった後の最適化として実施する. 2018.11.01
}





int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	tcout << _T( "//########################## Bit Count test ################################//\n" );

	unsigned __int32 val1 = 0xFFFFFFFF;
	tcout << _T( "32bit GetBitCount( " ) << val1 << _T( " )... " );
	tcout << GetBitCount( val1 ) << tendl;
	
	unsigned __int64 val2 = 0xFFFFFFFFFFFFFFFF;
	tcout << _T( "64bit GetBitCount( " ) << val2 << _T( " )... " );
	tcout << GetBitCount( val2 ) << tendl;


	tcout << _T( "//########################## MSB/LSB test ################################//\n" );

	unsigned __int32 val32 = 24;//0x0FFFFFFF;
	unsigned __int64 val64 = 24;//0x8000000000000000;

	// MSB
	tcout << _T( "32bit marupeke GetMSB( " ) << val32 << _T( " )... " );
	tcout << GetMSB( val32 ) << tendl;

	tcout << _T( "32bit matteconte GetMSB( " ) << val32 << _T( " )... " );
	tcout << GetMSB2( val32 ) << tendl;


	tcout << _T( "64 bit marupeke GetMSB( " ) << val64 << _T( " )... " );
	tcout << GetMSB( val64 ) << tendl;

	tcout << _T( "64 bit matteconte GetMSB( " ) << val64 << _T( " )... " );
	tcout << GetMSB2( val64 ) << tendl;

	// LSB
	tcout << _T( "32bit marupeke GetLSB( " ) << val32 << _T( " )... " );
	tcout << GetLSB( val32 ) << tendl;

	tcout << _T( "32bit matteconte GetLSB( " ) << val32 << _T( " )... " );
	tcout << GetLSB2( val32 ) << tendl;

	tcout << _T( "64 bit marupeke GetLSB( " ) << val64 << _T( " )... " );
	tcout << GetLSB( val64 ) << tendl;

	tcout << _T( "64 bit matteconte GetLSB( " ) << val64 << _T( " )... " );
	tcout << GetLSB2( val64 ) << tendl;




	tcout << _T( "//########################## 32 bit SLI/FLI test ################################//\n" );

	const int N = 2;// assume 2^N=4
	const unsigned __int32 memorySize32 = 300;

	auto fli_32= GetMSB2( memorySize32 );
	auto sli_32 = GetSLI( memorySize32, fli_32, N );

	tcout << "Num of 2nd Level Split: 2^" << N << tendl;
	tcout << "Memory size: " << memorySize32 << tendl;
	tcout << "First Level Index: " << fli_32 << tendl;
	tcout << "Second Level Index: " << sli_32 << tendl;

	tcout << "Free List Length: " << fli_32 * pow( 2, N ) + sli_32 << tendl;


	tcout << _T( "//########################## 64 bit SLI/FLI test ################################//\n" );

	//const int N = 2;// assume 2^N=4
	const unsigned __int64 memorySize64 = 300;

	auto fli_64 = GetMSB2( memorySize64 );
	auto sli_64 = GetSLI( memorySize64, fli_64, N );

	tcout << "Num of 2nd Level Split: 2^" << N << tendl;
	tcout << "Memory size: " << memorySize64 << tendl;
	tcout << "First Level Index: " << fli_64 << tendl;
	tcout << "Second Level Index: " << sli_64 << tendl;

	tcout << "Free List Length: " << fli_64 * pow( 2, N ) + sli_64 << tendl;

	return 0;
}