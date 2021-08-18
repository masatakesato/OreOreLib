#include	<crtdbg.h>

#include	<oreore/common/TString.h>
#include	<oreore/container/BitArray.h>



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );


	BitArray src, dst;

	// srcを初期化する
	src.Init( 18 );
	src.Randomize( 0, 18 );

	// srcからバイト配列を取り出して文字列化する
//	uint8* bytes = src.Ptr();
//	TCHAR* tbytes = nullptr;
//	CharToTChar( (char*)bytes, src.ByteSize(), tbytes );

//	uint8* bytes2 = nullptr;
//	TCharToChar( tbytes, src.ByteSize()+1, (char*&)bytes2 );


	tstring tstr = src.ToTString();

	// 文字列化したバイト配列を使ってdstを初期化する
	//dst.Init( 18 );
	//dst.SetPtr( /*bytes2*/ (uint8*)TStringToChar(tstr) );
	dst.Init( 18, tstr );

	// srcとdstが同じかどうかチェックする
	src.Display();
	dst.Display();


//	SafeDeleteArray( tbytes );
//	SafeDeleteArray( bytes2 );




	return 0;
}