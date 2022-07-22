//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//int main()
//{
//	auto fp = _popen( "start D:/ProgramData/Anaconda3/python.exe", "r" );
//	std::cout << "I'm still running!" << std::endl;
//
//	cout << _pclose( fp ) << endl;
//	return 0;
//}


// https://stackoverflow.com/questions/33526534/run-python-script-from-c

#include	<oreore/sys/win/SystemHelperFuncs.h>

#include <windows.h>
#include <iostream>
using namespace std;


int main()
{
	//{
	//	//=================== アプリケーション名で起動/終了をコントロールする場合 =================//

	//	system( "start D:/ProgramData/Anaconda3/python.exe test.py" );//"start notepad.exe" );//
	//	    
	//	cout << "!!!!\n";
	//	Sleep( 1000 );

	//	system( "taskkill /F /IM notepad.exe >nul 2>&1" );
	//}

	{
		//=================== プロセスハンドル使って起動/終了をコントロールする場合 ================//

		// ShellExecuteEx
		SHELLEXECUTEINFO exInfo ={ 0 };
		exInfo.cbSize = sizeof( SHELLEXECUTEINFO );
		exInfo.fMask = SEE_MASK_NOCLOSEPROCESS;//SEE_MASK_DEFAULT;//
		exInfo.lpVerb = L"open";
		exInfo.lpFile = L"D:/ProgramData/Anaconda3/python.exe";//L"c:/windows/notepad.exe";
		exInfo.lpParameters = L"test.py";
		exInfo.nShow = SW_NORMAL;

		ShellExecuteEx( &exInfo );// ShellExecute( 0, L"open", L"c:\\windows\\notepad.exe", 0, 0, SW_SHOW );
		cout << exInfo.hProcess << endl;

		Sleep( 5000 );

		// これでプロセス生きてるかどうか分かる
		cout << (WAIT_TIMEOUT==WaitForSingleObject( exInfo.hProcess, 0 )) << endl;

		OreOreLib::CloseProcess( exInfo );
		//TerminateProcess( exInfo.hProcess, 1 );
	}

	//{
	//	//=================== CreateProcess/CloseHandle使って起動/終了をコントロールする場合 ================//

	//	PROCESS_INFORMATION pInfo = { 0 };
	//	STARTUPINFO sInfo = { 0 };
	//	sInfo.cb = sizeof( sInfo );

	//	//wchar_t commandline[] = L"notepad.exe test.py";
	//	wchar_t commandline[] = L"D:/ProgramData/Anaconda3/python.exe";

	//	auto ret = CreateProcess(
	//		NULL,//L"c:\\Windows\\System32\\notepad.exe",
	//		commandline,
	//		NULL,
	//		NULL,
	//		FALSE,
	//		0,
	//		NULL,
	//		NULL,
	//		&sInfo,
	//		&pInfo
	//	);

	//	if( ret == 0 )
	//	{
	//		cout << GetLastError() << endl;
	//	}
	//	else
	//	{
	//		Sleep( 5000 );

	//		//cout << ( WAIT_TIMEOUT==WaitForSingleObject( pInfo.hProcess, 0 ) ) << endl;
	//		OreOreLib::CloseProcess( pInfo );
	//		//TerminateProcess( pInfo.hProcess, 1 );
	//	}

	//}


	return 0;
}
