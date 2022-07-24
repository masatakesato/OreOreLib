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
#include	<oreore/common/TString.h>

#include <windows.h>
#include <iostream>
using namespace std;


int main()
{
	auto procId = GetCurrentProcessId();
	cout << "testSubprocess_main... " << procId << endl;
	
	cout << to_string( procId ) << endl;

	//{
	//	//=================== アプリケーション名で起動/終了をコントロールする場合 =================//

	//	system( "start D:/ProgramData/Anaconda3/python.exe test.py" );//"start notepad.exe" );//
	//	    
	//	cout << "!!!!\n";
	//	Sleep( 1000 );

	//	system( "taskkill /F /IM notepad.exe >nul 2>&1" );
	//}

	//{
	//	//=================== プロセスハンドル使って起動/終了をコントロールする場合 ================//

	//	// ShellExecuteEx
	//	SHELLEXECUTEINFO exInfo = { 0 };
	//	exInfo.cbSize = sizeof( SHELLEXECUTEINFO );
	//	exInfo.fMask = SEE_MASK_NOCLOSEPROCESS;//SEE_MASK_DEFAULT;//
	//	exInfo.lpVerb = L"open";
	//	exInfo.lpFile = L"D:/ProgramData/Anaconda3/python.exe";//L"c:/windows/notepad.exe";
	//	exInfo.lpParameters = L"test.py";
	//	exInfo.nShow = SW_NORMAL;

	//	ShellExecuteEx( &exInfo );// ShellExecute( 0, L"open", L"c:\\windows\\notepad.exe", 0, 0, SW_SHOW );
	//	cout << exInfo.hProcess << endl;

	//	Sleep( 5000 );

	//	OreOreLib::CloseProcess( exInfo );
	//}

	{
		//=================== CreateProcess/CloseHandle使って起動/終了をコントロールする場合 ================//

		PROCESS_INFORMATION pInfo = { 0 };
		STARTUPINFO sInfo = { 0 };
		sInfo.cb = sizeof( sInfo );

		//wchar_t commandline[] = L"notepad.exe test.py";
		//wchar_t commandline[] = L"D:/ProgramData/Anaconda3/python.exe test.py -ppid 3333";
		tstring commandline = _T("D:/ProgramData/Anaconda3/python.exe test.py -ppid ") + to_tstring( procId );

		auto ret = CreateProcess(
			NULL,//L"c:\\Windows\\System32\\notepad.exe",
			(TCHAR*)( commandline.c_str() ),//commandline,
			NULL,
			NULL,
			FALSE,
			CREATE_NEW_CONSOLE,
			NULL,
			NULL,
			&sInfo,
			&pInfo
		);

		if( ret == 0 )
		{
			cout << GetLastError() << endl;
		}
		else
		{
			Sleep( 5000 );

			OreOreLib::CloseProcess( pInfo );
		}
	}

	//{
	//	//=================== TODO: PyQtアプリの起動/終了を試してみる ================//
	//
	//	PROCESS_INFORMATION pInfo = { 0 };
	//	STARTUPINFO sInfo = { 0 };
	//	sInfo.cb = sizeof( sInfo );
	//
	//	//wchar_t commandline[] = L"notepad.exe test.py";
	//	wchar_t commandline[] = L"D:/ProgramData/Anaconda3/envs/testpyqt/python.exe D:/Repository/DC/dc/dc_0_0_1/__main__.py";

	//	auto ret = CreateProcess(
	//		NULL,
	//		commandline,
	//		NULL,
	//		NULL,
	//		FALSE,
	//		CREATE_NEW_CONSOLE,
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
	//		Sleep( 7000 );

	//		OreOreLib::CloseProcess( pInfo );
	//	}
	//}

	//{
	//	//=================== プロセスハンドル使って起動/終了をコントロールする場合 ================//

	//	// ShellExecuteEx
	//	SHELLEXECUTEINFO exInfo = { 0 };
	//	exInfo.cbSize = sizeof( SHELLEXECUTEINFO );
	//	exInfo.fMask = SEE_MASK_NOCLOSEPROCESS;//SEE_MASK_DEFAULT;//
	//	exInfo.lpVerb = L"open";
	//	exInfo.lpFile = L"D:/ProgramData/Anaconda3/envs/testpyqt/python.exe";
	//	exInfo.lpParameters = L"D:/Repository/DC/dc/dc_0_0_1/__main__.py";
	//	exInfo.nShow = SW_NORMAL;

	//	ShellExecuteEx( &exInfo );// ShellExecute( 0, L"open", L"c:\\windows\\notepad.exe", 0, 0, SW_SHOW );
	//	cout << exInfo.hProcess << endl;

	//	Sleep( 5000 );

	//	OreOreLib::CloseProcess( exInfo );
	//}


	return 0;
}
