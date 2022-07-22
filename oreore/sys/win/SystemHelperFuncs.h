#ifndef WIN_HELPER_FUNCTIONS_H
#define	WIN_HELPER_FUNCTIONS_H

#include	<Windows.h>



namespace OreOreLib
{

	static BOOL CALLBACK SendWMCloseMsg( HWND hwnd, LPARAM lParam )
	{
		DWORD dwProcessId = 0;
		GetWindowThreadProcessId( hwnd, &dwProcessId );

		if( dwProcessId == lParam )
			SendMessageTimeout( hwnd, WM_CLOSE, 0, 0, SMTO_ABORTIFHUNG, 30000, NULL );

		return TRUE;
	}



	static void CloseProcess( PROCESS_INFORMATION pInfo, DWORD timeout_interval=0 )
	{
		CloseHandle( pInfo.hThread );
		WaitForInputIdle( pInfo.hProcess, INFINITE );

		if( WaitForSingleObject( pInfo.hProcess, timeout_interval ) == WAIT_TIMEOUT )
		{
			EnumWindows( &SendWMCloseMsg, pInfo.dwProcessId );
			// force termination if app cannot be closed.
			if( WaitForSingleObject( pInfo.hProcess, timeout_interval ) == WAIT_TIMEOUT )
				TerminateProcess( pInfo.hProcess, 0 );
		}

		CloseHandle( pInfo.hProcess );
	}



	static void CloseProcess( SHELLEXECUTEINFO exInfo, DWORD timeout_interval=0 )
	{
		//CloseHandle( pInfo.hThread );
		WaitForInputIdle( exInfo.hProcess, INFINITE );

		if( WaitForSingleObject( exInfo.hProcess, timeout_interval ) == WAIT_TIMEOUT )
		{
			EnumWindows( &SendWMCloseMsg, (LPARAM)exInfo.hwnd );
			// force termination if app cannot be closed.
			if( WaitForSingleObject( exInfo.hProcess, timeout_interval ) == WAIT_TIMEOUT )
				TerminateProcess( exInfo.hProcess, 0 );
		}

		CloseHandle( exInfo.hProcess );
	}


}// end of OreOreLib


#endif // !WIN_HELPER_FUNCTIONS_H
