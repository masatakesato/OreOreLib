#include	<Windows.h>
#include	<iostream>

#include	<oreore/common/TString.h>



int main()
{
	auto name = _T("Hoge");
	auto size = 4;



	// Open filehandle
	HANDLE hSharedMemory = CreateFileMapping( INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, name );
	if( hSharedMemory ==NULL )
	{
		return -1;
	}

	// Map to memory
	auto pMemory = MapViewOfFile( hSharedMemory, FILE_MAP_ALL_ACCESS, 0, 0, size );
	if( pMemory == NULL )
	{
		CloseHandle( hSharedMemory );
		return -1;
	}

	for( int i=0; i<10; ++i )
	{
		*reinterpret_cast<int *>(pMemory) = i;
		tcout << i << tendl;
		Sleep( 1000 );
	}

	// Release memory
	UnmapViewOfFile( pMemory );
	CloseHandle( hSharedMemory );

	return 0;
}