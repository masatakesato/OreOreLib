#ifndef	WINDOW_PROCEDURE_H
#define	WINDOW_PROCEDURE_H


//#include	<io.h>
//#include	<fcntl.h>
#include	<process.h>
#include	<windows.h>
#include	<iostream>
#include	<fstream>

using namespace std;

LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);


#endif	// WINDOW_PROCEDURE_H //