#include	"Window.h"


#include	<iostream>
#pragma comment(lib, "User32.lib")	// CreateWindow, ShowWindow, UpdateWindow, RegisterClassEx, LoadIcon, LoadCursor, 




Window::Window(HINSTANCE hInstance, TCHAR *classname, TCHAR *title, LPVOID controller, WNDPROC WinProc) :
	m_hWnd(0), m_hInst(hInstance), m_Style(WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN),
	m_PositionX(CW_USEDEFAULT), m_PositionY(CW_USEDEFAULT),
	m_Width(CW_USEDEFAULT), m_Height(CW_USEDEFAULT),
	m_hParent(0), m_hMenu(0), m_pController(controller)
{

#ifdef _DEBUG
	tcout << _T("Window::Window()...") << tendl;
#endif // _DEBUG

	_tcscpy_s( m_Title, MAX_STRING, title );
	_tcscpy_s( m_ClassName, MAX_STRING, classname );

	//==================== ウィンドウ属性値の初期化 ===================//
	m_WinClass.cbSize			= sizeof(WNDCLASSEX);
	m_WinClass.style			= 0;
	m_WinClass.lpfnWndProc		= (WNDPROC)WinProc;
	m_WinClass.cbClsExtra		= 0;
	m_WinClass.cbWndExtra		= 0;
	m_WinClass.hInstance		= hInstance;
	m_WinClass.hIcon			= LoadIcon(0, IDI_APPLICATION);
	m_WinClass.hCursor			= LoadCursor(0, IDC_ARROW);
	m_WinClass.hbrBackground	= (HBRUSH) (COLOR_WINDOW + 1);
	m_WinClass.lpszMenuName		= NULL;
	m_WinClass.lpszClassName	= m_ClassName;
	m_WinClass.hIconSm			= NULL;

}



Window::~Window()
{

}


HRESULT Window::Create()
{
	if( !RegisterClassEx(&m_WinClass) )
		return E_FAIL;
	
	// クライアント領域サイズに、枠幅をプラスしたウィンドウ全体のサイズを計算する
	RECT rc = { 0, 0, m_Width, m_Height };
	AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE);

	m_hWnd = CreateWindow(	m_ClassName,
							m_Title,
							m_Style,
							m_PositionX,
							m_PositionY,
							rc.right - rc.left,//m_Width,
							rc.bottom - rc.top,//m_Height,
							m_hParent,
							m_hMenu,
							m_hInst,
							m_pController
						);

	if(!m_hWnd)
		return E_FAIL;

	return S_OK;
}


void Window::Show(int nCmdShow)
{
	ShowWindow(m_hWnd, nCmdShow);
	UpdateWindow(m_hWnd);
}