#ifndef	WINDOW_H
#define	WINDOW_H

#include	<windows.h>

#include	"../../common/TString.h"



class Window
{
public:
	
	//Window();
	Window( HINSTANCE hInstance, const TCHAR *classname, const TCHAR *title, LPVOID controller, WNDPROC WinProc );
	~Window();

	HRESULT Create();			// ウィンドウ作成
	void	Show(int nCmdShow);	// ウィンドウ表示

	HWND	GetHandle()					{ return m_hWnd; };

	// WNDCLASSEXの各変数を設定する
	void SetClassStyle(UINT style)		{ m_WinClass.style = style; };
	void SetBackGround(int color)		{ m_WinClass.hbrBackground = (HBRUSH) GetStockObject(color); };
	void SetMenuName(LPCTSTR name)		{ m_WinClass.lpszClassName = name; };
	// void	SetIcon(int id)				{ ??? };
	// void SetCursor(int id)			{ ??? };
	
	// CreateWindowの各変数を設定する
	void SetWindowStyle(DWORD style)	{ m_Style = style; };
	void SetPosition(int x, int y)		{ m_PositionX = x; m_PositionY = y; };
	void SetWidth(int width)			{ m_Width = width; };
	void SetHeight(int height)			{ m_Height = height; };
	void SetParent(HWND handle)			{ m_hParent = handle; };
	void SetMenu(HMENU handle)			{ m_hMenu = handle; };


private:

	WNDCLASSEX	m_WinClass;	// ウィンドウクラス生成に使う構造体

	HWND		m_hWnd;		// ウィンドウのハンドル

	enum{ MAX_STRING = 256 };
	TCHAR	m_ClassName[MAX_STRING];// ウィンドウクラスの名称
	TCHAR	m_Title[MAX_STRING];		// タイトル
	DWORD	m_Style;				// スタイル
	int		m_PositionX;			// ウィンドウの位置x
	int		m_PositionY;			// ウィンドウの位置y
	int		m_Width;				// クライアント領域の幅
	int		m_Height;				// クライアント領域の高さ
	HWND	m_hParent;				// 親ウィンドウのハンドル
	HMENU	m_hMenu;				// メニューのハンドル
	HINSTANCE	m_hInst;			//
	LPVOID	m_pController;			// Controllerインスタンスへのポインタ

};





#endif	// WINDOW_H //