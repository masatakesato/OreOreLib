#include	"WindowProcedure.h"
#include	"Controller.h"


using namespace std;

LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT	ps;
	HDC			hDC;
	static Controller *ctrl = NULL;
	ctrl = (Controller*)GetWindowLongPtr(hWnd, 
#ifdef _WIN64
		GWLP_USERDATA
#else
		GWL_USERDATA
#endif		
		);

	if(message == WM_NCCREATE)
	{
		ctrl = (Controller*)( ((CREATESTRUCT*)lParam)->lpCreateParams );
		ctrl->SetHandle(hWnd);

		SetWindowLongPtr(hWnd,
#ifdef _WIN64
		GWLP_USERDATA
#else
		GWL_USERDATA
#endif
		,
			(LONG_PTR)ctrl);
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	if(!ctrl)
		return DefWindowProc(hWnd, message, wParam, lParam);
	

	switch(message)
	{
	case WM_CREATE:
		ctrl->Create();
		ctrl->Start();
		break;

	case WM_PAINT:
		hDC = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;

	case WM_LBUTTONDOWN:
		ctrl->lButtonDown(lParam);
		break;

	case WM_LBUTTONUP:
		ctrl->lButtonUp(lParam);
		break;

	case WM_RBUTTONDOWN:
		ctrl->rButtonDown(lParam);
		break;

	case WM_RBUTTONUP:
		ctrl->rButtonUp(lParam);
		break;

	case WM_MOUSEMOVE:
		ctrl->mouseMove(lParam);
		break;

	case WM_KEYDOWN:
		ctrl->keyDown(wParam);
		break;
		
	case WM_KEYUP:
		ctrl->keyUp(wParam);
		break;

	case WM_SIZE:
		ctrl->SetClientSize( LOWORD(lParam), HIWORD(lParam) );
		break;

	case WM_ERASEBKGND:
		break;

	case WM_DESTROY:
		ctrl->Destroy();
		PostQuitMessage(0);
		break;

	case WM_CLOSE:
		ctrl->Close();
		break;

	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	return 0;
}