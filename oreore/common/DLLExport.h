#ifndef DLL_EXPORT_H
#define	DLL_EXPORT_H


#ifdef EXPORTING_DLL
	#define CLASS_DECLSPEC __declspec( dllexport )
#else
	#define CLASS_DECLSPEC __declspec( dllimport )
#endif

// Disable C4251 warnings
#pragma warning(disable:4251)


#endif // !DLL_EXPORT_H