#ifndef DEBUG_NEW_H
#define	DEBUG_NEW_H

// http://wyw.dcweb.cn/leakage.htm


	#if defined( _MSC_VER ) && defined( _DEBUG )

		#include <crtdbg.h>

		#define _CRTDBG_MAP_ALLOC

		#ifndef DEBUG_NEW
		#define DEBUG_NEW	new( _NORMAL_BLOCK, __FILE__, __LINE__ )
		#endif

		#ifndef DEBUG_MALLOC
		#define DEBUG_MALLOC(X)	_malloc_dbg( X,_NORMAL_BLOCK, __FILE__, __LINE__ )
		#endif


	#else
		
		#define DEBUG_NEW new
		#define DEBUG_MALLOC malloc


	#endif // defined( _MSC_VER ) && defined( _DEBUG )


#endif // !DEBUG_NEW_H
