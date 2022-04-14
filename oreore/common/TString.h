#ifndef	TSTRING_H
#define	TSTRING_H

#include	<tchar.h>
#include	<iostream>
#include	<fstream>
#include	<string>
#include	<locale.h>



using tstring			= std::basic_string<TCHAR>;
using tstringstream		= std::basic_stringstream<TCHAR>;
using tostringstream	= std::basic_ostringstream<TCHAR>;
using tistringstream	= std::basic_istringstream<TCHAR>;
using tifstream			= std::basic_ifstream<TCHAR>;
using tofstream			= std::basic_ofstream<TCHAR>;
using tistream			= std::basic_istream<TCHAR>;
using tostream			= std::basic_ostream<TCHAR>;

using charstring		= std::basic_string<char>;
using wcharstring		= std::basic_string<wchar_t>;


#if defined(UNICODE) || defined(_UNICODE)

	#define tcout		std::wcout
	#define tcin		std::wcin
	#define tcerr		std::wcerr
	#define tclog		std::wclog
	#define tatof		_wtof
	#define tatoi		_wtoi
	#define tatoi64		_wtoi64
	#define tatol		_wtol
	#define to_tstring	std::to_wstring
	#define tsscanf		std::swscanf
	#define tstrcmp		std::wcscmp
	#define tstrcmpi	_wcsicmp// windows only
	#define tstrcpy		std::wcscpy

#else

	#define tcout		std::cout
	#define tcin		std::cin
	#define tcerr		std::cerr
	#define tclog		std::clog
	#define tatof		atof
	#define tatoi		atoi
	#define tatoi64		_atoi64
	#define tatol		atol
	#define to_tstring	std::to_string
	#define tsscanf		std::sscanf
	#define tstrcmp		std::strcmp
	#define tstrcmpi	strcmpi// windows only
	#define tstrcpy		std::strcpy

#endif

#define	tendl	std::endl


#ifndef	MAX_PATH
#define	MAX_PATH	260
#endif



#pragma warning( push )

#pragma warning( disable : 4996 )


inline static void CharToTChar( const char* src, size_t numchars, TCHAR*& dst )
{

#if defined(UNICODE) || defined(_UNICODE)

	dst = new TCHAR[ numchars + 1 ];
	mbstowcs( dst, src, numchars );
	dst[ numchars ] = _T('\0');

#else

	dst = new TCHAR[ numchars ];
	memcpy( dst, src, numchars );

#endif

}



inline static TCHAR* CharToTChar( const char* src, size_t numchars )
{

#if defined(UNICODE) || defined(_UNICODE)

	TCHAR* tchars = new TCHAR[ numchars + 1 ];
	mbstowcs( tchars, src, numchars );
	tchars[ numchars ] = _T('\0');

#else

	TCHAR* tchars = new TCHAR[ numchars ];
	memcpy( tchars, src, numchars );

#endif

	return tchars;
}




inline static void TCharToChar( const TCHAR* src, size_t numchars, char*& dst )
{
#if defined(UNICODE) || defined(_UNICODE)

	dst = new char[ numchars + 1 ];// character length + '\0' space
	wcstombs( dst, src, numchars );
	dst[ numchars ] = '\0';

#else

	dst = new char[ numchars ];
	memcpy( dst, src, numchars );

#endif

}




inline static char* TCharToChar( const TCHAR* src, size_t numchars )
{
#if defined(UNICODE) || defined(_UNICODE)

	char* chars = new char[ numchars + 1 ];// character length + '\0' space
	wcstombs( chars, src, numchars );// copy characters from src
	chars[ numchars ] = '\0';// put '\0' at the end of chars

#else

	char* chars = new char[ numchars ];
	memcpy( chars, src, numchars );

#endif

	return chars;
}


#pragma warning( pop )




inline static tstring CharToTString( const char* src, size_t size=0 )
{
	if( !src )	return _T("");

	TCHAR* tchars = CharToTChar( src, size ? size : strlen(src) );
	tstring tstr( tchars );
	delete [] tchars;

	return tstr;
}



inline static char* TStringToChar( const tstring& tstr )
{
	return TCharToChar( tstr.c_str(), tstr.length() + 1 );
}



template < typename T >
inline typename std::enable_if_t< std::is_arithmetic_v<T>, tstring > NumericToTString( const void* val )
{
	return to_tstring( *(T*)val );
}



template < typename T >
typename std::enable_if_t< std::is_floating_point_v<T>, T > TCharToNumeric( const TCHAR* str )
{
	return (T)tatof( str );
}



template < typename T >
typename std::enable_if_t< std::is_integral_v<T>, T > TCharToNumeric( const TCHAR* str )
{
	return (T)tatoi( str );
}




#endif	// TSTRING_H //