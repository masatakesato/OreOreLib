// https://stackoverflow.com/questions/46994332/strcpy-s-buffer-l-buffer-is-too-small-0
// http://www.cplusplus.com/reference/cstdlib/wcstombs/
// https://stackoverflow.com/questions/18645874/converting-stdwsting-to-char-with-wcstombs-s

#include	<stdio.h>
#include	<stdlib.h>

#include	<oreore/common/TString.h>




int main()
{
	tstring src = _T( "AAAAワイド文字列ppp" );
	
	char *multiByteString = NULL;

#ifdef UNICODE
	
	tcout << "Unicode articles..." << tendl;

	//setlocale( LC_ALL, "japanese" );
	//setlocale( LC_CTYPE, "jpn" );

	size_t wcSize = ( src.size() + 1 ) * sizeof(TCHAR);
	multiByteString = new char[wcSize];
	memset( multiByteString, 0, sizeof( char ) * wcSize );
	size_t wLen = 0;
	errno_t err = 0;
	
	//err = wcstombs_s( &wLen, multiByteString, wcSize, src.c_str(), _TRUNCATE );	// Convert from w_char_t to char
	_locale_t locale = _create_locale( LC_CTYPE, "jpn" );
	err = _wcstombs_s_l( &wLen, multiByteString, wcSize, src.c_str(), _TRUNCATE, locale );

#else

	tcout << "Multibyte articles..." << tendl;
	int size = src.length()+1;
	multiByteString = new char[size ];
	strcpy_s( multiByteString, size, src.c_str() );
	
#endif // UNICODE
	
	tcout << src << tendl;//printf( "%s\n", src.c_str() );
	std::cout <<multiByteString << std::endl;//tcout << m_Name << tendl;

	return 0;
}
