#ifndef FOO_H
#define FOO_H



#ifdef DLL_EXPORTS
#define CLASS_DECLSPEC __declspec(dllexport)
#else
#define CLASS_DECLSPEC __declspec(dllimport)
#endif

#pragma warning(disable:4251)


template< typename T >
struct CLASS_DECLSPEC Foo
{
	void doSomething( T param );
};


template class Foo<int>;

#ifdef DLL_EXPORTS
#include	"Foo.cpp"
#endif

#endif