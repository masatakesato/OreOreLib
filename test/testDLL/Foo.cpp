//#ifdef DLL_EXPORTS

#include	<iostream>

template< typename T >
void Foo<T>::doSomething( T param )
{
	std::cout << (T)param << std::endl;
}




//#endif