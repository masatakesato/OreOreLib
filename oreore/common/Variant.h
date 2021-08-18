#ifndef VARIANT_H
#define	VARIANT_H


#include	"Utility.h"


// http://marupeke296.com/IKDADV_CPP_VARIANT.html


struct Variant
{

	Variant()
		: m_bytes{0}
	{

	}


	template < typename T >
	Variant( const T& val )
	{
		*(T*)m_bytes.bytes = val;
	}


	template < typename T >
	operator T&() const
	{
		return *(T*)m_bytes.bytes;
	}


	template < typename T >
	T& operator=( const T& rhs )
	{
		*(T*)m_bytes.bytes = rhs;
 		return *this;
	}



private:

	union
	{
		uint8	bytes[16];
	} m_bytes;

};







#endif // !VARIANT_H
