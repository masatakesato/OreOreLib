#ifndef MATHLIB_H
#define MATHLIB_H


#include	<math.h>
#include	<algorithm>


#ifndef	M_PI
#define	M_PI	3.14159265358979323846f
#endif

#ifndef	M_PI_2
#define	M_PI_2	1.57079632679489661923f
#endif

#ifndef	M_PI_4
#define M_PI_4	0.785398163397448309616f
#endif

#ifndef M_E
#define	M_E		2.718281828459045235360f
#endif // !M_E



#ifndef EPSILON_E4
#define	EPSILON_E4	1e-4
#endif // !EPSILON_E4

#ifndef EPSILON_E5
#define	EPSILON_E5	1e-5
#endif // !EPSILON_E5

#ifndef EPSILON_E6
#define	EPSILON_E6	1e-6
#endif // !EPSILON_E6

#ifndef EPSILON_E9
#define	EPSILON_E9	1e-9
#endif // !EPSILON_E9



template< typename T >
inline bool IsPositive( const T& a )
{
	return	a > 0;
}



template< typename T >
inline bool IsNegative( const T& a )
{
	return	a < 0;
}



template< typename T >
T Sign(  const T& a )
{
	return T( (a>0)-(a<0) );
}



template< typename T1, typename T2 >
inline const T1 Max( const T1& a, const T2& b )
{
	return a < (T1)b ? (T1)b : a;     // or: return comp(a,b)?b:a; for version (2)
}
//template< typename T >
//inline const T Max( const T& a, const T& b )
//{
//	return a < b ? b : a;     // or: return comp(a,b)?b:a; for version (2)
//}



template< typename T1, typename T2 >
inline const T1 Min( const T1& a, const T2& b )
{
	return !( (T1)b < a ) ? a : (T1)b;
}
//template< typename T >
//inline const T Min( const T& a, const T& b )
//{
//	return !( b < a ) ? a : b;
//}



template< typename T >
inline const T Saturate( const T& a )
{
	return Min( Max( a, (T)0 ), (T)1 );
}



template< typename T1, typename T2, typename T3 >
inline const T1 Clamp( const T1& x, const T2& a, const T3& b )
{
	return Min( Max( x, (T1)a ), (T1)b );
}
//template< typename T >
//inline const T Clamp( const T& x, const T& a, const T& b )
//{
//	return Min( Max( x, a ), b );
//}



template< typename T1, typename T2, typename T3, typename T4 >
inline void Lerp( T1& out, const T2& start, const T3& end, T4 percent )
{
	out	= T1( T1(start) + percent * T1( end - start ) );
}
//template< typename T1, typename T2 >
//inline void Lerp( T1& out, const T1& start, const T1& end, T2 percent )
//{
//	out	= T1( T2(start) + percent * T2( end - start ) );
//}
//template< typename T >
//inline void Lerp( T& out, const T& start, const T& end, T percent )
//{
//	out	= start + percent * ( end - start );
//}



template< typename T >
inline const T ToRadian( const T& degree )
{
	return	degree * ( T )0.01745329251;
}



template< typename T >
inline const T ToDegree( const T& radian )
{
	return	radian * ( T )57.2957795131;
}



template< typename T >
inline const T Log( const T& a, const T& b )
{
	return	log( b ) / log( a );
}



template< typename T >
inline const T Floor( const T& val, const T& unit )
{
	return	static_cast<T>( floor( float64( val ) / float64( unit ) ) * float64( unit ) );
}



template< typename T >
inline const T Ceil( const T& val, const T& unit )
{
	return	static_cast<T>( ceil( float64( val ) / float64( unit ) ) * float64( unit ) );
}



template< typename T >
inline const int fRound( const T& x )
{
	return	(int)rintf( x );
}



template< typename T >
inline constexpr/*const*/ T DivUp( const T& a, const T& b )
{
	return	(T)( a%b==0 ? a/b : a/b+1 );
}



template< typename T >
inline constexpr/*const*/ T RoundUp( const T& a, const T& b )
{
	return	(T)( a%b==0 ? a : (a/b+1) * b );
}



template< typename T >
inline constexpr/*const*/ T Round( const T& a, const T& b )
{
	return	(T)( a/b * b );
}



template< typename T >
inline int FastMod( const T& input, const T& ceil )
{
	// apply the modulo operator only when needed
	// (i.e. when the input is greater than the ceiling)
	return input >= ceil ? input % ceil : input;
	// NB: the assumption here is that the numbers are positive
}



template< typename T >
inline void DivMod( T& div, T& mod, const T& input, const T& ceil )
{
	div = input / ceil;
	mod = input >= ceil ? input % ceil : input;
}


#endif /* MATHLIB_H */