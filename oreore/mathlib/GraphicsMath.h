#ifndef GRAPHICS_MATH_H
#define GRAPHICS_MATH_H

#include	"MathLib.h"
#include	"../common/Utility.h"
#include	"../container/Array.h"




template< typename T >
union Vec2;

using Vec2uc	= Vec2<uint8>;
using Vec2s		= Vec2<int16>;
using Vec2us	= Vec2<uint16>;
using Vec2i		= Vec2<int32>;
using Vec2ui	= Vec2<uint32>;
using Vec2f		= Vec2<float32>;
using Vec2d		= Vec2<float64>;


template< typename T >
union Vec3;

using Vec3uc	= Vec3<uint8>;
using Vec3s		= Vec3<int16>;
using Vec3us	= Vec3<uint16>;
using Vec3i		= Vec3<int32>;
using Vec3ui	= Vec3<uint32>;
using Vec3f		= Vec3<float32>;
using Vec3d		= Vec3<float64>;


template< typename T >
union Vec4;

using Vec4uc	= Vec4<uint8>;
using Vec4s		= Vec4<int16>;
using Vec4us	= Vec4<uint16>;
using Vec4i		= Vec4<int32>;
using Vec4ui	= Vec4<uint32>;
using Vec4f		= Vec4<float32>;
using Vec4d		= Vec4<float64>;


template< typename T >
union Mat4;

using Mat4f		= Mat4<float32>;
using Mat4d		= Mat4<float64>;


template< typename T >
union Quaternion;

using Quatf		= Quaternion<float32>;
using Quatd		= Quaternion<float64>;


using ucArray	= OreOreLib::Array<uint8>;
using sArray	= OreOreLib::Array<int16>;
using usArray	= OreOreLib::Array<uint16>;
using iArray	= OreOreLib::Array<int32>;
using uiArray	= OreOreLib::Array<uint32>;
using fArray	= OreOreLib::Array<float32>;
using dArray	= OreOreLib::Array<float64>;




//##############################################################################//
//									Scalar										//
//##############################################################################//


union ieee754
{
	struct
	{
		unsigned int mantissa : 23;
		unsigned int exponent : 8;
		unsigned int sign     : 1;
	};

	float            _f32;
	unsigned int     _u32;
};




//##############################################################################//
//									2D Vector									//
//##############################################################################//

// 2次元ベクトル共用体
template< typename T >
union Vec2
{

	struct { T x, y; };
	struct { T u, v; };
	T	xy[2];
	T	uv[2];

	Vec2()
		: x( 0 )
		, y( 0 )
	{

	}


	Vec2( T x_, T y_ )
		: x( x_ )
		, y( y_ )
	{
	
	}


	//=========== experimental implementation. 2018.10.14 ============//
	// Copy constructor
	Vec2( const Vec2& obj )
	{
		x = obj.x;
		y = obj.y;
	}


	// Copy constructor
	~Vec2()
	{

	}


	// Move constructor
	Vec2( Vec2&& obj )
	{
		x = obj.x;
		y = obj.y;
	}


	// Copy assignment operator
	Vec2& operator=( const Vec2& obj )
	{
		if( this != &obj )
		{
			x = obj.x;
			y = obj.y;
		}
		return *this;
	}


	// Move assignment opertor =
	Vec2& operator=( Vec2&& obj )
	{
		if( this != &obj )
		{
			x = obj.x;
			y = obj.y;
		}
		return *this;
	}


	friend tostream& operator<<( tostream& stream, const Vec2& obj )
	{
		stream << _T("(") << obj.x << _T(", ") << obj.y << _T(")");
		return stream;
	}

};



// Init Vector
template< typename T >
void InitVec( Vec2<T>& inout, T x, T y )
{
	inout.x	= x;
	inout.y	= y;
}


// Init Vector
template< typename T >
inline void InitVec( Vec2<T>& inout, T arr[2] )
{
	inout.x	= arr[0];
	inout.y	= arr[1];
}


// Init Vector with zero
template< typename T >
void InitZero( Vec2<T>& inout )
{
	inout.x	= 0;
	inout.y	= 0;
}


// Reverse
template< typename T >
void Reverse( Vec2<T>& out, const Vec2<T>& in )
{
	out.x = -in.x;
	out.y = -in.y;
}


// Reverse
template< typename T >
void Reverse( Vec2<T>& inout )
{
	inout.x = -inout.x;
	inout.y = -inout.y;
}


// Add
template< typename T >
void Add( Vec2<T>& out, const Vec2<T>& in1, const Vec2<T>& in2 )
{
	out.x	= in1.x + in2.x;
	out.y	= in1.y + in2.y;
}


template< typename T >
inline void Add( Vec2<T>& inout, const Vec2<T>& in )
{
	inout.x += in.x;
	inout.y += in.y;
}


// Subtract
template< typename T >
void Subtract( Vec2<T> &out, const Vec2<T>& in1, const Vec2<T>& in2 )
{
	out.x	= in1.x - in2.x;
	out.y	= in1.y - in2.y;
}


template< typename T >
void Subtract( Vec2<T> &inout, const Vec2<T>& in )
{
	inout.x	-= in.x;
	inout.y	-= in.y;
}


// Dot product
template< typename T >
inline T DotProduct( const Vec2<T>& in1, const Vec2<T>& in2 )
{
	return in1.x * in2.x + in1.y * in2.y;
}


// Cross product
template< typename T >
inline void CrossProduct( Vec2<T>& out, const Vec2<T>& in1, const Vec2<T>& in2 )
{
	out.x = 0;
	out.y = 0;
	out.z = in1.x * in2.y - in1.y * in2.x;
}


// Length
template< typename T >
inline T Length( const Vec2<T>& in )
{
	return sqrt( Max( in.x * in.x + in.y * in.y, ( std::numeric_limits<T>::min )( ) ) );
}


// Squared Length
template< typename T >
inline T LengthSqrd( const Vec2<T>& in )
{
	return in.x * in.x + in.y * in.y;
}


// Distance between two vectors
template< typename T >
inline T Distance( const Vec2<T>& in1, const Vec2<T>& in2 )
{
	const T dx	= in1.x - in2.x;
	const T dy	= in1.y - in2.y;
	return	sqrt( Max( dx * dx + dy * dy, ( std::numeric_limits<T>::min )( ) ) );
}


// Squared Distance between two vectors
template< typename T >
inline T DistanceSqrd( const Vec2<T>& in1, const Vec2<T>& in2 )
{
	const T dx	= in1.x - in2.x;
	const T dy	= in1.y - in2.y;
	return	dx * dx + dy * dy;
}


// Normalize
template< typename T >
inline void Normalize( Vec2<T>& inout )
{
	T length_inv	= ( T )1.0 / sqrt( Max( inout.x * inout.x + inout.y * inout.y, ( std::numeric_limits<T>::min )( ) ) );
	inout.x *= length_inv;
	inout.y *= length_inv;
}


// Scale
template< typename T >
inline void Scale( Vec2<T>& inout, T scale )
{
	inout.x *= scale;
	inout.y *= scale;
}


template< typename T >
inline void Scale( Vec2<T>& out, const Vec2<T>& in, T scale )
{
	out.x = in.x * scale;
	out.y = in.y * scale;
}


template< typename T >
inline void Max( Vec2<T>& out, const Vec2<T>& in1, const Vec2<T>& in2 )
{
	out.x	= in1.x > in2.x ? in1.x : in2.x;
	out.y	= in1.y > in2.y ? in1.y : in2.y;
}


template< typename T >
inline void Min( Vec2<T>& out, const Vec2<T>& in1, const Vec2<T>& in2 )
{
	out.x	= in1.x < in2.x ? in1.x : in2.x;
	out.y	= in1.y < in2.y ? in1.y : in2.y;
}


// Clamp
template< typename T >
inline void Clamp( Vec2<T>& inout, const Vec2<T>& minVal, const Vec2<T>& maxVal )
{
	inout.x = Max( Min( inout.x, maxVal.x ), minVal.x );
	inout.y = Max( Min( inout.y, maxVal.y ), minVal.y );
}


// Lerp
template< typename T >
inline void Lerp( Vec2<T>& out, const Vec2<T>& start, const Vec2<T>& end, T percent )
{
	out.x	= start.x + percent * ( end.x - start.x );
	out.y	= start.y + percent * ( end.y - start.y );
}


// Spherilca Linear Interpolation
template< typename T >
inline void Slerp( Vec2<T>& out, const Vec2<T>& start, const Vec2<T>& end, T percent )
{
	// Dot product - the cosine of the angle between 2 vectors.
	T dot = (T)DotProduct( start, end );// Vector3.Dot(start, end);     
										// Clamp it to be in the range of Acos()
										// This may be unnecessary, but floating point
										// precision can be a fickle mistress.
	Clamp( dot, (T)-1, (T)1 ); //Mathf.Clamp(dot, -1.0f, 1.0f);
							   // Acos(dot) returns the angle between start and end,
							   // And multiplying that by percent returns the angle between
							   // start and the final result.
	T theta = (T)acos( dot ) * percent;//Mathf.Acos(dot)*percent;
	Vec2<T> RelativeVec; //Vector3 RelativeVec = end - start*dot;
	RelativeVec.x = end.x - dot * start.x;
	RelativeVec.y = end.y - dot * start.y;

	Normalize( RelativeVec );//RelativeVec.Normalize();     // Orthonormal basis
							 // The final result.
							 //return ((start*Mathf.Cos(theta)) + (RelativeVec*Mathf.Sin(theta)));
	T cos_theta = (T)cos( theta );
	T sin_theta = (T)sin( theta );
	out.x	= start.x * cos_theta +  RelativeVec.x * sin_theta;
	out.y	= start.y * cos_theta +  RelativeVec.y * sin_theta;
}


// Normalized Linear Interpolation
template< typename T >
inline void Nlerp( Vec2<T>& out, const Vec2<T>& start, const Vec2<T>& end, T percent )
{
	Lerp( out, start, end, percent );
	Normalize( out );
	//return Lerp( start, end, percent ).normalized();
}


template< typename T >
inline bool IsSame( const Vec2<T>& in1, const Vec2<T>& in2 )
{
	return in1.x==in2.x && in1.y==in2.y;
}


template< typename T >
inline void AddScaled( Vec2<T>& out, float coeff1, const Vec2<T>& in1, float coeff2, const Vec2<T>& in2 )
{
	out.x	= coeff1 * in1.x + coeff2 * in2.x;
	out.y	= coeff1 * in1.y + coeff2 * in2.y;
}


template< typename T >
inline void AddScaled( Vec2<T>& out, const Vec2<T>& in1, float coeff2, const Vec2<T>& in2 )
{
	out.x	= in1.x + coeff2 * in2.x;
	out.y	= in1.y + coeff2 * in2.y;
}


template< typename T >
inline void AddScaled( Vec2<T>& inout, const Vec2<T>& in, const T scale )
{
	inout.x += in.x * scale;
	inout.y += in.y * scale;
}




//##############################################################################//
//									3D Vector									//
//##############################################################################//

template< typename T >
union Vec3
{
	T	xyz[3];
	T	rgb[3];
	struct { T x, y, z; };
	struct { T r, g, b; };


	Vec3()
	{
		x = 0;
		y = 0;
		z = 0;
	}


	Vec3( T x_, T y_, T z_ )
	{
		x = x_;
		y = y_;
		z = z_;
	}


	friend tostream& operator<<( tostream& stream, const Vec3& obj )
	{
		stream << _T("(") << obj.x << _T(", ") << obj.y << _T(", ") << obj.z << _T(")");
		return stream;
	}

};





// Init Vector
template< typename T >
void InitVec( Vec3<T>& inout, T x, T y, T z )
{
	inout.x	= x;
	inout.y	= y;
	inout.z	= z;
}


// Init Vector
template< typename T >
inline void InitVec( Vec3<T>& inout, T arr[3] )
{
	inout.x	= arr[0];
	inout.y	= arr[1];
	inout.z	= arr[2];
}


// Init Vector with zero
template< typename T >
void InitZero( Vec3<T>& inout )
{
	inout.x	= 0;
	inout.y	= 0;
	inout.z	= 0;
}


// Reverse
template< typename T >
void Reverse( Vec3<T>& out, const Vec3<T>& in )
{
	out.x = -in.x;
	out.y = -in.y;
	out.z = -in.z;
}


// Reverse
template< typename T >
void Reverse( Vec3<T>& inout )
{
	inout.x = -inout.x;
	inout.y = -inout.y;
	inout.z = -inout.z;
}


// Add
template< typename T >
inline void Add( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x = in1.x + in2.x;
	out.y = in1.y + in2.y;
	out.z = in1.z + in2.z;
}


template< typename T >
inline void Add( Vec3<T>& inout, const Vec3<T>& in )
{
	inout.x += in.x;
	inout.y += in.y;
	inout.z += in.z;
}


template< typename T >
inline void AddScaled( Vec3<T>& inout, const Vec3<T>& in, const T scale )
{
	inout.x += in.x * scale;
	inout.y += in.y * scale;
	inout.z += in.z * scale;
}


// Subtract
template< typename T >
inline void Subtract( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x = in1.x - in2.x;
	out.y = in1.y - in2.y;
	out.z = in1.z - in2.z;
}


// Multiply
template< typename T >
inline void Multiply( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x = in1.x * in2.x;
	out.y = in1.y * in2.y;
	out.z = in1.z * in2.z;
}


// Divide
template< typename T >
inline void Divide( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x = in1.x / in2.x;
	out.y = in1.y / in2.y;
	out.z = in1.z / in2.z;
}


// Dot product
template< typename T >
inline T DotProduct( const Vec3<T>& in1, const Vec3<T>& in2 )
{
	return in1.x * in2.x + in1.y * in2.y + in1.z * in2.z;
}


// Cross product
template< typename T >
inline void CrossProduct( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x = in1.y * in2.z - in1.z * in2.y;
	out.y = in1.z * in2.x - in1.x * in2.z;
	out.z = in1.x * in2.y - in1.y * in2.x;
}


// Length
template< typename T >
inline T Length( const Vec3<T>& in )
{
	return sqrt( Max( in.x * in.x + in.y * in.y + in.z * in.z, ( std::numeric_limits<T>::min )( ) ) );
}


// Squared Length
template< typename T >
inline T LengthSqrd( const Vec3<T>& in )
{
	return in.x * in.x + in.y * in.y + in.z * in.z;
}


// Distance between two vectors
template< typename T >
inline T Distance( const Vec3<T>& in1, const Vec3<T>& in2 )
{
	const T dx	= in1.x - in2.x;
	const T dy	= in1.y - in2.y;
	const T dz	= in1.z - in2.z;
	return	sqrt( Max( dx * dx + dy * dy + dz * dz, ( std::numeric_limits<T>::min )( ) ) );
}


// Squared Distance between two vectors
template< typename T >
inline T DistanceSqrd( const Vec3<T>& in1, const Vec3<T>& in2 )
{
	const T dx	= in1.x - in2.x;
	const T dy	= in1.y - in2.y;
	const T dz	= in1.z - in2.z;
	return	dx * dx + dy * dy + dz * dz;
}


// Normalize
template< typename T >
inline void Normalize( Vec3<T>& inout )
{
	T length_inv	= ( T )1.0 / sqrt( Max( inout.x * inout.x + inout.y * inout.y + inout.z * inout.z, ( std::numeric_limits<T>::min )( ) ) );
	inout.x *= length_inv;
	inout.y *= length_inv;
	inout.z *= length_inv;
}


// Scale
template< typename T >
inline void Scale( Vec3<T>& inout, T scale )
{
	inout.x *= scale;
	inout.y *= scale;
	inout.z *= scale;
}


template< typename T >
inline void Scale( Vec3<T>& out, const Vec3<T>& in, T scale )
{
	out.x = in.x * scale;
	out.y = in.y * scale;
	out.z = in.z * scale;
}


template< typename T >
inline void Max( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x	= in1.x > in2.x ? in1.x : in2.x;
	out.y	= in1.y > in2.y ? in1.y : in2.y;
	out.z	= in1.z > in2.z ? in1.z : in2.z;
}


template< typename T >
inline void Min( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 )
{
	out.x	= in1.x < in2.x ? in1.x : in2.x;
	out.y	= in1.y < in2.y ? in1.y : in2.y;
	out.z	= in1.z < in2.z ? in1.z : in2.z;
}


// Clamp
template< typename T >
inline void Clamp( Vec3<T>& inout, const Vec3<T>& minVal, const Vec3<T>& maxVal )
{
	inout.x = Max( Min( inout.x, maxVal.x ), minVal.x );
	inout.y = Max( Min( inout.y, maxVal.y ), minVal.y );
	inout.z = Max( Min( inout.z, maxVal.z ), minVal.z );
}


// Lerp
template< typename T >
inline void Lerp( Vec3<T>& out, const Vec3<T>& start, const Vec3<T>& end, T percent )
{
	out.x	= start.x + percent * ( end.x - start.x );
	out.y	= start.y + percent * ( end.y - start.y );
	out.z	= start.z + percent * ( end.z - start.z );
}


// Spherilca Linear Interpolation
template< typename T >
inline void Slerp( Vec3<T>& out, const Vec3<T>& start, const Vec3<T>& end, T percent )
{
	// Dot product - the cosine of the angle between 2 vectors.
	T dot = (T)DotProduct( start, end );// Vector3.Dot(start, end);     
										// Clamp it to be in the range of Acos()
										// This may be unnecessary, but floating point
										// precision can be a fickle mistress.
	Clamp( dot, (T)-1, (T)1 ); //Mathf.Clamp(dot, -1.0f, 1.0f);
							   // Acos(dot) returns the angle between start and end,
							   // And multiplying that by percent returns the angle between
							   // start and the final result.
	T theta = (T)acos( dot ) * percent;//Mathf.Acos(dot)*percent;
	Vec3<T>	RelativeVec; //Vector3 RelativeVec = end - start*dot;
	RelativeVec.x = end.x - dot * start.x;
	RelativeVec.y = end.y - dot * start.y;
	RelativeVec.z = end.z - dot * start.z;

	Normalize( RelativeVec );//RelativeVec.Normalize();     // Orthonormal basis
							 // The final result.
							 //return ((start*Mathf.Cos(theta)) + (RelativeVec*Mathf.Sin(theta)));
	T cos_theta = (T)cos( theta );
	T sin_theta = (T)sin( theta );
	out.x	= start.x * cos_theta +  RelativeVec.x * sin_theta;
	out.y	= start.y * cos_theta +  RelativeVec.y * sin_theta;
	out.z	= start.z * cos_theta +  RelativeVec.z * sin_theta;
}


// Normalized Linear Interpolation
template< typename T >
inline void Nlerp( Vec3<T>& out, const Vec3<T>& start, const Vec3<T>& end, T percent )
{
	Lerp( out, start, end, percent );
	Normalize( out );
	//return Lerp( start, end, percent ).normalized();
}


template< typename T >
inline bool IsSame( const Vec3<T>& in1, const Vec3<T>& in2 )
{
	return in1.x==in2.x && in1.y==in2.y && in1.z==in2.z;
}


template< typename T >
inline void AddScaled( Vec3<T>& out, float coeff1, const Vec3<T>& in1, float coeff2, const Vec3<T>& in2 )
{
	out.x	= coeff1 * in1.x + coeff2 * in2.x;
	out.y	= coeff1 * in1.y + coeff2 * in2.y;
	out.z	= coeff1 * in1.z + coeff2 * in2.z;
}


template< typename T >
inline void AddScaled( Vec3<T>& out, const Vec3<T>& in1, float coeff2, const Vec3<T>& in2 )
{
	out.x	= in1.x + coeff2 * in2.x;
	out.y	= in1.y + coeff2 * in2.y;
	out.z	= in1.z + coeff2 * in2.z;
}



//##############################################################################//
//									4D Vector									//
//##############################################################################//

template< typename T >
union Vec4
{
	T	xyzw[4];
	T	rgba[4];
	struct { T x, y, z, w; };
	struct { T r, g, b, a; };
	struct { Vec3<T>xyz; T w; };


	Vec4() : xyz()
	{
		//x = 0;
		//y = 0;
		//z = 0;
		w = 0;
	}


	Vec4( T x_, T y_, T z_, T w_ )
	{
		x = x_;
		y = y_;
		z = z_;
		w = w_;
	}


	friend tostream& operator<<( tostream& stream, const Vec4& obj )
	{
		stream << _T("(") << obj.x << _T(", ") << obj.y << _T(", ") << obj.z << _T(", ") << obj.w << _T(")");
		return stream;
	}

};




// Init Vector
template< typename T >
void InitVec( Vec4<T>& inout, T x, T y, T z, T w )
{
	inout.x	= x;
	inout.y	= y;
	inout.z	= z;
	inout.w	= w;
}


// Init Vector
template< typename T >
inline void InitVec( Vec4<T>& inout, T arr[4] )
{
	inout.x	= arr[0];
	inout.y	= arr[1];
	inout.z	= arr[2];
	inout.w	= arr[3];
}


// Init Vector with zero
template< typename T >
void InitZero( Vec4<T>& inout )
{
	inout.x	= 0;
	inout.y	= 0;
	inout.z	= 0;
	inout.w	= 0;
}


// Reverse
template< typename T >
void Reverse( Vec4<T>& out, const Vec4<T>& in )
{
	out.x = -in.x;
	out.y = -in.y;
	out.z = -in.z;
	out.w = -in.w;
}


// Reverse
template< typename T >
void Reverse( Vec4<T>& inout )
{
	inout.x = -inout.x;
	inout.y = -inout.y;
	inout.z = -inout.z;
	inout.w = -inout.w;
}


// Add
template< typename T >
inline void Add( Vec4<T>& out, const Vec4<T>& in1, const Vec4<T>& in2 )
{
	out.x = in1.x + in2.x;
	out.y = in1.y + in2.y;
	out.z = in1.z + in2.z;
	out.w = in1.w + in2.w;
}


template< typename T >
inline void Add( Vec4<T>& inout, const Vec4<T>& in )
{
	inout.x += in.x;
	inout.y += in.y;
	inout.z += in.z;
	inout.w += in.w;
}


// Subtract
template< typename T >
inline void Subtract( Vec4<T>& out, const Vec4<T>& in1, const Vec4<T>& in2 )
{
	out.x = in1.x - in2.x;
	out.y = in1.y - in2.y;
	out.z = in1.z - in2.z;
	out.w = in1.w - in2.w;
}


// Dot product
template< typename T >
inline T DotProduct( const Vec4<T>& in1, const Vec4<T>& in2 )
{
	return in1.x * in2.x + in1.y * in2.y + in1.z * in2.z + in1.w * in2.w;
}


// Cross product
//template< typename T >
//inline void CrossProduct( Vec3<T>& out, const Vec3<T>& in1, const Vec3<T>& in2 ) 
//{
//	out.x = in1.y * in2.z - in1.z * in2.y;
//	out.y = in1.z * in2.x - in1.x * in2.z;
//	out.z = in1.x * in2.y - in1.y * in2.x;
//}


// Length
template< typename T >
inline T Length( const Vec4<T>& in )
{
	return sqrt( Max( in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w, ( std::numeric_limits<T>::min )( ) ) );
}


// Squared Length
template< typename T >
inline T LengthSqrd( const Vec4<T>& in )
{
	return in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w;
}


// Distance between two vectors
template< typename T >
inline T Distance( const Vec4<T>& in1, const Vec4<T>& in2 )
{
	const T dx	= in1.x - in2.x;
	const T dy	= in1.y - in2.y;
	const T dz	= in1.z - in2.z;
	const T dw	= in1.w - in2.w;
	return	sqrt( Max( dx * dx + dy * dy + dz * dz + dw * dw, ( std::numeric_limits<T>::min )( ) ) );
}


// Squared Distance between two vectors
template< typename T >
inline T DistanceSqrd( const Vec4<T>& in1, const Vec4<T>& in2 )
{
	const T dx	= in1.x - in2.x;
	const T dy	= in1.y - in2.y;
	const T dz	= in1.z - in2.z;
	const T dw	= in1.w - in2.w;
	return	dx * dx + dy * dy + dz * dz + dw* dw;
}


// Normalize
template< typename T >
inline void Normalize( Vec4<T>& inout )
{
	T length_inv	= ( T )1.0 / sqrt( Max( inout.x * inout.x + inout.y * inout.y + inout.z * inout.z + inout.w * inout.w, ( std::numeric_limits<T>::min )( ) ) );
	inout.x *= length_inv;
	inout.y *= length_inv;
	inout.z *= length_inv;
	inout.w *= length_inv;
}


// Scale
template< typename T >
inline void Scale( Vec4<T>& inout, T scale )
{
	inout.x *= scale;
	inout.y *= scale;
	inout.z *= scale;
	inout.w *= scale;
}


template< typename T >
inline void Scale( Vec4<T>& out, const Vec4<T>& in, T scale )
{
	out.x = in.x * scale;
	out.y = in.y * scale;
	out.z = in.z * scale;
	out.w = in.w * scale;
}


template< typename T >
inline void Max( Vec4<T>& out, const Vec4<T>& in1, const Vec4<T>& in2 )
{
	out.x	= in1.x > in2.x ? in1.x : in2.x;
	out.y	= in1.y > in2.y ? in1.y : in2.y;
	out.z	= in1.z > in2.z ? in1.z : in2.z;
	out.w	= in1.w > in2.w ? in1.w : in2.w;
}


template< typename T >
inline void Min( Vec4<T>& out, const Vec4<T>& in1, const Vec4<T>& in2 )
{
	out.x	= in1.x < in2.x ? in1.x : in2.x;
	out.y	= in1.y < in2.y ? in1.y : in2.y;
	out.z	= in1.z < in2.z ? in1.z : in2.z;
	out.w	= in1.w < in2.w ? in1.w : in2.w;
}


// Clamp
template< typename T >
inline void Clamp( Vec4<T>& inout, const Vec4<T>& minVal, const Vec4<T>& maxVal )
{
	inout.x = Max( Min( inout.x, maxVal.x ), minVal.x );
	inout.y = Max( Min( inout.y, maxVal.y ), minVal.y );
	inout.z = Max( Min( inout.z, maxVal.z ), minVal.z );
	inout.w = Max( Min( inout.w, maxVal.w ), minVal.w );
}


// Lerp
template< typename T >
inline void Lerp( Vec4<T>& out, const Vec4<T>& start, const Vec4<T>& end, T percent )
{
	out.x	= start.x + percent * ( end.x - start.x );
	out.y	= start.y + percent * ( end.y - start.y );
	out.z	= start.z + percent * ( end.z - start.z );
	out.w	= start.w + percent * ( end.w - start.w );
}


// Spherilca Linear Interpolation
template< typename T >
inline void Slerp( Vec4<T>& out, const Vec4<T>& start, const Vec4<T>& end, T percent )
{
	// Dot product - the cosine of the angle between 2 vectors.
	T dot = (T)DotProduct( start, end );// Vector3.Dot(start, end);
										// Clamp it to be in the range of Acos()
										// This may be unnecessary, but floating point
										// precision can be a fickle mistress.
	Clamp( dot, (T)-1, (T)1 ); //Mathf.Clamp(dot, -1.0f, 1.0f);
							   // Acos(dot) returns the angle between start and end,
							   // And multiplying that by percent returns the angle between
							   // start and the final result.
	T theta = (T)acos( dot ) * percent;//Mathf.Acos(dot)*percent;
	Vec4<T>	RelativeVec; //Vector3 RelativeVec = end - start*dot;
	RelativeVec.x = end.x - dot * start.x;
	RelativeVec.y = end.y - dot * start.y;
	RelativeVec.z = end.z - dot * start.z;
	RelativeVec.w = end.w - dot * start.w;

	Normalize( RelativeVec );//RelativeVec.Normalize();     // Orthonormal basis
							 // The final result.
							 //return ((start*Mathf.Cos(theta)) + (RelativeVec*Mathf.Sin(theta)));
	T cos_theta = (T)cos( theta );
	T sin_theta = (T)sin( theta );
	out.x	= start.x * cos_theta +  RelativeVec.x * sin_theta;
	out.y	= start.y * cos_theta +  RelativeVec.y * sin_theta;
	out.z	= start.z * cos_theta +  RelativeVec.z * sin_theta;
	out.w	= start.w * cos_theta +  RelativeVec.w * sin_theta;
}


// Normalized Linear Interpolation
template< typename T >
inline void Nlerp( Vec4<T>& out, const Vec4<T>& start, const Vec4<T>& end, T percent )
{
	Lerp( out, start, end, percent );
	Normalize( out );
	//return Lerp( start, end, percent ).normalized();
}


template< typename T >
inline bool IsSame( const Vec4<T>& in1, const Vec4<T>& in2 )
{
	return in1.x==in2.x && in1.y==in2.y && in1.z==in2.z && in1.w==in2.w;
}


template< typename T >
inline void AddScaled( Vec4<T>& out, float coeff1, const Vec4<T>& in1, float coeff2, const Vec4<T>& in2 )
{
	out.x	= coeff1 * in1.x + coeff2 * in2.x;
	out.y	= coeff1 * in1.y + coeff2 * in2.y;
	out.z	= coeff1 * in1.z + coeff2 * in2.z;
	out.w	= coeff1 * in1.w + coeff2 * in2.w;
}


template< typename T >
inline void AddScaled( Vec4<T>& out, const Vec4<T>& in1, float coeff2, const Vec4<T>& in2 )
{
	out.x	= in1.x + coeff2 * in2.x;
	out.y	= in1.y + coeff2 * in2.y;
	out.z	= in1.z + coeff2 * in2.z;
	out.w	= in1.w + coeff2 * in2.w;
}



//##############################################################################//
//									4x4 Matrix									//
//##############################################################################//

// 4x4 matrix union. left to right multiplication order
template< typename T>
union Mat4
{
	struct
	{
		T	m00, m01, m02, m03,
			m10, m11, m12, m13,
			m20, m21, m22, m23,
			m30, m31, m32, m33;
	};

	struct
	{
		Vec4<T>	mat[4];
	};

	//T		m[4][4];
	T		m[16];


	Mat4()
	{
		m00=0; m01=0; m02=0; m03=0;
		m10=0; m11=0; m12=0; m13=0;
		m20=0; m21=0; m22=0; m23=0;
		m30=0; m31=0; m32=0; m33=0;
	}

};




template< typename T >
inline void MatInit( Mat4<T>& inout,
	const T& m00, const T& m01, const T& m02, const T& m03,
	const T& m10, const T& m11, const T& m12, const T& m13,
	const T& m20, const T& m21, const T& m22, const T& m23,
	const T& m30, const T& m31, const T& m32, const T& m33 )
{
	inout.m00 = m00; inout.m01 = m01; inout.m02 = m02; inout.m03 = m03;
	inout.m10 = m10; inout.m11 = m11; inout.m12 = m12; inout.m13 = m13;
	inout.m20 = m20; inout.m21 = m21; inout.m22 = m22; inout.m23 = m23;
	inout.m30 = m30; inout.m31 = m31; inout.m32 = m32; inout.m33 = m33;
}



template< typename T >
inline void MatIdentity( Mat4<T>& inout )
{
	inout.m00 = 1; inout.m01 = 0; inout.m02 = 0; inout.m03 = 0;
	inout.m10 = 0; inout.m11 = 1; inout.m12 = 0; inout.m13 = 0;
	inout.m20 = 0; inout.m21 = 0; inout.m22 = 1; inout.m23 = 0;
	inout.m30 = 0; inout.m31 = 0; inout.m32 = 0; inout.m33 = 1;
}



template< typename T >
inline void MatZero( Mat4<T>& inout )
{
	inout.m00 = 0; inout.m01 = 0; inout.m02 = 0; inout.m03 = 0;
	inout.m10 = 0; inout.m11 = 0; inout.m12 = 0; inout.m13 = 0;
	inout.m20 = 0; inout.m21 = 0; inout.m22 = 0; inout.m23 = 0;
	inout.m30 = 0; inout.m31 = 0; inout.m32 = 0; inout.m33 = 0;
}



// inverse　
template< typename T >
inline void MatInverse( Mat4<T>& out, T& pDeterminant, const Mat4<T>& in )
{
	//pDeterminant =	(in.m00 * in.m11 * in.m22 * in.m33) + (in.m00 * in.m12 * in.m23 * in.m31) + (in.m00 * in.m13 * in.m21 * in.m32)
	//			+	(in.m01 * in.m10 * in.m23 * in.m32) + (in.m01 * in.m12 * in.m20 * in.m33) + (in.m01 * in.m13 * in.m22 * in.m30)
	//			+	(in.m02 * in.m10 * in.m21 * in.m33) + (in.m02 * in.m11 * in.m23 * in.m30) + (in.m02 * in.m13 * in.m20 * in.m31)
	//			+	(in.m03 * in.m10 * in.m22 * in.m31) + (in.m03 * in.m11 * in.m20 * in.m32) + (in.m03 * in.m12 * in.m21 * in.m30)
	//			-	(in.m00 * in.m11 * in.m23 * in.m32) - (in.m00 * in.m12 * in.m21 * in.m33) - (in.m00 * in.m13 * in.m22 * in.m31)
	//			-	(in.m01 * in.m10 * in.m22 * in.m33) - (in.m01 * in.m12 * in.m23 * in.m30) - (in.m01 * in.m13 * in.m20 * in.m32)
	//			-	(in.m02 * in.m10 * in.m23 * in.m31) - (in.m02 * in.m11 * in.m20 * in.m33) - (in.m02 * in.m13 * in.m21 * in.m30)
	//			-	(in.m03 * in.m10 * in.m21 * in.m32) - (in.m03 * in.m11 * in.m22 * in.m30) - (in.m03 * in.m12 * in.m20 * in.m31);

	//	
	//out.m00	= ( (in.m11 * in.m22 * in.m33) + (in.m12 * in.m23 * in.m31) + (in.m13 * in.m21 * in.m32) - (in.m11 * in.m23 * in.m32) - (in.m12 * in.m21 * in.m33) - (in.m13 * in.m22 * in.m31) ) / pDeterminant;
	//out.m01	= ( (in.m01 * in.m23 * in.m32) + (in.m02 * in.m21 * in.m33) + (in.m03 * in.m22 * in.m31) - (in.m01 * in.m22 * in.m33) - (in.m02 * in.m23 * in.m31) - (in.m03 * in.m21 * in.m32) ) / pDeterminant;
	//out.m02	= ( (in.m01 * in.m12 * in.m33) + (in.m02 * in.m13 * in.m31) + (in.m03 * in.m11 * in.m32) - (in.m01 * in.m13 * in.m32) - (in.m02 * in.m11 * in.m33) - (in.m03 * in.m12 * in.m31) ) / pDeterminant;
	//out.m03	= ( (in.m01 * in.m13 * in.m22) + (in.m02 * in.m11 * in.m23) + (in.m03 * in.m12 * in.m21) - (in.m01 * in.m12 * in.m23) - (in.m02 * in.m13 * in.m21) - (in.m03 * in.m11 * in.m22) ) / pDeterminant;
	//	
	//out.m10	= ( (in.m10 * in.m23 * in.m32) + (in.m12 * in.m20 * in.m33) + (in.m13 * in.m22 * in.m30) - (in.m10 * in.m22 * in.m33) - (in.m12 * in.m23 * in.m30) - (in.m13 * in.m20 * in.m32) ) / pDeterminant;
	//out.m11	= ( (in.m00 * in.m22 * in.m33) + (in.m02 * in.m23 * in.m30) + (in.m03 * in.m20 * in.m32) - (in.m00 * in.m23 * in.m32) - (in.m02 * in.m20 * in.m33) - (in.m03 * in.m22 * in.m30) ) / pDeterminant;
	//out.m12	= ( (in.m00 * in.m13 * in.m32) + (in.m02 * in.m10 * in.m33) + (in.m03 * in.m12 * in.m30) - (in.m00 * in.m12 * in.m33) - (in.m02 * in.m13 * in.m30) - (in.m03 * in.m10 * in.m32) ) / pDeterminant;
	//out.m13	= ( (in.m00 * in.m12 * in.m23) + (in.m02 * in.m13 * in.m20) + (in.m03 * in.m10 * in.m22) - (in.m00 * in.m13 * in.m22) - (in.m02 * in.m10 * in.m23) - (in.m03 * in.m12 * in.m20) ) / pDeterminant;
	//	
	//out.m20	= ( (in.m10 * in.m21 * in.m33) + (in.m11 * in.m23 * in.m30) + (in.m13 * in.m20 * in.m31) - (in.m10 * in.m23 * in.m31) - (in.m11 * in.m20 * in.m33) - (in.m13 * in.m21 * in.m30) ) / pDeterminant;
	//out.m21	= ( (in.m00 * in.m23 * in.m31) + (in.m01 * in.m20 * in.m33) + (in.m03 * in.m21 * in.m30) - (in.m00 * in.m21 * in.m33) - (in.m01 * in.m23 * in.m30) - (in.m03 * in.m20 * in.m31) ) / pDeterminant;
	//out.m22	= ( (in.m00 * in.m11 * in.m33) + (in.m01 * in.m13 * in.m30) + (in.m03 * in.m10 * in.m31) - (in.m00 * in.m13 * in.m31) - (in.m01 * in.m10 * in.m33) - (in.m03 * in.m11 * in.m30) ) / pDeterminant;
	//out.m23	= ( (in.m00 * in.m13 * in.m21) + (in.m01 * in.m10 * in.m23) + (in.m03 * in.m11 * in.m20) - (in.m00 * in.m11 * in.m23) - (in.m01 * in.m13 * in.m20) - (in.m03 * in.m10 * in.m21) ) / pDeterminant;

	//out.m30	= ( (in.m10 * in.m22 * in.m31) + (in.m11 * in.m20 * in.m32) + (in.m12 * in.m21 * in.m30) - (in.m10 * in.m21 * in.m32) - (in.m11 * in.m22 * in.m30) - (in.m12 * in.m20 * in.m31) ) / pDeterminant;
	//out.m31	= ( (in.m00 * in.m21 * in.m32) + (in.m01 * in.m22 * in.m30) + (in.m02 * in.m20 * in.m31) - (in.m00 * in.m22 * in.m31) - (in.m01 * in.m20 * in.m32) - (in.m02 * in.m21 * in.m30) ) / pDeterminant;
	//out.m32	= ( (in.m00 * in.m12 * in.m31) + (in.m01 * in.m10 * in.m32) + (in.m02 * in.m11 * in.m30) - (in.m00 * in.m11 * in.m32) - (in.m01 * in.m12 * in.m30) - (in.m02 * in.m10 * in.m31) ) / pDeterminant;
	//out.m33	= ( (in.m00 * in.m11 * in.m22) + (in.m01 * in.m12 * in.m20) + (in.m02 * in.m10 * in.m21) - (in.m00 * in.m12 * in.m21) - (in.m01 * in.m10 * in.m22) - (in.m02 * in.m11 * in.m20) ) / pDeterminant;
	//
	T s0	= in.m00 * in.m11 - in.m10 * in.m01;
	T s1	= in.m00 * in.m12 - in.m10 * in.m02;
	T s2	= in.m00 * in.m13 - in.m10 * in.m03;
	T s3	= in.m01 * in.m12 - in.m11 * in.m02;
	T s4	= in.m01 * in.m13 - in.m11 * in.m03;
	T s5	= in.m02 * in.m13 - in.m12 * in.m03;

	T c5	= in.m22 * in.m33 - in.m32 * in.m23;
	T c4	= in.m21 * in.m33 - in.m31 * in.m23;
	T c3	= in.m21 * in.m32 - in.m31 * in.m22;
	T c2	= in.m20 * in.m33 - in.m30 * in.m23;
	T c1	= in.m20 * in.m32 - in.m30 * in.m22;
	T c0	= in.m20 * in.m31 - in.m30 * in.m21;


	// Should check for 0 determinant
	pDeterminant	= ( s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0 );
	T invdet	= (T)1 / pDeterminant;

	out.m00	= ( in.m11 * c5 - in.m12 * c4 + in.m13 * c3 ) * invdet;
	out.m01	= ( -in.m01 * c5 + in.m02 * c4 - in.m03 * c3 ) * invdet;
	out.m02	= ( in.m31 * s5 - in.m32 * s4 + in.m33 * s3 ) * invdet;
	out.m03	= ( -in.m21 * s5 + in.m22 * s4 - in.m23 * s3 ) * invdet;

	out.m10	= ( -in.m10 * c5 + in.m12 * c2 - in.m13 * c1 ) * invdet;
	out.m11	= ( in.m00 * c5 - in.m02 * c2 + in.m03 * c1 ) * invdet;
	out.m12	= ( -in.m30 * s5 + in.m32 * s2 - in.m33 * s1 ) * invdet;
	out.m13	= ( in.m20 * s5 - in.m22 * s2 + in.m23 * s1 ) * invdet;

	out.m20	= ( in.m10 * c4 - in.m11 * c2 + in.m13 * c0 ) * invdet;
	out.m21	= ( -in.m00 * c4 + in.m01 * c2 - in.m03 * c0 ) * invdet;
	out.m22	= ( in.m30 * s4 - in.m31 * s2 + in.m33 * s0 ) * invdet;
	out.m23	= ( -in.m20 * s4 + in.m21 * s2 - in.m23 * s0 ) * invdet;

	out.m30	= ( -in.m10 * c3 + in.m11 * c1 - in.m12 * c0 ) * invdet;
	out.m31	= ( in.m00 * c3 - in.m01 * c1 + in.m02 * c0 ) * invdet;
	out.m32	= ( -in.m30 * s3 + in.m31 * s1 - in.m32 * s0 ) * invdet;
	out.m33	= ( in.m20 * s3 - in.m21 * s1 + in.m22 * s0 ) * invdet;


}



template< typename T >
inline void MatInverse( Mat4<T>& out, const Mat4<T>& in )
{
	T s0	= in.m00 * in.m11 - in.m10 * in.m01;
	T s1	= in.m00 * in.m12 - in.m10 * in.m02;
	T s2	= in.m00 * in.m13 - in.m10 * in.m03;
	T s3	= in.m01 * in.m12 - in.m11 * in.m02;
	T s4	= in.m01 * in.m13 - in.m11 * in.m03;
	T s5	= in.m02 * in.m13 - in.m12 * in.m03;

	T c5	= in.m22 * in.m33 - in.m32 * in.m23;
	T c4	= in.m21 * in.m33 - in.m31 * in.m23;
	T c3	= in.m21 * in.m32 - in.m31 * in.m22;
	T c2	= in.m20 * in.m33 - in.m30 * in.m23;
	T c1	= in.m20 * in.m32 - in.m30 * in.m22;
	T c0	= in.m20 * in.m31 - in.m30 * in.m21;


	// Should check for 0 determinant
	T invdet	= (T)1 / ( s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0 );

	out.m00	= ( in.m11 * c5 - in.m12 * c4 + in.m13 * c3 ) * invdet;
	out.m01	= ( -in.m01 * c5 + in.m02 * c4 - in.m03 * c3 ) * invdet;
	out.m02	= ( in.m31 * s5 - in.m32 * s4 + in.m33 * s3 ) * invdet;
	out.m03	= ( -in.m21 * s5 + in.m22 * s4 - in.m23 * s3 ) * invdet;

	out.m10	= ( -in.m10 * c5 + in.m12 * c2 - in.m13 * c1 ) * invdet;
	out.m11	= ( in.m00 * c5 - in.m02 * c2 + in.m03 * c1 ) * invdet;
	out.m12	= ( -in.m30 * s5 + in.m32 * s2 - in.m33 * s1 ) * invdet;
	out.m13	= ( in.m20 * s5 - in.m22 * s2 + in.m23 * s1 ) * invdet;

	out.m20	= ( in.m10 * c4 - in.m11 * c2 + in.m13 * c0 ) * invdet;
	out.m21	= ( -in.m00 * c4 + in.m01 * c2 - in.m03 * c0 ) * invdet;
	out.m22	= ( in.m30 * s4 - in.m31 * s2 + in.m33 * s0 ) * invdet;
	out.m23	= ( -in.m20 * s4 + in.m21 * s2 - in.m23 * s0 ) * invdet;

	out.m30	= ( -in.m10 * c3 + in.m11 * c1 - in.m12 * c0 ) * invdet;
	out.m31	= ( in.m00 * c3 - in.m01 * c1 + in.m02 * c0 ) * invdet;
	out.m32	= ( -in.m30 * s3 + in.m31 * s1 - in.m32 * s0 ) * invdet;
	out.m33	= ( in.m20 * s3 - in.m21 * s1 + in.m22 * s0 ) * invdet;
}






// transpose
template< typename T >
inline void MatTranspose( Mat4<T>& out, const Mat4<T>& in )
{
	out.m00	= in.m00;
	out.m01	= in.m10;
	out.m02	= in.m20;
	out.m03	= in.m30;

	out.m10	= in.m01;
	out.m11	= in.m11;
	out.m12	= in.m21;
	out.m13	= in.m31;

	out.m20	= in.m02;
	out.m21	= in.m12;
	out.m22	= in.m22;
	out.m23	= in.m32;

	out.m30	= in.m03;
	out.m31	= in.m13;
	out.m32	= in.m23;
	out.m33	= in.m33;
}


// add
template< typename T >
inline void Add( Mat4<T>& out, const Mat4<T>& in1, const Mat4<T>& in2 )
{
	out.m00	= in1.m00 + in2.m00;
	out.m01	= in1.m01 + in2.m01;
	out.m02	= in1.m02 + in2.m02;
	out.m03	= in1.m03 + in2.m03;

	out.m10	= in1.m10 + in2.m10;
	out.m11	= in1.m11 + in2.m11;
	out.m12	= in1.m12 + in2.m12;
	out.m13	= in1.m13 + in2.m13;

	out.m20	= in1.m20 + in2.m20;
	out.m21	= in1.m21 + in2.m21;
	out.m22	= in1.m22 + in2.m22;
	out.m23	= in1.m23 + in2.m23;

	out.m30	= in1.m30 + in2.m30;
	out.m31	= in1.m31 + in2.m31;
	out.m32	= in1.m32 + in2.m32;
	out.m33	= in1.m33 + in2.m33;
}


// subtract
template< typename T >
inline void Subtract( Mat4<T>& out, const Mat4<T>& in1, const Mat4<T>& in2 )
{
	out.m00	= in1.m00 - in2.m00;
	out.m01	= in1.m01 - in2.m01;
	out.m02	= in1.m02 - in2.m02;
	out.m03	= in1.m03 - in2.m03;

	out.m10	= in1.m10 - in2.m10;
	out.m11	= in1.m11 - in2.m11;
	out.m12	= in1.m12 - in2.m12;
	out.m13	= in1.m13 - in2.m13;

	out.m20	= in1.m20 - in2.m20;
	out.m21	= in1.m21 - in2.m21;
	out.m22	= in1.m22 - in2.m22;
	out.m23	= in1.m23 - in2.m23;

	out.m30	= in1.m30 - in2.m30;
	out.m31	= in1.m31 - in2.m31;
	out.m32	= in1.m32 - in2.m32;
	out.m33	= in1.m33 - in2.m33;
}


// note: in.w is assumed to be 1.0
template< typename T >
inline void Multiply( Vec3<T>& out, const Mat4<T>& mat, const Vec3<T>& in )
{
	out.x	= mat.m00 * in.x +  mat.m01 * in.y +  mat.m02 * in.z + mat.m03;
	out.y	= mat.m10 * in.x +  mat.m11 * in.y +  mat.m12 * in.z + mat.m13;
	out.z	= mat.m20 * in.x +  mat.m21 * in.y +  mat.m22 * in.z + mat.m23;
}


template< typename T >
inline void Multiply( Vec4<T>& out, const Mat4<T>& mat, const Vec3<T>& in )
{
	out.x	= mat.m00 * in.x +  mat.m01 * in.y +  mat.m02 * in.z + mat.m03;
	out.y	= mat.m10 * in.x +  mat.m11 * in.y +  mat.m12 * in.z + mat.m13;
	out.z	= mat.m20 * in.x +  mat.m21 * in.y +  mat.m22 * in.z + mat.m23;
	out.w	= mat.m30 * in.x +  mat.m31 * in.y +  mat.m32 * in.z + mat.m33;
}



template< typename T >
inline void Multiply( Vec4<T>& out, const Mat4<T>& mat, const Vec4<T>& in )
{
	out.x	= mat.m00 * in.x +  mat.m01 * in.y +  mat.m02 * in.z + mat.m03 * in.w;
	out.y	= mat.m10 * in.x +  mat.m11 * in.y +  mat.m12 * in.z + mat.m13 * in.w;
	out.z	= mat.m20 * in.x +  mat.m21 * in.y +  mat.m22 * in.z + mat.m23 * in.w;
	out.w	= mat.m30 * in.x +  mat.m31 * in.y +  mat.m32 * in.z + mat.m33 * in.w;
}


template< typename T >
inline void Multiply( Mat4<T>& out, const Mat4<T>& in1, const Mat4<T>& in2 )
{
	out.m00 = in1.m00*in2.m00 + in1.m01*in2.m10 + in1.m02*in2.m20 + in1.m03*in2.m30;
	out.m01 = in1.m00*in2.m01 + in1.m01*in2.m11 + in1.m02*in2.m21 + in1.m03*in2.m31;
	out.m02 = in1.m00*in2.m02 + in1.m01*in2.m12 + in1.m02*in2.m22 + in1.m03*in2.m32;
	out.m03 = in1.m00*in2.m03 + in1.m01*in2.m13 + in1.m02*in2.m23 + in1.m03*in2.m33;

	out.m10 = in1.m10*in2.m00 + in1.m11*in2.m10 + in1.m12*in2.m20 + in1.m13*in2.m30;
	out.m11 = in1.m10*in2.m01 + in1.m11*in2.m11 + in1.m12*in2.m21 + in1.m13*in2.m31;
	out.m12 = in1.m10*in2.m02 + in1.m11*in2.m12 + in1.m12*in2.m22 + in1.m13*in2.m32;
	out.m13 = in1.m10*in2.m03 + in1.m11*in2.m13 + in1.m12*in2.m23 + in1.m13*in2.m33;

	out.m20 = in1.m20*in2.m00 + in1.m21*in2.m10 + in1.m22*in2.m20 + in1.m23*in2.m30;
	out.m21 = in1.m20*in2.m01 + in1.m21*in2.m11 + in1.m22*in2.m21 + in1.m23*in2.m31;
	out.m22 = in1.m20*in2.m02 + in1.m21*in2.m12 + in1.m22*in2.m22 + in1.m23*in2.m32;
	out.m23 = in1.m20*in2.m03 + in1.m21*in2.m13 + in1.m22*in2.m23 + in1.m23*in2.m33;

	out.m30 = in1.m30*in2.m00 + in1.m31*in2.m10 + in1.m32*in2.m20 + in1.m33*in2.m30;
	out.m31 = in1.m30*in2.m01 + in1.m31*in2.m11 + in1.m32*in2.m21 + in1.m33*in2.m31;
	out.m32 = in1.m30*in2.m02 + in1.m31*in2.m12 + in1.m32*in2.m22 + in1.m33*in2.m32;
	out.m33 = in1.m30*in2.m03 + in1.m31*in2.m13 + in1.m32*in2.m23 + in1.m33*in2.m33;
}






template< typename T >
inline void MatScale( Mat4<T>& out, const Vec3<T>& scale )
{
	out.m00 = scale.x;	out.m01 = 0;		out.m02 = 0;		out.m03 = 0;
	out.m10 = 0;		out.m11 = scale.y;	out.m12 = 0;		out.m13 = 0;
	out.m20 = 0;		out.m21 = 0;		out.m22 = scale.z;	out.m23 = 0;
	out.m30 = 0;		out.m31 = 0;		out.m32 = 0;		out.m33 = 1;
}


template< typename T >
inline void MatScale( Mat4<T>& out, T sx, T sy, T sz )
{
	out.m00 = sx;		out.m01 = 0;		out.m02 = 0;		out.m03 = 0;
	out.m10 = 0;		out.m11 = sy;		out.m12 = 0;		out.m13 = 0;
	out.m20 = 0;		out.m21 = 0;		out.m22 = sz;		out.m23 = 0;
	out.m30 = 0;		out.m31 = 0;		out.m32 = 0;		out.m33 = 1;
}


template< typename T >
inline void MatTranslation( Mat4<T>& out, const Vec3<T>& vec )
{
	out.m00 = 1; out.m01 = 0; out.m02 = 0; out.m03 = vec.x;
	out.m10 = 0; out.m11 = 1; out.m12 = 0; out.m13 = vec.y;
	out.m20 = 0; out.m21 = 0; out.m22 = 1; out.m23 = vec.z;
	out.m30 = 0; out.m31 = 0; out.m32 = 0; out.m33 = 1;
}


template< typename T >
inline void MatTranslation( Mat4<T>& out, T vx, T vy, T vz )
{
	out.m00 = 1; out.m01 = 0; out.m02 = 0; out.m03 = vx;
	out.m10 = 0; out.m11 = 1; out.m12 = 0; out.m13 = vy;
	out.m20 = 0; out.m21 = 0; out.m22 = 1; out.m23 = vz;
	out.m30 = 0; out.m31 = 0; out.m32 = 0; out.m33 = 1;
}


template< typename T >
inline void MatRotationX( Mat4<T>& mat, T theta )
{
	mat.m00 = 1;		mat.m01 = 0;			mat.m02 = 0;			mat.m03 = 0;
	mat.m10 = 0;		mat.m11 = cos( theta );	mat.m12 = -sin( theta );	mat.m13 = 0;
	mat.m20 = 0;		mat.m21 = sin( theta );	mat.m22 = cos( theta );	mat.m23 = 0;
	mat.m30 = 0;		mat.m31 = 0;			mat.m32 = 0;			mat.m33 = 1;
}


template< typename T >
inline void MatRotationY( Mat4<T>& mat, T theta )
{
	mat.m00 = cos( theta );	mat.m01 = 0;		mat.m02 = sin( theta );	mat.m03 = 0;
	mat.m10 = 0;			mat.m11 = 1;		mat.m12 = 0;			mat.m13 = 0;
	mat.m20 = -sin( theta );	mat.m21 = 0;		mat.m22 = cos( theta );	mat.m23 = 0;
	mat.m30 = 0;			mat.m31 = 0;		mat.m32 = 0;			mat.m33 = 1;
}


template< typename T >
inline void MatRotationZ( Mat4<T>& mat, T theta )
{
	mat.m00 = cos( theta );	mat.m01 = -sin( theta );	mat.m02 = 0;		mat.m03 = 0;
	mat.m10 = sin( theta );	mat.m11 = cos( theta );	mat.m12 = 0;		mat.m13 = 0;
	mat.m20 = 0;			mat.m21 = 0;			mat.m22 = 1;		mat.m23 = 0;
	mat.m30 = 0;			mat.m31 = 0;			mat.m32 = 0;		mat.m33 = 1;
}


// 座標変換行列を作成する(右手座標系/左手座標系の違いに注意！).行順
// u: horizontal, v: vertical, n: forward
template< typename T >
inline void MatViewGL( Mat4<T>& out, const Vec3<T>& u, const Vec3<T>& v, const Vec3<T>& n, const Vec3<T>& c )
{
	out.m00 = -u.x;		out.m01 = -u.y;		out.m02 = -u.z;		out.m03 = DotProduct( u, c );
	out.m10 = v.x;		out.m11 = v.y;		out.m12 = v.z;		out.m13 = -DotProduct( v, c );
	out.m20 = -n.x;		out.m21 = -n.y;		out.m22 = -n.z;		out.m23 = DotProduct( n, c );
	out.m30 = 0;		out.m31 = 0;		out.m32 = 0;		out.m33 = 1;
}


template< typename T >
inline void MatView( Mat4<T>& out, const Vec3<T>& u, const Vec3<T>& v, const Vec3<T>& n, const Vec3<T>& pos )
{
	out.m00 = u.x;	out.m01 = u.y;	out.m02 = u.z;	out.m03 = DotProduct( u, pos );
	out.m10 = v.x;	out.m11 = v.y;	out.m12 = v.z;	out.m13 = DotProduct( v, pos );
	out.m20 = n.x;	out.m21 = n.y;	out.m22 = n.z;	out.m23 = DotProduct( n, pos );
	out.m30 = 0;	out.m31 = 0;	out.m32 = 0;	out.m33 = 1;
}



// 射影変換行列を作成する(gluPerspectiveと同じ).行順
template< typename T >
inline void MatPerspectiveFov( Mat4<T>& out, T fovy, T aspect, T znear, T zfar )
{
	T	depth = (znear)-( zfar );
	T	f = ( T )1.0 / tan( ( T )0.5*( fovy ) );

	out.m00 = f / ( aspect );		out.m01 = 0;			out.m02 = 0;							out.m03 = 0;
	out.m10 = 0;				out.m11 = f;			out.m12 = 0;							out.m13 = 0;
	out.m20 = 0;				out.m21 = 0;			out.m22 = ( (zfar)+( znear ) ) / depth;		out.m23 = (T)2*( zfar )*( znear ) / depth;
	out.m30 = 0;				out.m31 = 0;			out.m32 = -1;							out.m33 = 0;
}


// 射影変換行列を作成する(glFrustumと同じ).行順
template< typename T >
inline void MatPerspectiveOffCenter( Mat4<T>& out, T left, T right, T bottom, T top, T znear, T zfar )
{
	out.m00 = 2*znear/( right-left );	out.m01 = 0;						out.m02 = ( right+left )/( right-left );	out.m03 = 0;
	out.m10 = 0;					out.m11 = 2*znear/( top-bottom );		out.m12 = ( top+bottom )/( top-bottom );	out.m13 = 0;
	out.m20 = 0;					out.m21 = 0;						out.m22 = -( zfar+znear )/( zfar-znear );	out.m23 = -2*zfar*znear/( zfar-znear ); //znear*zfar/(znear-zfar);
	out.m30 = 0;					out.m31 = 0;						out.m32 = -1;							out.m33 = 0;
}



// 射影変換行列を作成する(glOrthoと等価).
template< typename T >
inline void MatOrtho( Mat4<T>& out, T left, T right, T bottom, T top, T znear, T zfar )
{
	T width		= right -left;
	T height	= top - bottom;
	T depth		= zfar - znear;

	out.m00 = (T)2 / width;		out.m01 = 0;				out.m02 = 0;				out.m03 = -( right + left ) / width;
	out.m10 = 0;				out.m11 = (T)2 / height;	out.m12 = 0;				out.m13 = -( top + bottom ) / height;
	out.m20 = 0;				out.m21 = 0;				out.m22 = (T)-2 / depth;	out.m23 = -( zfar + znear ) / depth;
	out.m30 = 0;				out.m31 = 0;				out.m32 = 0;				out.m33 = 1;
}




//##############################################################################//
//									4x4 Matrix									//
//##############################################################################//







inline void InitQuat( Vec4f& quat, float angle, float x, float y, float z )
{
	const float theta = angle * 0.5f;
	const float sine_theta = sin( theta );
	quat.w	= cos( theta );
	quat.x	= x * sine_theta;
	quat.y	= y * sine_theta;
	quat.z	= z * sine_theta;
}


inline float length( Vec4f quat )
{
	return sqrt( quat.x*quat.x + quat.y*quat.y +quat.z*quat.z + quat.w*quat.w );
}


inline Vec4f normalize( Vec4f quat )
{
	float L_inv = 1.0f / length( quat );

	quat.x *= L_inv;
	quat.y *= L_inv;
	quat.z *= L_inv;
	quat.w *= L_inv;

	return quat;
}


inline Vec4f conjugate( Vec4f quat )
{
	quat.x = -quat.x;
	quat.y = -quat.y;
	quat.z = -quat.z;
	return quat;
}


// クォータニオン合成蓄積誤差除去
inline void rm_cal_err_qt( Vec4f quat )
{

	float s = length( quat );
	if( s <= 0.0f ) return;
	quat.x /= s;
	quat.y /= s;
	quat.z /= s;
	quat.w /= s;
}



inline Vec4f mult( Vec4f A, Vec4f B )
{
	Vec4f C;

	C.x = A.w*B.x + A.x*B.w + A.y*B.z - A.z*B.y;
	C.y = A.w*B.y - A.x*B.z + A.y*B.w + A.z*B.x;
	C.z = A.w*B.z + A.x*B.y - A.y*B.x + A.z*B.w;
	C.w = A.w*B.w - A.x*B.x - A.y*B.y - A.z*B.z;


	return C;
}


// *=も大丈夫なバージョン
//inline Vec4f mult( Vec4f lpP, Vec4f lpQ )
//{
//	Vec4f lpR;
//
//    float pw, px, py, pz;
//    float qw, qx, qy, qz;
//
//    pw = lpP.w; px = lpP.x; py = lpP.y; pz = lpP.z;
//    qw = lpQ.w; qx = lpQ.x; qy = lpQ.y; qz = lpQ.z;
//
//    lpR.w = pw * qw - px * qx - py * qy - pz * qz;
//    lpR.x = pw * qx + px * qw + py * qz - pz * qy;
//    lpR.y = pw * qy - px * qz + py * qw + pz * qx;
//    lpR.z = pw * qz + px * qy - py * qx + pz * qw;
//
//	return lpR;
//}












template< typename T >
union Quaternion
{
	struct { T w, x, y, z; };
};



template< typename T >
inline void InitQuat( Quaternion<T>& quat, T angle, T x, T y, T z )
{
	const double theta	= (double)angle * 0.5;
	const T sine_theta	= (T)sin( theta );

	Vec3f axis			={ x, y, z };
	Normalize( axis );

	quat.w	= (T)cos( theta );
	quat.x	= axis.x * sine_theta;
	quat.y	= axis.y * sine_theta;
	quat.z	= axis.z * sine_theta;
}


template< typename T >
inline void InitQuat( Quaternion<T>& quat, T angle, const Vec3<T>& axis )
{
	const double theta = (double)angle * 0.5;
	const T sine_theta = (T)sin( theta );
	quat.w	= (T)cos( theta );
	quat.x	= axis.x * sine_theta;
	quat.y	= axis.y * sine_theta;
	quat.z	= axis.z * sine_theta;
}


template< typename T >
inline void QuatIdentity( Quaternion<T>& quat )
{
	quat.w	= (T)1;
	quat.x	= 0;
	quat.y	= 0;
	quat.z	= 0;
}


template< typename T >
inline void QuatConjugate( Quaternion<T>& quat )
{
	quat.x = -quat.x;
	quat.y = -quat.y;
	quat.z = -quat.z;
}


template< typename T >
inline void QuatConjugate( Quaternion<T>& out, const Quaternion<T>& quat )
{
	out.w	= quat.w;
	out.x	= -quat.x;
	out.y	= -quat.y;
	out.z	= -quat.z;
}


template< typename T >
inline T Length( const Quaternion<T>& quat )
{
	return sqrt( quat.w*quat.w + quat.x*quat.x + quat.y*quat.y +quat.z*quat.z );
}


template < typename T >
inline void Normalize( Quaternion<T>& quat )
{
	float L_inv = ( T )1.0 / Length( quat );

	quat.w *= L_inv;
	quat.x *= L_inv;
	quat.y *= L_inv;
	quat.z *= L_inv;
}


// クォータニオン合成蓄積誤差除去
template< typename T >
inline void rm_cal_err_qt( Quaternion<T>& quat )
{
	T s = Length( quat );
	if( s <= (T)0 ) return;

	quat.w /= s;
	quat.x /= s;
	quat.y /= s;
	quat.z /= s;
}


// Quaternionの乗算. Aの右にBをかける.B回転が最初に作用して、次にA回転が作用する
template< typename T >
inline void Multiply( Quaternion<T>& C, const Quaternion<T>& A, const Quaternion<T>& B )
{
	C.x = A.w*B.x + A.x*B.w + A.y*B.z - A.z*B.y;
	C.y = A.w*B.y - A.x*B.z + A.y*B.w + A.z*B.x;
	C.z = A.w*B.z + A.x*B.y - A.y*B.x + A.z*B.w;
	C.w = A.w*B.w - A.x*B.x - A.y*B.y - A.z*B.z;
}

// Quaternionの乗算. lpPの右にlpQをかける.lpQ回転が最初に作用して、次にlpP回転が作用する
template< typename T >
inline void Multiply_( Quaternion<T>& lpR, const Quaternion<T>& lpP, const Quaternion<T>& lpQ )
{
	T pw, px, py, pz;
	T qw, qx, qy, qz;

	pw = lpP.w; px = lpP.x; py = lpP.y; pz = lpP.z;
	qw = lpQ.w; qx = lpQ.x; qy = lpQ.y; qz = lpQ.z;

	lpR.w = pw * qw - px * qx - py * qy - pz * qz;
	lpR.x = pw * qx + px * qw + py * qz - pz * qy;
	lpR.y = pw * qy - px * qz + py * qw + pz * qx;
	lpR.z = pw * qz + px * qy - py * qx + pz * qw;
}



template< typename T >
inline void Rotate( Vec3<T>& inout, const Quaternion<T>& quat )
{
	Quaternion<T>	q1, quatConjugate, result;
	Quaternion<T>	quat_inout ={ 0, inout.x, inout.y, inout.z };

	QuatConjugate( quatConjugate, quat );

	// quatConjugate * quat_inout * quat( TODO: 注意!. 最後にかけたものが最初に有効になる。quatConjugate * inout * quatは後ろからかける)
	Multiply( q1, quat, quat_inout );
	Multiply( result, q1, quatConjugate );

	inout.x = result.x;
	inout.y = result.y;
	inout.z = result.z;
}


// 姿勢変化をクォータニオンに変換する.Z軸方向がforwardに、Y軸方向がupにアライメントされる
template< typename T >
inline void QuaternionLookAt( Quaternion<T>& out, const Vec3<T>& forward, const Vec3<T>& up, const Vec3<T>& basis_forward, const Vec3<T>& basis_up )
{
	//const Vec3<T> Y_AXIS = { 0.0, 1.0, 0.0 };
	//const Vec3<T> Z_AXIS = { 0.0, 0.0, 1.0 };

	//=============== Z軸からforwardへの回転のクォータニオンを計算する ================//
	// Z_AXISとforwardの外積を計算して、回転軸を取得する
	Vec3<T> rot_axis_forward;
	CrossProduct( rot_axis_forward, forward, basis_forward/*Z_AXIS*/ );
	Normalize( rot_axis_forward );

	// Z_AXISとforwardの内積を計算して、回転角を取得する( TODO: 注意!. 外積の回転方向は右ねじの法則。クォータニオンは逆)
	T angle_forward	= -acos( DotProduct( basis_forward/*Z_AXIS*/, forward ) );

	// forwardのクォータニオンを作成する
	Quaternion<T> quat_forward;
	InitQuat( quat_forward, angle_forward, rot_axis_forward );


	//=============== forward分のクォータニオンをY_AXIS軸に作用させる =================//
	Vec3<T> up_before = basis_up;//Y_AXIS;
	Rotate( up_before, quat_forward );

	//================ up_beforeからupへのクォータニオンを計算する ===================//
	// 回転軸を計算する
	Vec3<T> rot_axis_up;
	CrossProduct( rot_axis_up, up, up_before );
	Normalize( rot_axis_up );

	// up_beforeとupの内積を計算して、回転角を取得する( TODO: 注意!. 外積の回転方向は右ねじの法則。クォータニオンは逆)
	T angle_up	= -acos( DotProduct( up_before, up ) );

	// upのクォータニオンを作成する
	Quaternion<T> quat_up;
	InitQuat( quat_up, angle_up, rot_axis_up );


	//========================== クォータニオンを合成する ============================//
	Multiply( out, quat_up, quat_forward );
	//out = quat_forward;
}




// クォータニオンから回転行列への変換
template< typename T >
inline void Quat2Mat( Mat4<T>& out, const Quaternion<T>& quat )
{
	T x2_2 = quat.x * quat.x * 2;
	T xy_2 = quat.x * quat.y * 2;
	T xz_2 = quat.x * quat.z * 2;
	T wx_2 = quat.x * quat.w * 2;

	T y2_2 = quat.y * quat.y * 2;
	T yz_2 = quat.y * quat.z * 2;
	T wy_2 = quat.y * quat.w * 2;

	T z2_2 = quat.z * quat.z * 2;
	T zw_2 = quat.z * quat.w * 2;

	out.m00	= 1-y2_2-z2_2;	out.m01 = xy_2-zw_2;	out.m02 = xz_2+wy_2;	out.m03 = 0;
	out.m10 = xy_2 + zw_2;	out.m11 = 1-x2_2-z2_2;	out.m12 = yz_2-wx_2;	out.m13 = 0;
	out.m20 = xz_2 - wy_2;	out.m21 = yz_2 + wx_2;	out.m22 = 1-x2_2-y2_2;	out.m23 = 0;
	out.m30 = 0;			out.m31 = 0;			out.m32 = 0;			out.m33 = 1;
}




#endif // GRAPHICS_MATH_H