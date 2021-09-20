﻿#ifndef ND_SHAPE_H
#define	ND_SHAPE_H

#include	"../common/TString.h"
#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"



namespace OreOreLib
{

	// https://stackoverflow.com/questions/32921192/c-variadic-template-limit-number-of-args

	template < uint64 N >
	class NDShape
	{
	public:

		// Default constructor
		NDShape()
			: m_Shape{ 0 } 
			, m_Strides{ 0 } 
		{
		
		}


		// Constructor(variadic tempaltes)
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDShape( const Args& ... args )// x, y, z, w...
			: m_Shape{ uint64(args)... }
		{
			InitStrides();
		}


		// Constructor(initializer list)
		template < typename T, std::enable_if_t< std::is_convertible<uint64, T>::value >* = nullptr >
		NDShape( std::initializer_list<T> indexND )// x, y, z, w...
		{
			Init( indexND );
		}


		// Destructor
		~NDShape()
		{
			Release();
		}


		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		Init( const Args& ... args )
		{
			auto p = m_Shape;
			for( const auto& val : { args... } )
				*(p++) = val;

			InitStrides();
		}


		template < typename T >
		std::enable_if_t< std::is_convertible<uint64, T>::value, void >
		Init( std::initializer_list<T> indexND )
		{
			auto p = m_Shape;
			for( const auto& val : indexND )
				*(p++) = val;

			InitStrides();
		}


		void Release()
		{
			for( int i=1; i<N; ++i )
				m_Shape[i] = m_Strides[i] = 0;
		}



		//=============== 1D to ND index conversion ===============//
		// indexND[3]	= ( id / m_Strides[2] );
		// indexND[2]	= ( id % m_Strides[2] ) / m_Strides[1];
		// indexND[1]	= ( id % m_Strides[2] ) % m_Strides[1] / m_Strides[0];
		// indexND[0]	= ( id % m_Strides[2] ) % m_Strides[1] % m_Strides[0];
		// ...

		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, void >
		ToND( uint64 indexd1D, T indexND[] ) const
		{
			indexND[N-1] = (T)indexd1D;
			for( int i=N-2; i>=0; --i )
				indexND[i] = indexND[i+1] % (T)m_Strides[i];

			for( int i=N-1; i>=1; --i )
				indexND[i] /= (T)m_Strides[i-1];
		}



		//　Single dim only
		uint64 ToND( uint64 indexd1D, int dim ) const
		{
			uint64 index = indexd1D;
			for( int i=N-2; i>=dim; --i )
				index = index % m_Strides[i];

			if( dim > 0 )
				index /= m_Strides[dim-1];

			return index;
		}



		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		ToND( uint64 index1D, Args& ... args ) const
		{
			using T = typename TypeTraits::first_type<Args...>::type;

			auto indexND = { &args... };
			auto head = indexND.begin();
			auto iter =(T**)indexND.end();
			iter--;
			**iter = (T)index1D;
			iter--;
			
			// (1)
			for( ; iter>=indexND.begin(); --iter )
			{
				**iter = (T)123;
				//tcout << *iter << tendl;
				//indexND[i] = indexND[i+1] % (T)m_Strides[i];
			}
			

/*
			(1)
			indexND[N-1] = (T)indexd1D;
			for( int i=N-2; i>=0; --i )
				indexND[i] = indexND[i+1] % (T)m_Strides[i];

			(2)
			for( int i=N-1; i>=1; --i )
				indexND[i] /= (T)m_Strides[i-1];
*/
		}




		//============== ND to 1D index conversion ===============//
		// indexND[0] +				// x
		// indexND[1] * m_Strides[0] +	// y
		// indexND[2] * m_Strides[1] +	// z
		// indexND[3] * m_Strides[2];	// w
		// ...

		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, int64 >
		To1D( const Args ... args ) const// x, y, z, w...
		{
			auto indexND = { args... };
			auto itr = std::begin( indexND );

			uint64 index = *(itr++);
			auto offset = m_Strides;

			while( itr !=std::end( indexND ) )
				index += *(itr++) * *(offset++);

			return index;
		}


		template < typename T >
		std::enable_if_t< std::is_convertible<uint64, T>::value, int64 >
		To1D( std::initializer_list<T> indexND ) const// x, y, z, w...
		{
			auto itr = std::begin( indexND );

			uint64 index = (uint64)*(itr++);
			auto offset = m_Strides;

			while( itr !=std::end( indexND ) )
				index += (uint64)*(itr++) * *(offset++);

			return index;
		}



		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, int64 >
		To1D( const T (&indexND)[N] ) const
		{
			int64 index = indexND[0];
			for( int i=1; i<N; ++i )	index += indexND[i] * m_Strides[i-1];
			return index;

			//indexND[0] +					// x
			//indexND[1] * m_Strides[0] +	// y
			//indexND[2] * m_Strides[1] +	// z
			//indexND[3] * m_Strides[2];	// w
		}


		template < typename T >
		std::enable_if_t< std::is_convertible_v<T, uint64>, int64 >
		From3DTo1D( const T& x, const T& y, const T& z ) const
		{
			return z * m_Strides[1] + y * m_Strides[0] + x;
		}




		uint64 NumDims() const
		{
			return N;
		}


		uint64 Dim( uint64 i ) const
		{
			return i<N ? m_Shape[ i ] : 0;
		}


		uint64 Size() const
		{
			return m_Strides[ N-1 ];
		}


		void Disiplay()
		{
			tcout << typeid(*this).name() << _T(":\n");
			tcout << _T("  Shape = [");
			for( int i=0; i<N; ++i )
				tcout << m_Shape[i] << ( i==N-1 ? _T("];\n") : _T(", ") );
		}



	private:

		uint64	m_Shape[ N ];
		uint64	m_Strides[ N ];// strides for multidimensional element access.


		const void InitStrides()
		{
			m_Strides[0] = m_Shape[0];
			for( int i=1; i<N; ++i )	m_Strides[i] = m_Shape[i] * m_Strides[i-1];

			//m_Strides[0]	= m_Shape[0];
			//m_Strides[1]	= m_Shape[0] * m_Shape[1];
			//m_Strides[2]	= m_Shape[0] * m_Shape[1] * m_Shape[2];
			//m_Strides[3]	= m_Shape[0] * m_Shape[1] * m_Shape[2] * m_Shape[3];
			//...
		}


	};


















	//class IIndexSpace
	//{

	//};




	////##################################################################################//
	//// N-D Index
	////##################################################################################//

	//class IndexND : IIndexSpace
	//{
	//public:

	//	IndexND();
	//	IndexND( uint32 numdim, uint32 dim[] );
	//	~IndexND();
	//	IndexND( const IndexND& obj );

	//	void Init( uint32 numdim, uint32 dim[] );
	//	void Release();

	//	// 1次元の通し番号からN次元インデックスへの変換
	//	inline void Id2CellIdx( int Output[], int id ) const
	//	{
	//		for( int i=m_numDim-1; i>=0; --i )
	//		{
	//			Output[i]	= id / m_DimCoeff[i];
	//			id			-= m_DimCoeff[i] * Output[i];
	//		}
	//	}


	//	inline uint32 ArrayIdx1D( const uint32 idx_nd[] ) const
	//	{
	//		uint32 id	= 0;
	//		for( uint32 i=0; i<m_numDim; ++i )	id	+= idx_nd[i] * m_DimCoeff[i];
	//		return id;
	//	}


	//private:

	//	uint32	m_numDim;	// インデックス空間の次元数
	//	uint32	*m_Dim;		// 次元毎の要素数
	//	uint32	*m_DimCoeff;// 次元毎のオフセット {1, DimX, DimX*DimY, DimX*DimY*DimZ, DimX*DimY*DimZ*DimW...}

	//};




	////##################################################################################//
	//// 2D Index
	////##################################################################################//

	//class Index2D : IIndexSpace
	//{
	//public:

	//	Index2D();
	//	Index2D( uint32 dimx, uint32 dimy );
	//	Index2D( uint32 dim[] );
	//	~Index2D(){};
	//	Index2D( const Index2D& obj );

	//	void Init( uint32 dimx, uint32 dimy );
	//	//void Release();

	//	// convert 1d-index to 2d-index
	//	inline void ArrayIdx2D( uint32 Output[], uint32 id ) const
	//	{
	//		Output[1]	= id / m_Dim.x;
	//		Output[0]	= id % m_Dim.x;
	//	}


	//	inline uint32 ArrayIdx1D( uint32 x, uint32 y ) const
	//	{
	//		return y * m_Dim.x + x;
	//	}


	//private:

	//	uint32	m_numDim;
	//	Vec2ui	m_Dim;
	//	//uint32	m_DimCoeff[4];
	//};






	////##################################################################################//
	//// 3D Index
	////##################################################################################//

	//class Index3D : IIndexSpace
	//{
	//public:

	//	Index3D();
	//	Index3D( uint32 dimx, uint32 dimy, uint32 dimz );
	//	Index3D( uint32 dim[] );
	//	~Index3D();
	//	Index3D( const Index3D& obj );

	//	void Init( uint32 dimx, uint32 dimy, uint32 dimz );
	//	void Release();

	//	// convert 1d-index to 3d-index
	//	inline void ArrayIdx3D( uint32 Output[], uint32 id ) const
	//	{
	//		Output[2]	= id / m_DimCoeff[2];
	//		Output[1]	= ( id % m_DimCoeff[2] ) / m_DimCoeff[1];
	//		Output[0]	= ( id % m_DimCoeff[2] ) % m_DimCoeff[1];
	//	}


	//	inline void ArrayIdx3D( uint32 &x, uint32 &y, uint32 &z, uint32 id ) const
	//	{
	//		z	= id / m_DimCoeff[2];
	//		y	= ( id % m_DimCoeff[2] ) / m_DimCoeff[1];
	//		x	= ( id % m_DimCoeff[2] ) % m_DimCoeff[1];
	//	}


	//	inline uint32 ArrayIdx1D( uint32 x, uint32 y, uint32 z ) const
	//	{
	//		return z * m_DimCoeff[2] + y * m_DimCoeff[1] + x;
	//	}


	//private:

	//	uint32	m_numDim;
	//	Vec3ui	m_Dim;
	//	uint32	m_DimCoeff[4];// 次元毎のオフセット {1, DimX, DimX*DimY, DimX*DimY*DimZ, DimX*DimY*DimZ*DimW...}

	//};




	////##################################################################################//
	//// 4D Index
	////##################################################################################//

	//class Index4D : IIndexSpace
	//{
	//public:

	//	Index4D();
	//	Index4D( uint32 dimx, uint32 dimy, uint32 dimz, uint32 dimw );
	//	Index4D( uint32 dim[] );
	//	~Index4D();
	//	Index4D( const Index4D& obj );

	//	void Init( uint32 dimx, uint32 dimy, uint32 dimz, uint32 dimw );
	//	void Release();

	//	// convert 1d-index to 4d-index
	//	inline void ArrayIdx4D( uint32 Output[], uint32 id ) const
	//	{
	//		Output[3]	= ( id / m_DimCoeff[3] );
	//		Output[2]	= ( id % m_DimCoeff[3] ) / m_DimCoeff[2];
	//		Output[1]	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] / m_DimCoeff[1];
	//		Output[0]	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] % m_DimCoeff[1];
	//	}


	//	inline void ArrayIdx4D( uint32 &x, uint32 &y, uint32 &z, uint32 &w, uint32 id ) const
	//	{
	//		w	= ( id / m_DimCoeff[3] );
	//		z	= ( id % m_DimCoeff[3] ) / m_DimCoeff[2];
	//		y	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] / m_DimCoeff[1];
	//		x	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] % m_DimCoeff[1];
	//	}


	//	inline uint32 ArrayIdx1D( uint32 x, uint32 y, uint32 z, uint32 w ) const
	//	{
	//		return w * m_DimCoeff[3] + z * m_DimCoeff[2] + y * m_DimCoeff[1] + x;
	//	}


	//private:

	//	uint32	m_numDim;
	//	Vec4ui	m_Dim;
	//	uint32	m_DimCoeff[5];// 次元毎のオフセット {1, DimX, DimX*DimY, DimX*DimY*DimZ, DimX*DimY*DimZ*DimW...}

	//};










}// end of namespace



#endif // !ND_SHAPE_H