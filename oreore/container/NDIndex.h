#ifndef ND_INDEX_H
#define	ND_INDEX_H

#include	"../common/TString.h"
#include	"../common/Utility.h"
#include	"../meta/TypeTraits.h"
#include	"../mathlib/MathLib.h"



namespace OreOreLib
{

	// https://stackoverflow.com/questions/32921192/c-variadic-template-limit-number-of-args

	template < uint64 N >
	class NDIndex
	{
	public:
		
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDIndex( Args ... args )
			: m_Shape{ uint64(args)... }
		{

			for( int i=0; i<N; ++i )
			{
				m_Offsets[i] = m_Shape[i];
				if( i>0 )  m_Offsets[i] *= m_Offsets[i-1];
			}

		}


		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		Init( Args ... args )
		{
			auto p = m_Shape;
			for( const auto& val : { args... } )
				*(p++) = val;

			m_Offsets[0] = m_Shape[0];
			for( int i=1; i<N; ++i )
				m_Offsets[i] = m_Shape[i] * m_Offsets[i-1];

			//m_DimCoeff[0]	= m_Shape[0];
			//m_DimCoeff[1]	= m_Shape[0] * m_Shape[1];
			//m_DimCoeff[2]	= m_Shape[0] * m_Shape[1] * m_Shape[2];
			//m_DimCoeff[3]	= m_Shape[0] * m_Shape[1] * m_Shape[2] * m_Shape[3];
			//...
		}


		// convert 1d-index to Nd-index
		void ArrayIdx4D( uint64 indexND[], uint64 id ) const
		{
			indexND[N-1] = id;
			for( int i=N-2; i>=0; --i )
				indexND[i] = indexND[i+1] % m_Offsets[i];

			for( int i=N-1; i>=1; --i )
				indexND[i] /= m_Offsets[i-1];

			//indexND[3]	= ( id / m_Offsets[2] );
			//indexND[2]	= ( id % m_Offsets[2] ) / m_Offsets[1];
			//indexND[1]	= ( id % m_Offsets[2] ) % m_Offsets[1] / m_Offsets[0];
			//indexND[0]	= ( id % m_Offsets[2] ) % m_Offsets[1] % m_Offsets[0];
		}


	//	inline uint64 ArrayIdx1D( uint64 x, uint64 y, uint64 z, uint32 w ) const
	//	{
	//		return w * m_Offsets[3] + z * m_Offsets[2] + y * m_Offsets[1] + x;
	//	}



		void Disiplay()
		{
			tcout << typeid(*this).name() << _T(":\n");
			tcout << _T("  Shape = [");
			for( int i=0; i<N; ++i )
				tcout << m_Shape[i] << ( i==N-1 ? _T("];\n") : _T(", ") );
		}





	private:

		uint64	m_Shape[ N ];
		uint64	m_Offsets[ N ];

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



#endif // !ND_INDEX_H