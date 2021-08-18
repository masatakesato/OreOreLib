#ifndef INDEX_ND_H
#define	INDEX_ND_H


#include	<oreore/common/Utility.h>
#include	<oreore/MathLib.h>


namespace OreOreLib
{


//##################################################################################//
//						配列データに任意次元数でアクセスするクラス					//
//##################################################################################//

class IIndexSpace
{

};




//##################################################################################//
// N-D Index
//##################################################################################//

class IndexND : IIndexSpace
{
public:
	
	IndexND();
	IndexND( uint32 numdim, uint32 dim[] );
	~IndexND();
	IndexND( const IndexND& obj );
	
	void Init( uint32 numdim, uint32 dim[] );
	void Release();

	// 1次元の通し番号からN次元インデックスへの変換
	inline void Id2CellIdx( int Output[], int id ) const
	{
		for(int i=m_numDim-1; i>=0; --i)
		{
			Output[i]	= id / m_DimCoeff[i];
			id			-= m_DimCoeff[i] * Output[i];
		}
	}


	inline uint32 ArrayIdx1D( const uint32 idx_nd[] ) const
	{
		uint32 id	= 0;
		for( uint32 i=0; i<m_numDim; ++i )	id	+= idx_nd[i] * m_DimCoeff[i];
		return id;
	}


private:

	uint32	m_numDim;	// インデックス空間の次元数
	uint32	*m_Dim;		// 次元毎の要素数
	uint32	*m_DimCoeff;// 次元毎のオフセット {1, DimX, DimX*DimY, DimX*DimY*DimZ, DimX*DimY*DimZ*DimW...}

};




//##################################################################################//
// 2D Index
//##################################################################################//

class Index2D : IIndexSpace
{
public:

	Index2D();
	Index2D( uint32 dimx, uint32 dimy );
	Index2D( uint32 dim[] );
	~Index2D(){};
	Index2D( const Index2D& obj );

	void Init( uint32 dimx, uint32 dimy );
	//void Release();

	// convert 1d-index to 2d-index
	inline void ArrayIdx2D( uint32 Output[], uint32 id ) const
	{
		Output[1]	= id / m_Dim.x;
		Output[0]	= id % m_Dim.x;
	}


	inline uint32 ArrayIdx1D( uint32 x, uint32 y ) const
	{
		return y * m_Dim.x + x;
	}


private:

	uint32	m_numDim;
	Vec2ui	m_Dim;
	//uint32	m_DimCoeff[4];
};






//##################################################################################//
// 3D Index
//##################################################################################//

class Index3D : IIndexSpace
{
public:

	Index3D();
	Index3D( uint32 dimx, uint32 dimy, uint32 dimz );
	Index3D( uint32 dim[] );
	~Index3D();
	Index3D( const Index3D& obj );

	void Init( uint32 dimx, uint32 dimy, uint32 dimz );
	void Release();

	// convert 1d-index to 3d-index
	inline void ArrayIdx3D( uint32 Output[], uint32 id ) const
	{
		Output[2]	= id / m_DimCoeff[2];
		Output[1]	= ( id % m_DimCoeff[2] ) / m_DimCoeff[1];
		Output[0]	= ( id % m_DimCoeff[2] ) % m_DimCoeff[1];
	}


	inline void ArrayIdx3D( uint32 &x, uint32 &y, uint32 &z, uint32 id ) const
	{
		z	= id / m_DimCoeff[2];
		y	= ( id % m_DimCoeff[2] ) / m_DimCoeff[1];
		x	= ( id % m_DimCoeff[2] ) % m_DimCoeff[1];
	}


	inline uint32 ArrayIdx1D( uint32 x, uint32 y, uint32 z ) const
	{
		return z * m_DimCoeff[2] + y * m_DimCoeff[1] + x;
	}
	

private:

	uint32	m_numDim;
	Vec3ui	m_Dim;
	uint32	m_DimCoeff[4];// 次元毎のオフセット {1, DimX, DimX*DimY, DimX*DimY*DimZ, DimX*DimY*DimZ*DimW...}

};




//##################################################################################//
// 4D Index
//##################################################################################//

class Index4D : IIndexSpace
{
public:

	Index4D();
	Index4D( uint32 dimx, uint32 dimy, uint32 dimz, uint32 dimw );
	Index4D( uint32 dim[] );
	~Index4D();
	Index4D( const Index4D& obj );

	void Init( uint32 dimx, uint32 dimy, uint32 dimz, uint32 dimw );
	void Release();

	// convert 1d-index to 4d-index
	inline void ArrayIdx4D( uint32 Output[], uint32 id ) const
	{
		Output[3]	= ( id / m_DimCoeff[3] );
		Output[2]	= ( id % m_DimCoeff[3] ) / m_DimCoeff[2];
		Output[1]	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] / m_DimCoeff[1];
		Output[0]	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] % m_DimCoeff[1];
	}


	inline void ArrayIdx4D( uint32 &x, uint32 &y, uint32 &z, uint32 &w, uint32 id ) const
	{
		w	= ( id / m_DimCoeff[3] );
		z	= ( id % m_DimCoeff[3] ) / m_DimCoeff[2];
		y	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] / m_DimCoeff[1];
		x	= ( id % m_DimCoeff[3] ) % m_DimCoeff[2] % m_DimCoeff[1];
	}


	inline uint32 ArrayIdx1D( uint32 x, uint32 y, uint32 z, uint32 w ) const
	{
		return w * m_DimCoeff[3] + z * m_DimCoeff[2] + y * m_DimCoeff[1] + x;
	}
	

private:

	uint32	m_numDim;
	Vec4ui	m_Dim;
	uint32	m_DimCoeff[5];// 次元毎のオフセット {1, DimX, DimX*DimY, DimX*DimY*DimZ, DimX*DimY*DimZ*DimW...}

};






}// end of namespace



#endif // !ARRAY_INDEX_H
