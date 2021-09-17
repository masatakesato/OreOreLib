//#include	"IndexND.h"
//
//#include	<oreore/common/TString.h>
//#include	<oreore/common/Utility.h>
//
//
//namespace OreOreLib
//{
//
//
////##################################################################################//
//// N-D Array Index
////##################################################################################//
//
//IndexND::IndexND()
//{
//	m_numDim	= 0;
//	m_Dim		= NULL;
//	m_DimCoeff	= NULL;
//}
//
//
//
//IndexND::IndexND( uint32 numdim, uint32 dim[] )
//{
//	Init( numdim, dim );
//}
//
//
//
//IndexND::~IndexND()
//{
//	Release();
//}
//
//
//
//IndexND::IndexND( const IndexND& obj )
//{
//	m_numDim	= obj.m_numDim;
//	m_Dim		= new uint32[ m_numDim ];
//	m_DimCoeff	= new uint32[ m_numDim + 1 ];
//
//	memcpy( m_Dim,		obj.m_Dim,		sizeof(uint32)*m_numDim );
//	memcpy( m_DimCoeff,	obj.m_DimCoeff,	sizeof(uint32)*(m_numDim+1) );
//}
//
//
//
//void IndexND::Init( uint32 numdim, uint32 dim[] )
//{
//	uint32 i;
//
//	Release();
//
//	m_numDim	= numdim;
//	m_Dim		= new uint32[ m_numDim ];
//	m_DimCoeff	= new uint32[ m_numDim + 1 ];
//
//	for( i=0; i<m_numDim; ++i )
//		m_Dim[i] = dim[i];
//
//	// インデックス計算時に使う、次元毎のオフセット
//	m_DimCoeff[0]	= 1;
//	for( i=1; i<m_numDim+1; ++i )
//		m_DimCoeff[i] = m_Dim[i-1] * m_DimCoeff[i-1];
//}
//
//
//
//void IndexND::Release()
//{
//	SafeDeleteArray( m_Dim );
//	SafeDeleteArray( m_DimCoeff );
//	m_numDim	= 0;
//}
//
//
//
//
//
////##################################################################################//
//// 2D Array Index
////##################################################################################//
//
//Index2D::Index2D()
//{
//	m_numDim	= 2;
//	InitZero( m_Dim );
//	//memset( m_DimCoeff, 0, sizeof(uint32)*4 );
//}
//
//
//
//Index2D::Index2D( uint32 dimx, uint32 dimy )
//{
//	Init( dimx, dimy );
//}
//
//
//
//Index2D::Index2D( uint32 dim[] )
//{
//	Init( dim[0], dim[1] );
//}
//
//
//
////Index2D::~Index2D()
////{
////	Release();
////}
//
//
//
//Index2D::Index2D( const Index2D& obj )
//{
//	m_numDim	= obj.m_numDim;
//	m_Dim		= obj.m_Dim;
//	
//	//memcpy( m_DimCoeff,	obj.m_DimCoeff,	sizeof(uint32) * 4 );
//}
//
//
//
//void Index2D::Init( uint32 dimx, uint32 dimy )
//{
//	//int i;
//
//	//Release();
//
//	m_numDim	= 2;
//	//m_Dim		= new uint32[ m_numDim ];
//	//m_DimCoeff	= new uint32[ m_numDim + 1 ];
//
//	//for( i=0; i<m_numDim; ++i )
//	//	m_Dim[i] = dim[i];
//	InitVec( m_Dim, dimx, dimy );
//
//
//	// インデックス計算時に使う、次元毎のオフセット
////	m_DimCoeff[0]	= 1;
////	m_DimCoeff[1]	= dimx;
////	m_DimCoeff[2]	= dimx * dimy;
////	m_DimCoeff[3]	= dimx * dimy * dimz;
//
//
//	//for( i=1; i<m_numDim+1; ++i )
//	//	m_DimCoeff[i] = m_Dim[i-1] * m_DimCoeff[i-1];
//}
//
//
//
////void Index2D::Release()
////{
////	//SafeDeleteArray( m_Dim );
////	//SafeDeleteArray( m_DimCoeff );
////	//m_numDim	= 0;
////}
//
//
//
//
////##################################################################################//
//// 3D Array Index
////##################################################################################//
//
//Index3D::Index3D()
//{
//	m_numDim	= 3;
//	InitZero( m_Dim );
//	memset( m_DimCoeff, 0, sizeof(uint32)*4 );
//}
//
//
//
//Index3D::Index3D( uint32 dimx, uint32 dimy, uint32 dimz )
//{
//	Init( dimx, dimy, dimz );
//}
//
//
//
//Index3D::Index3D( uint32 dim[] )
//{
//	Init( dim[0], dim[1], dim[2] );
//}
//
//
//
//Index3D::~Index3D()
//{
//	Release();
//}
//
//
//
//Index3D::Index3D( const Index3D& obj )
//{
//	//m_numDim	= obj.m_numDim;
//	m_Dim		= obj.m_Dim;
//	
//	memcpy( m_DimCoeff,	obj.m_DimCoeff,	sizeof(uint32) * 4 );
//}
//
//
//
//void Index3D::Init( uint32 dimx, uint32 dimy, uint32 dimz )
//{
//	//int i;
//
//	//Release();
//
//	m_numDim	= 3;
//	//m_Dim		= new uint32[ m_numDim ];
//	//m_DimCoeff	= new uint32[ m_numDim + 1 ];
//
//	//for( i=0; i<m_numDim; ++i )
//	//	m_Dim[i] = dim[i];
//	InitVec( m_Dim, dimx, dimy, dimz );
//
//
//	// インデックス計算時に使う、次元毎のオフセット
//	m_DimCoeff[0]	= 1;
//	m_DimCoeff[1]	= dimx;
//	m_DimCoeff[2]	= dimx * dimy;
//	m_DimCoeff[3]	= dimx * dimy * dimz;
//
//
//	//for( i=1; i<m_numDim+1; ++i )
//	//	m_DimCoeff[i] = m_Dim[i-1] * m_DimCoeff[i-1];
//}
//
//
//
//void Index3D::Release()
//{
//	//SafeDeleteArray( m_Dim );
//	//SafeDeleteArray( m_DimCoeff );
//	//m_numDim	= 0;
//}
//
//
//
//
//
////##################################################################################//
//// 4D Array Index
////##################################################################################//
//
//Index4D::Index4D()
//{
//	m_numDim	= 4;
//	InitZero( m_Dim );
//	memset( m_DimCoeff, 0, sizeof(uint32)*4 );
//}
//
//
//
//Index4D::Index4D( uint32 dimx, uint32 dimy, uint32 dimz, uint32 dimw )
//{
//	Init( dimx, dimy, dimz, dimw );
//}
//
//
//
//Index4D::Index4D( uint32 dim[] )
//{
//	Init( dim[0], dim[1], dim[2], dim[3] );
//}
//
//
//
//Index4D::~Index4D()
//{
//	Release();
//}
//
//
//
//Index4D::Index4D( const Index4D& obj )
//{
//	m_numDim	= obj.m_numDim;
//	m_Dim		= obj.m_Dim;
//	
//	memcpy( m_DimCoeff,	obj.m_DimCoeff,	sizeof(uint32) * 4 );
//}
//
//
//
//void Index4D::Init( uint32 dimx, uint32 dimy, uint32 dimz, uint32 dimw )
//{
//	//int i;
//
//	//Release();
//
//	m_numDim	= 4;
//	//m_Dim		= new uint32[ m_numDim ];
//	//m_DimCoeff	= new uint32[ m_numDim + 1 ];
//
//	//for( i=0; i<m_numDim; ++i )
//	//	m_Dim[i] = dim[i];
//	InitVec( m_Dim, dimx, dimy, dimz, dimw );
//
//
//	// インデックス計算時に使う、次元毎のオフセット
//	m_DimCoeff[0]	= 1;
//	m_DimCoeff[1]	= dimx;
//	m_DimCoeff[2]	= dimx * dimy;
//	m_DimCoeff[3]	= dimx * dimy * dimz;
//	m_DimCoeff[4]	= dimx * dimy * dimz * dimw;
//
//	//for( i=1; i<m_numDim+1; ++i )
//	//	m_DimCoeff[i] = m_Dim[i-1] * m_DimCoeff[i-1];
//}
//
//
//
//void Index4D::Release()
//{
//	//SafeDeleteArray( m_Dim );
//	//SafeDeleteArray( m_DimCoeff );
//	//m_numDim	= 0;
//}
//
//
//
//
//
//}// end of namespace