#ifndef VECTOR_OPERATIONS_H
#define	VECTOR_OPERATIONS_H


#include	<math.h>



// Euclidean Distance(L2 norm)
template< typename T>
T EuclideanDistance( int dim, const T vec1[], const T vec2[] )
{
	double dist = 0;
	for( int i=0; i<dim; ++i )
	{
		dist += double( ( vec1[i] - vec2[i] ) * ( vec1[i] - vec2[i] ) );
	}
	return (T)sqrt( dist );
}



// Chebyshev Distance
template< typename T>
T ChebyshevDistance( int dim, const T vec1[], const T vec2[] )
{
	double dist = 0;
	for( int i=0; i<dim; ++i )
	{
		T d = fabs( double( vec1[i] - vec2[i] ) );
		dist = d > dist ? d : dist;
	}
	return dist;
}



// Standardized Euclidean distance



// Manhattan Distance
template< typename T>
T MahnattannDistance( int dim, const T vec1[], const T vec2[] )
{
	double dist = 0;
	for( int i=0; i<dim; ++i )
	{
		dist += fabs( double( vec1[i] - vec2[i] ) );
	}
	return (T)dist;
}



// Cosine similarity
template< typename T>
static T CosineDistance( int dim, const T vec1[], const T vec2[] )
{
	double inner_product = 0;
	double len1	= 0;
	double len2 = 0;

	for( int i=0; i<dim; ++i )
	{
		inner_product += double( vec1[i] * vec2[i] );
		len1 += double( vec1[i] * vec1[i] );
		len2 += double( vec2[i] * vec2[i] );
	}
	return T( inner_product / ( sqrt( len1 ) * sqrt( len2 ) ) );
}



// Histogram intersection
template < typename T >
static T HistogramIntersection( int dim, const T vec1[], const T vec2[] )//const Histogram &bof1, const Histogram &bof2 )
{
	T d = 0;
	T div = 0;

	for( int i=0; i<dim.m_numBins; ++i )
	{
		d += min( vec1[i], vec2[i] );
		div += vec1[i];
	}// end of i loop

	div = div<1.0e-9 ? 1 : div;

	return T( d / div );
}




/*
double BhattacharyyaDistance( int dim, const T vec1[], const T vec2[] )//const Histogram &bof1, const Histogram &bof2 )
{
	const int NUM = bof1.m_numBins;

	int i;						 //iteration変数
	double vec1, vec2;  //正規化ヒストグラム  (NUMはヒストグラムの要素数)
	double likelihood=0.0;   //類似度
	double sum1 = 0.0, sum2 = 0.0;   //正規化に使う合計値格納変数

	for(i=0; i<NUM; i++)    //ヒストグラムを正規化するため合計値を計算する(要素サイズはNUM)
	{
		sum1 += bof1.m_Histogram[i];
		sum2 += bof2.m_Histogram[i];
	}
	for(i=0; i<NUM; i++)    //正規化(合計値から割る)して係数を累積している(要素サイズはNUM)
	{
		vec1 = bof1.m_Histogram[i] / sum1; 
		vec2 = bof2.m_Histogram[i] / sum2;
		likelihood += sqrt(vec1 * vec2);      //累積
	}
	//計算した類似度を返す
	return likelihood;

}
*/



/*
// bof1から見た
double KullbackLeiblerDistance( const Histogram &bof1, const Histogram &bof2 )
{
	double result = 0.0;

	for( int i=0; i<bof1.m_numBins; ++i )
	{
		double P_Q = max( bof1.m_Histogram[i] / max( bof2.m_Histogram[i], 1.0e-9 ), 1.0e-9 );

		result	+= //bof1.m_Histogram[i] * log( P_Q );// P(x)*log( P(x)/Q(x) );
					( bof1.m_Histogram[i] - bof2.m_Histogram[i] ) * log( P_Q );// modified version for symmetry
	}// end of i loop


	return result;
}
*/





#endif // !VECTOR_OPERATIONS_H
