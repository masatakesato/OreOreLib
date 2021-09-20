#ifndef ND_ARRAY_VIEW_PROTO_H
#define	ND_ARRAY_VIEW_PROTO_H

#include	<oreore/common/TString.h>

//#include	<oreore/container/ArrayView.h>
//#include	<oreore/container/NDShape.h>

#include	"NDArrayBase.h"


// https://www.codeproject.com/Articles/848746/ArrayView-StringView


//TODO: Disable Subscript operator

namespace OreOreLib
{


	template< typename T, uint64 N >
	class /*NDArrayView_proto*/NDArrayBase< detail::NDARRVIEW<T>, N > : public ArrayView<T>
	{
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		/*NDArrayView_proto*/NDArrayBase()
			: ArrayView<T>()
			, m_pSrcShape( nullptr )
		{

		}


		// Constructor
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		/*NDArrayView_proto*/NDArrayBase( ConstPtr const pdata, const Args& ... args )
			: m_pSrcShape( nullptr )
		{
			Init( pdata, args... );
		}


		// Constructor
		/*NDArrayView_proto*/NDArrayBase( const Memory<T>& obj )
			: m_pSrcShape( nullptr )
		{
			Init( obj );
		}


		//// Constructor using INDArray
		///*NDArrayView_proto*/NDArrayBase( const INDArray<T>& obj )
		//{
		//	Init( obj );
		//}


		// Constructor using NDArray



		// Destructor
		~NDArrayBase/*NDArrayView_proto*/()
		{
			Release();
		}


		// Copy constructor
		/*NDArrayView_proto*/NDArrayBase( const /*NDArrayView_proto*/NDArrayBase& obj )
			: ArrayView( obj )
			, m_Shape( obj.m_Shape )
		{

		}


		//================= Element access operators(variadic templates) ===================//


/*
	T *ptr( int row, int col ) const// row: vertical, col: horizontal
	{
		return (T *)( this->m_pData + m_DimX * y + x );
	}


	T *ptr( int i ) const
	{
		// 一旦NDArrayView空間上のN次元インデックスに戻す
		static int y, x;
		y = i / this->m_numCols;
		x = i % this->m_numCols;
		
		// 参照元空間の大きさを使って、N次元インデックスから1Dインデックスに変換する
		return (T*)( m_refData + m_NumSrcColumn * y + x );

		//DivMod( div, mod, i, this->m_numCols );
		//return (T *)(m_refData + m_ColOffset * div + mod);
	}
*/


		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
// TODO: Convert to Soruce data address offset using m_SrcStride
auto index = m_pSrcShape->To1D( args... );
			return this->m_pData[ index/*m_Shape.To1D( args... )*/ ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
// TODO: Convert to Soruce data address offset using m_SrcStride
auto index = m_pSrcShape->To1D( args... );
			return this->m_pData[ index/*m_Shape.To1D( args... )*/ ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
// TODO: Convert to Soruce data address offset using m_SrcStride
auto index = m_pSrcShape->To1D( args... );
			return (T&&)this->m_pData[ index/*m_Shape.To1D( args... )*/ ];
		}


		//================= Element acces operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
// TODO: Convert indexND to Soruce data index
auto index = m_pSrcShape->To1D( indexND );
			return this->m_pData[ index/*m_Shape.To1D( indexND )*/ ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
// TODO: Convert indexND to Soruce data index
auto index = m_pSrcShape->To1D( indexND );
			return this->m_pData[ index/*m_Shape.To1D( indexND )*/ ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
// TODO: Convert indexND to Soruce data index
auto index = m_pSrcShape->To1D( indexND );
			return (T&&)this->m_pData[ index/*m_Shape.To1D( indexND )*/ ];
		}



		void Init( const Memory<T>& obj )
		{
			ArrayView<T>::Init( obj );
			m_Shape.Init( obj.Length() );
		}


		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, void >
		Init( ConstPtr const pdata, const Args& ... args )
		{
			m_Shape.Init( args... );
			ArrayView<T>::Init( pdata, m_Shape.Size() );
		}


		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, void >
		Init( ConstPtr const pdata, std::initializer_list<T_INDEX> indexND )
		{
			m_Shape.Init( indexND );
			ArrayView<T>::Init( pdata, int(m_Shape.Size()) );
		}


		void Release()
		{
			ArrayView<T>::Release();
			m_Shape.Release();
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T(":\n" );

			for( int i=0; i<this->m_Length; ++i )
			{
				tcout << _T("  ");
				for( int dim=(int)m_Shape.NumDims()-1; dim>=0; --dim )
					tcout << _T("[") << m_Shape.ToND(i, dim) << _T("]");

				tcout << _T(": ") << this->m_pData[i] << tendl;
			}

			tcout << tendl;
		}		


		// Disabled subscript operators
		const T& operator[]( std::size_t n ) const& = delete;
		T& operator[]( std::size_t n ) & = delete;
		T operator[]( std::size_t n ) const&& = delete;



	private:

		NDShape<N>	m_Shape;
		NDShape<N>*	m_pSrcShape=nullptr;


		using Memory<T>::operator[];
		using Memory<T>::begin;
		using Memory<T>::end;
	};


}// end of namespace


#endif // !ND_ARRAY_VIEW_PROTO_H
