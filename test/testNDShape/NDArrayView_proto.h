#ifndef ND_ARRAY_VIEW_PROTO_H
#define	ND_ARRAY_VIEW_PROTO_H

#include	<oreore/common/TString.h>

#include	<oreore/container/ArrayView.h>
#include	<oreore/container/NDShape.h>


// https://www.codeproject.com/Articles/848746/ArrayView-StringView


//TODO: Disable Subscript operator

namespace OreOreLib
{


	template< typename T, uint64 N >
	class NDArrayView_proto : public ArrayView<T>
	{
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		NDArrayView_proto()
			: ArrayView<T>()
		{

		}


		// Constructor
		template < typename ... Args, std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value >* = nullptr >
		NDArrayView_proto( ConstPtr const pdata, const Args& ... args )
		{
			Init( pdata, args... );
		}


		// Constructor
		NDArrayView_proto( const Memory<T>& obj )
		{
			Init( obj );
		}


		//// Constructor using INDArray
		//NDArrayView_proto( const INDArray<T>& obj )
		//{
		//	Init( obj );
		//}


		// Constructor using NDArray



		// Destructor
		~NDArrayView_proto()
		{
			Release();
		}


		// Copy constructor
		NDArrayView_proto( const NDArrayView_proto& obj )
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
		static int div, mod;
		DivMod( div, mod, i, this->m_numCols );
		return (T *)(m_refData + m_ColOffset * div + mod);
	}
*/

		// Read only.( called if NDArray is const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, const T& >
		operator()( const Args& ... args ) const&// x, y, z, w...
		{
// TODO: Convert to Soruce data address offset using m_SrcStride

			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T& >
		operator()( const Args& ... args ) &// x, y, z, w...
		{
// TODO: Convert to Soruce data address offset using m_SrcStride

			return this->m_pData[ m_Shape.To1D( args... ) ];
		}


		// Subscript operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)(x, y)", "auto&& a = NDArray<T, 2>(10,10)(x, y)" )
		template < typename ... Args >
		std::enable_if_t< (sizeof...(Args)==N) && TypeTraits::all_convertible<uint64, Args...>::value, T >
		operator()( const Args& ... args ) const&&// x, y, z, w...
		{
// TODO: Convert to Soruce data address offset using m_SrcStride

			return (T&&)this->m_pData[ m_Shape.To1D( args... ) ];
		}


		//================= Element acces operators(initializer list) ===================//

		// Read only.( called if NDArray is const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, const T& >
		operator()( std::initializer_list<T_INDEX> indexND ) const&// x, y, z, w...
		{
// TODO: Convert indexND to Soruce data index

			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// Read-write.( called if NDArray is non-const )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T& >
		operator()( std::initializer_list<T_INDEX> indexND ) &// x, y, z, w...
		{
// TODO: Convert indexND to Soruce data index

			return this->m_pData[ m_Shape.To1D( indexND ) ];
		}


		// operator. ( called by following cases: "T& a = NDArray<T, 2>(10,10)({x, y})", "auto&& a = NDArray<T, 2>(10,10)({x, y})" )
		template < typename T_INDEX >
		std::enable_if_t< std::is_convertible<uint64, T_INDEX>::value, T >
		operator()( std::initializer_list<T_INDEX> indexND ) const&&// x, y, z, w...
		{
			return (T&&)this->m_pData[ m_Shape.To1D( indexND ) ];
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

				tcout << _T(": ") << *(this->begin() + i) << tendl;
			}

			tcout << tendl;
		}		


		// Disabled subscript operators
		const T& operator[]( std::size_t n ) const& = delete;
		T& operator[]( std::size_t n ) & = delete;
		T operator[]( std::size_t n ) const&& = delete;



	private:

		NDShape<N>	m_Shape;
		uint64		m_SrcStrides[N];


		using Memory<T>::operator[];
	};


}// end of namespace


#endif // !ND_ARRAY_VIEW_PROTO_H
