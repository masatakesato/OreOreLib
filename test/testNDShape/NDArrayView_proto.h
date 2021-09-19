#ifndef ND_ARRAY_VIEW_PROTO_H
#define	ND_ARRAY_VIEW_PROTO_H

#include	<oreore/common/TString.h>

#include	<oreore/container/ArrayView.h>
#include	<oreore/container/NDShape.h>


// https://www.codeproject.com/Articles/848746/ArrayView-StringView



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
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( int i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}
		


	private:

		NDShape<N>	m_Shape;

	};


}// end of namespace


#endif // !ND_ARRAY_VIEW_PROTO_H
