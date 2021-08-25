#ifndef ARRAY_VIEW_H
#define	ARRAY_VIEW_H

#include	"../common/TString.h"
#include	"../memory/Memory.h"



namespace OreOreLib
{

	// https://www.codeproject.com/Articles/848746/ArrayView-StringView

	template< typename T >
	class ArrayView : public Memory<T>
	{
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		ArrayView()
			: Memory()
		{
		}


		ArrayView( ConstPtr const pdata, int length )
		{
			Init( pdata, length );
		}


		~ArrayView()
		{
			Release();
		}


		// copy constructor
		ArrayView( const ArrayView& obj )
		{
			this->m_pData		= obj.m_pData;
			this->m_Length		= obj.m_Length;
			this->m_AllocSize	= 0;
		}


		void Init( ConstPtr const pdata, int length )
		{
			this->m_pData		= (Ptr)pdata;
			this->m_Length		= length;
			this->m_AllocSize	= 0;
		}


		void Release()
		{
			this->m_pData = nullptr;
			this->m_Length = 0;
		}


		void Display() const
		{
			tcout << typeid(*this).name() << _T("[ ") << this->m_Length << _T(" ]:\n" );

			for( int i=0; i<this->m_Length; ++i )
				tcout << _T("  [") << i << _T("]: ") << this->m_pData[i] << tendl;

			tcout << tendl;
		}
		


	private:

		bool Extend( int ) = delete;//using Memory<T>::Extend;
		bool Resize( int ) = delete;//using Memory<T>::Resize;
		bool Shrink( int ) = delete;//using Memory<T>::Shrink;


	};


}// end of namespace


#endif // !ARRAY_VIEW_H
