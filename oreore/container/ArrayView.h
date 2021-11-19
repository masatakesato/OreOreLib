﻿#ifndef ARRAY_VIEW_H
#define	ARRAY_VIEW_H

#include	"../common/TString.h"
//#include	"../memory/Memory.h"

#include	"ArrayBase.h"



// https://www.codeproject.com/Articles/848746/ArrayView-StringView



namespace OreOreLib
{


	template< typename T >
	class ArrayBase< detail::ARRVIEW<T>, detail::DynamicSize > : public Memory<T>
	{
		using Ptr = T*;
		using ConstPtr = const T*;

	public:

		// Default constructor
		ArrayBase()
			: Memory<T>()
		{
		}


		// Constructor
		ArrayBase( ConstPtr const pdata, int length )
		{
			Init( pdata, length );
		}


		// Constructor
		ArrayBase( const Memory<T>& obj )
		{
			Init( obj );
		}


		// Destructor
		~ArrayBase()
		{
			Release();
		}


		// Copy constructor
		ArrayBase( const ArrayBase& obj )
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


		void Init( const Memory<T>& obj )
		{
			this->m_pData		= (T*)obj.begin();
			this->m_Length		= obj.Length();
			this->m_AllocSize	= 0;
		}


		void Init( std::initializer_list<T> ilist )
		{
			MemCopy( this->begin(), ilist.begin(), Min( ilist.size(), this->m_Length ) );
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

		using Memory<T>::Init;
		using Memory<T>::Release;

		bool Resize( int ) = delete;//using Memory<T>::Resize;
		bool Resize( int, const T& ) = delete;
		bool Reserve( int ) = delete;
		bool Extend( int ) = delete;//using Memory<T>::Extend;
		bool Shrink( int ) = delete;//using Memory<T>::Shrink;


	};


}// end of namespace


#endif // !ARRAY_VIEW_H
