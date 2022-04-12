#include	<iostream>
#include	<string>
using namespace std;

#include	<oreore/memory/DebugNew.h>
#include	<oreore/memory/UniquePtr.h>


struct FloatStruct
{
	float val;

	FloatStruct( ){}
	FloatStruct( const float& f )
		: val(f)
	{
	}
};




template < typename T >
class GenericGetter
{
public:

	static T* Get()	{ return m_Instance.Get(); }


protected:

	~GenericGetter()
	{
		cout << "~GenericGetter()\n";
		m_Instance.Reset();
	}

	template < typename ... Args >
	static bool Register( Args ... args )
	{
		m_Instance = new T( args... );
		return true;
	}

	static OreOreLib::UniquePtr<T> m_Instance;
	//static T* m_Instance;

};


template < typename T >
OreOreLib::UniquePtr<T> GenericGetter<T>::m_Instance = nullptr;
//T* GenericGetter<T>::m_Instance = nullptr;




// https://github.com/EQMG/Acid/blob/cb1e62a80cdba662a0b2c1ba008b2bf4a397877a/Sources/Animations/AnimatedMesh.hpp
class Float : public GenericGetter<FloatStruct>
{
public:

	Float()
	{
		Register( 9999.9f );
	}

private:

};



class String : public GenericGetter<string>
{
public:

	String()
	{
		Register( "oreore" );
	}

private:

};





int main()
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	Float testfloat;
	String teststr;

	cout << (*Float::Get()).val << endl;
	cout << *String::Get() << endl;

	return 0;
}