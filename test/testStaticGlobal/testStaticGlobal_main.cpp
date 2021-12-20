#include	<iostream>
#include	<string>
using namespace std;

#include	<oreore/memory/DebugNew.h>
#include	<oreore/memory/UniquePtr.h>


struct fValue
{
	float val;

	fValue( ){}
	fValue( const float& f )
		: val(f)
	{
	}
};




template < typename T >
class AAA
{
public:

	static T* Get()	{ return m_Instance.Get(); }


protected:

	~AAA()
	{
		cout << "~AAA()\n";
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
OreOreLib::UniquePtr<T> AAA<T>::m_Instance = nullptr;
//T* AAA<T>::m_Instance = nullptr;




// https://github.com/EQMG/Acid/blob/cb1e62a80cdba662a0b2c1ba008b2bf4a397877a/Sources/Animations/AnimatedMesh.hpp
class TestStruct : public AAA<fValue>
{
public:

	TestStruct()
	{
		Register( 9999.9f );
	}

private:

};



class TestStr : public AAA<string>
{
public:

	TestStr()
	{
		Register( "oreore" );
	}

private:

};





int main()
{
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	TestStruct test;
	TestStr teststr;

	cout << (*TestStruct::Get()).val << endl;
	cout << *TestStr::Get() << endl;

	return 0;
}