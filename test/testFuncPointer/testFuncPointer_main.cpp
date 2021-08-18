#include	<iostream>
using namespace std;



class B
{
public:
	B()
	{
		cout << "B()..." << endl;
		pfunc = NULL;
	}


	void BindFunc( void( *pfunc )( ) )
	{
		this->pfunc = pfunc;
	}


	void DoFunc()
	{
		this->pfunc();
	}

private:

	void( *pfunc )( );
};



class A
{
public:
	A()
	{
		cout << "A()..." << endl;
	}

	static void func()
	{
		cout << "I am class A func" << endl;
	}

private:



};




int main( int argc, char **argv )
{
	A a = A();
	B b = B();

	b.BindFunc( &a.func );

	b.DoFunc();

	return 0;
}