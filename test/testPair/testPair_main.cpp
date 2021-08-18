#include	<oreore/common/TString.h>
#include	<oreore/container/Pair.h>


class A
{
public:

	A()
	{
		tcout << "A()...\n";
	}


	A( const A& obj )
	{
		tcout << "Copy constructor...\n";
	}


	A( A&& obj )
	{
		tcout << "Move constructor...\n";
	}

};



int main()
{
	int v1 = 6;
	int v2 = 9;
	Pair<int, int> _(v1, v2);

	Pair<int, int> a(3, 3);
	Pair<int, int> b(3, 3);


	int* val1 = new int(5);
	int* val2 = new int(-5);
	A A_ = A();
	Pair<A, A> c( std::move(A_), std::move(A_) );//std::move(val1), std::move(val2) );

//	delete val1;

///	tcout << c.first << tendl;


	return 0;
}