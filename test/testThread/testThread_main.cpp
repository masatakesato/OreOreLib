#include	<oreore/common/TString.h>
#include	<oreore/thread/IRunnable.h>
#include	<oreore/thread/Thread.h>


class MyFunc : public OreOreLib::IRunnable
{
public:

	MyFunc( int& val1, int &val2 )
		: m_val1( val1 )
		, m_val2( val2 )
	{

	}



	void IRunnable::Run()
	{
		tcout << "//============= MyFunc::Run() ==============//\n";

		m_val1++;
		m_val2--;
	}


private:

	int& m_val1;
	int& m_val2;


};



int main()
{
	OreOreLib::Thread thread1;

	int a = 0;
	int b = 0;


	thread1.Init( new MyFunc( a, b ) );
	//thread1.Init( a );
	//thread1.Play();


	tcout << a << tendl;
	tcout << b << tendl;

	return 0;
}