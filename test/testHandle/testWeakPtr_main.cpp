#include	<crtdbg.h>

#include	<oreore/Vector.h>
#include	<oreore/memory/SharedPtr.h>
#include	<oreore/memory/WeakPtr.h>
using namespace OreOreLib;



int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );



	fArray *array = new fArray(20);

	SharedPtr<fArray> arrptr( new fArray(3) );

	arrptr = SharedPtr<fArray>( array );

	
	(*arrptr)[0] = 5555;			// Call T& operator*()
	arrptr->begin()[1] = -5869486;	// Call T* operator->()

	tcout << (*arrptr)[0] << tendl;

	//tcout << arrptr.UseCount() << tendl;

	WeakPtr<fArray> w_arrptr = arrptr;
	WeakPtr<fArray> *w_arrptr2 = new WeakPtr<fArray>(arrptr);


//	//(*const_arrptr)[0];		// Call const T& operator*() const
//	//const_arrptr->begin();	// const T* operator->() const
//
//
//	
//	(*w_arrptr)[2] = -6646;
//	//tcout <<  (*w_arrptr)[2] << tendl;
//
//	w_arrptr.Reset();
//
//	SharedPtr<fArray> arrptr2( *w_arrptr2 );
//
//
//	(*w_arrptr2)->Display();
//
//
	w_arrptr2->Reset();
	delete w_arrptr2;
//
//
//	
//	//tcout << (*arrptr)[2] << tendl;
//	
//	arrptr2.Reset();
//	arrptr.Reset();
//
//	
//
////	tcout << const_arrptr.UseCount() << tendl;
//
//	int *gggg = new int(55);
//	SharedPtr<int> ddd( gggg );
//
//	SharedPtr<int[]> bbb( new int[10] );
//
//	WeakPtr<int[]> w_bbb( bbb );
//	WeakPtr<int[]> w2_bbb( w_bbb );
//
//	w_bbb.Swap( w2_bbb );
//
//	SharedPtr<int[]> bbb2( w2_bbb );
//
//
//	w2_bbb.Reset();
//

	return 0;
}







//#include <functional>
//#include <iostream>
//using namespace std;
//
//
//template< typename T >
//class Deleter
//{
//  public:
//  
//  void operator()( T* ptr ) const
//  {
//      cout << "Deleter\n";
//      delete ptr;
//      ptr=nullptr;
//  }
//    
//};
//
//
//template< typename T >
//class Deleter<T[]>
//{
//  public:
//  
//  void operator()( T* ptr ) const
//  {
//      cout << "Array Deleter\n";
//      delete [] ptr;
//      ptr=nullptr;
//  }
//    
//};
//
//
//template< typename T >
//class A
//{
//    public:
//    
//    void func( T* ptr, const std::function<void(T*)>& f=Deleter<T>()  )
//    {
//        f(ptr);
//    }
//};
//
//
//
//template< typename T >
//class A<T[]>
//{
//    public:
//    
//    void func( T* ptr, const std::function<void(T*)>& f=Deleter<T[]>() )// Compile error occurs with permissive mode
//    {
//        f(ptr);
//    }
//
//    void func( T* ptr, const std::function<void(T*)>& f=(Deleter<T[]>()) )// OK
//    {
//        f(ptr);
//    }
//
//};
//
//
//
//template< typename T >
//void func( T* ptr, const std::function<void(T*)>& f=Deleter<T[]>() )// OK
//{
//    f(ptr);
//    
//}
//
//
//
//
//int main(void)
//{
//    int *a = new int[99];
//    func(a);
//    
//    //A<int[]> a_;
//    //a_.func( a );
//}