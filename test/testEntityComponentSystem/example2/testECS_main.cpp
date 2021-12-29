// https://austinmorlan.com/posts/entity_component_system/

#include	<oreore/container/Array.h>
using namespace OreOreLib;


int main()
{
//	tcout << typeid(int).hash_code() << tendl;
//	tcout << typeid(uint32).hash_code() << tendl;

	Array<int32> a;

	for( int32 i=0; i<10; ++i )
		a.AddToTail( i );


	return 0;
}