#include	<oreore/datamanager/ObjectPool.h>
using namespace OreOreLib;



int main()
{
	PoolAllocator allocator;
	ObjectPool< int, 4096 > pool;

	pool.Init( &allocator );


	allocator.Display();

	int* ptr = pool.Allocate();

	allocator.Display();


	return 0;
}