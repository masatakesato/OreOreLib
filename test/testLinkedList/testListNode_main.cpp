#define	_CRTDBG_MAP_ALLOC
#include	<crtdbg.h>
#include	<list>

#include	<oreore/mathlib/MathLib.h>
#include	<oreore/common/TString.h>
#include	<oreore/container/LinkedList.h>
using namespace OreOreLib;




int main()
{
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );

	{

		ListNode<int>* o = new ListNode<int>;
		ListNode<int>* a = new ListNode<int>;
		ListNode<int>* b = new ListNode<int>;
		ListNode<int>* c = new ListNode<int>;
		ListNode<int>* d = new ListNode<int>;

		// initialize
		o->next = o->prev = o;
		o->data = -1;

		a->data = 0;
		b->data = 1;
		c->data = 2;
		d->data = 3;

		// connect
		c->ConnectAfter( o );

		d->ConnectAfter( c );

		a->ConnectBefore( c );

		b->ConnectAfter( a );

		b->Disconnect();

		//o->Disconnect();

		return 0;
	}

	

	//{
	//	tcout << "//====================== ======================//" << tendl;

	//	uint8 *data = new uint8[ sizeof(ListNode<int>) ];

	//	// placement new
	//	ListNode<int>* pnode = new (data) ListNode<int>();

	//	pnode->next = pnode;
	//	pnode->prev = pnode;
	//	pnode->data = 99999;

	//	tcout << pnode->data << tendl;

	//	// direct mapping from bytearray
	//	ListNode<int>* pnode2 = (ListNode<int>*)data;

	//	tcout << pnode2->next->data << tendl;
	//}

	//tcout << tendl;

	//{
	//	using Page = ListNode<float[1]>;

	//	int numElms = 11;

	//	tcout << "Page size with data[1]: " << sizeof(Page) << tendl;
	//
	//	size_t allocSize = DivUp( sizeof(Page*) * 2 + sizeof(float)*numElms, std::alignment_of_v<Page> ) * std::alignment_of_v<Page>;
	//	tcout << "Page size with data[" << numElms << "]: " << allocSize << tendl;

	//	uint8 *data = new uint8[ allocSize ];

	//	// placement new
	//	tcout << "Page* ppage = new (data) Page();\n";
	//	Page* ppage = new (data) Page();

	//	ppage->next = ppage;
	//	ppage->prev = ppage;
	//	ppage->data[ numElms-1 ] = 99999;
	//	ppage->data[0] = 55;

	//	tcout << "ppage->next->data[ numElms-1 ] = " << ppage->next->data[ numElms-1 ] << tendl;
	//	tcout << "ppage->next->data[ numElms-1 ] = " << ppage->prev->data[ numElms-1 ] << tendl;
	//	tcout << tendl;

	//	// direct mapping from bytearray
	//	tcout << "Page* ppage2 = (Page*)data;\n";
	//	Page* ppage2 = (Page*)data;

	//	tcout << "ppage2->next->data[ numElms-1 ] = " << ppage2->next->data[ numElms-1 ] << tendl;
	//	tcout << "ppage2->prev->data[ numElms-1 ] = " << ppage2->prev->data[ numElms-1 ] << tendl;
	//	tcout << tendl;
	//


	//	tcout<< data << tendl;
	//	tcout<< *(float*)(data + sizeof(Page*)*2 + sizeof(float) * (numElms-1))  << tendl;

	//	//delete ppage2;

	//	//tcout<< data << tendl;
	//	//tcout<< *(float*)(data + sizeof(Page*)*2 + sizeof(float) * (numElms-1))  << tendl;

	//	//delete [] data;

	//	//tcout<< data << tendl;
	//	//tcout<< *(float*)(data + sizeof(Page*)*2 + sizeof(float) * (numElms-1))  << tendl;

	//}

	return 0;
}