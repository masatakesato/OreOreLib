#ifndef COMPONENT_ARRAY_H
#define	COMPONENT_ARRAY_H

#include	<oreore/container/StaticArray.h>
#include	<oreore/container/HashMap.h>

#include	"ECSCommon.h"



class IComponentArray
{
public:

	virtual ~IComponentArray() = default;
	virtual void EntityDestroyed( Entity entity ) = 0;
};



template < typename T >
class ComponentArray : public IComponentArray
{
public:

	void InsertData( Entity entity, T component )
	{

	}


	void RemoveData( Entity entity )
	{

	}


	T& GetData( Entity entity )
	{

	}


	void EntityDestroyed( Entity entity ) override
	{

	}



private:


	OreOreLib::StaticArray<T, MAX_ENTITIES>	mComponentArray;

	OreOreLib::HashMap<Entity, size_t, 128>	mEntityToIndexMap;


	OreOreLib::HashMap<Entity, size_t, 128>	mIndexToEntityMap;

	size_t	mSize;
};


#endif // !COMPONENT_ARRAY_H
