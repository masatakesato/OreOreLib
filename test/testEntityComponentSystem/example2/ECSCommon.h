#ifndef ECS_COMMON_H
#define	ECS_COMMON_H

#include	<oreore/common/Utility.h>
#include	<oreore/container/BitArray.h>


// Entity
using Entity = uint32;

const Entity MAX_ENTITIES	= 5000;



// Component
using ComponentType = uint8;

const ComponentType MAX_COMPONENTS = 32;


using Signature = StaticBitArray<MAX_COMPONENTS>;



#endif // !ECS_COMMON_H
