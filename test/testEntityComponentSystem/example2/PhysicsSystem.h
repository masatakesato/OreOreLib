#ifndef PHYSICS_SYSTEM_H
#define	PHYSICS_SYSTEM_H

#include	"System.h"


class PhysicsSystem : public System
{
public:

	void Init();

	void Update( float dt );

};


#endif // !PHYSICS_SYSTEM_H
