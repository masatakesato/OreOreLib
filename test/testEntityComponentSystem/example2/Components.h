#ifndef COMPONENTS_H
#define	COMPONENTS_H

#include	<oreore/mathlib/GraphicsMath.h>



struct Gravity
{
	Vec3f force;
};


struct RigidBody
{
	Vec3f velocity;
	Vec3f acceleration;
};


struct Transform
{
	Vec3f position;
	Vec3f rotation;
	Vec3f scale;
};



#endif // !COMPONENTS_H
