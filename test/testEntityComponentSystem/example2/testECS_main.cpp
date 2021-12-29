// https://austinmorlan.com/posts/entity_component_system/

#include	<oreore/mathlib/GraphicsMath.h>

#include	"Coordinator.h"



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




Coordinator	gCoordinator;



int main()
{


	return 0;
}