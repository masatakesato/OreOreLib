// https://austinmorlan.com/posts/entity_component_system/


#include	"Components.h"
#include	"Coordinator.h"
#include	"PhysicsSystem.h"


Coordinator	gCoordinator;




int main()
{
	gCoordinator.Init();

	gCoordinator.RegisterComponent<Gravity>();
	gCoordinator.RegisterComponent<RigidBody>();
	gCoordinator.RegisterComponent<Transform>();

	auto physicsSystem = gCoordinator.RegisterSystem<PhysicsSystem>();

	Signature signature;
	signature.Set( gCoordinator.GetComponentType<Gravity>() );
	signature.Set( gCoordinator.GetComponentType<RigidBody>() );
	signature.Set( gCoordinator.GetComponentType<Transform>() );

	gCoordinator.SetSystemSignature<PhysicsSystem>( signature );




	return 0;
}