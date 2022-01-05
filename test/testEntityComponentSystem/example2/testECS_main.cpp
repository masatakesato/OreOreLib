// https://austinmorlan.com/posts/entity_component_system/

#include	<oreore/memory/DebugNew.h>
#include	<oreore/container/Array.h>


#include	"Components.h"
#include	"Coordinator.h"
#include	"PhysicsSystem.h"


Coordinator	gCoordinator;




int main()
{
	gCoordinator.Init();

	// Component Registration
	gCoordinator.RegisterComponent<Gravity>();
	gCoordinator.RegisterComponent<RigidBody>();
	gCoordinator.RegisterComponent<Transform>();

	// PhysicsSystem setup
	auto physicsSystem = gCoordinator.RegisterSystem<PhysicsSystem>();

	Signature signature;// register component signatures
	signature.Set( gCoordinator.GetComponentType<Gravity>() );
	signature.Set( gCoordinator.GetComponentType<RigidBody>() );
	signature.Set( gCoordinator.GetComponentType<Transform>() );

	tcout << signature << tendl;

	gCoordinator.SetSystemSignature<PhysicsSystem>( signature );



	// Create Instances
	OreOreLib::Array<Entity> entities( MAX_ENTITIES );

	for( auto& entity : entities )
	{
		entity = gCoordinator.CreateEntity();

		gCoordinator.AddComponent( entity, Gravity{ Vec3f( 0.0f, -9.8f, 0.0f ) } );

		gCoordinator.AddComponent( entity, RigidBody{ Vec3f(), Vec3f() } );

		gCoordinator.AddComponent( entity, Transform{ Vec3f(), Vec3f(), Vec3f(1,1,1) } );
	}



	return 0;
}