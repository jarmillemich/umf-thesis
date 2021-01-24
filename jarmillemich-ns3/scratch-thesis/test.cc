#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "fixed-wing-waycircle-model.h"
#include <math.h>

using namespace ns3;

void logLocations(FixedWingWaycircleModel *model) {
	std::cout
		<< Simulator::Now().GetSeconds()
		<< ": "
		<< model->GetPosition()
		<< std::endl;

	Simulator::Schedule(Seconds(1), &logLocations, model);
}

int mainz(int argc, char *argv[]) {
	NS_LOG_UNCOND("Hello world!");

	FixedWingWaycircleModel foo;

	foo.addLineSegment(Vector(0, 0, 0), Vector(100, 0, 0), 10);
	foo.addArcSegment(Vector(100, 50, 0), 50, -M_PI / 2, M_PI, 10);
	foo.addLineSegment(Vector(100, 100, 0), Vector(0, 100, 0), 10);
	foo.addArcSegment(Vector(0, 50, 0), 50, M_PI / 2, M_PI, 10);
	foo.finalize();

	std::cout << foo.GetPosition() << std::endl;

	Simulator::Schedule(Seconds(1), &logLocations, &foo);

	Simulator::Schedule(Seconds(60), []() {
		Simulator::Stop();
	});
	Simulator::Run();
	Simulator::Destroy();

	return 0;
}
