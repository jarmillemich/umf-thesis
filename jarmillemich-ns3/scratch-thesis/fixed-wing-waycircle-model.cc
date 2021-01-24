#include "fixed-wing-waycircle-model.h"
#include "ns3/simulator.h"



namespace ns3 {

//#region Pathing
namespace FixedWingWaycircle {
  // ns3 Vector is missing some things...
  Vector operator*(const Vector &v, const double s) {
    return Vector(
      v.x * s,
      v.y * s,
      v.z * s
    );
  }

  // Path segment
  PathSegment::PathSegment(double velocity)
  : velocity(velocity) {

  }

  PathSegment::~PathSegment() {

  }

  double PathSegment::getCycleTime() const {
    return getLength() / velocity;
  }

  // Line segment
  LineSegment::LineSegment(Vector p0, Vector p1, double velocity)
  : PathSegment(velocity), p0(p0), p1(p1), dp(p1 - p0) {

  }

  double LineSegment::getLength() const {
    return CalculateDistance(p0, p1);
  }

  Vector LineSegment::getPosition(double dTime) const {
    double totalTime = getLength() / velocity;
    double timeLerp = dTime / totalTime;

    return p0 + dp * timeLerp;
  }

  // Arc segment
  ArcSegment::ArcSegment(Vector center, double radius, double theta, double deltaTheta, double velocity)
  : PathSegment(velocity), center(center), radius(radius), theta(theta), deltaTheta(deltaTheta) {

  }

  double ArcSegment::getLength() const {
    return radius * deltaTheta;
  }

  Vector ArcSegment::getPosition(double dTime) const {
    double totalTime = getLength() / velocity;
    double timeLerp = dTime / totalTime;

    double currentTheta = theta + deltaTheta * timeLerp;

    return Vector(
      center.x + radius * cos(currentTheta),
      center.y + radius * sin(currentTheta),
      center.z
    );
  }
}
//#endregion Pathing

using namespace FixedWingWaycircle;

NS_OBJECT_ENSURE_REGISTERED (FixedWingWaycircleModel);

TypeId FixedWingWaycircleModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FixedWingWaycircleModel")
    .SetParent<MobilityModel> ()
    .SetGroupName ("Mobility")
    .AddConstructor<FixedWingWaycircleModel> ();
  return tid;
}

FixedWingWaycircleModel::FixedWingWaycircleModel() {

}

FixedWingWaycircleModel::~FixedWingWaycircleModel() {
  // Clean up our segments
  for (auto it : segments) {
    delete it;
  }
}

void FixedWingWaycircleModel::addLineSegment(Vector p0, Vector p1, double velocity) {
  if (isFinalized) {
    throw std::logic_error("Modifying path after finalized");
  }

  segments.push_back(new FixedWingWaycircle::LineSegment(p0, p1, velocity));
}

void FixedWingWaycircleModel::addArcSegment(Vector center, double radius, double theta, double deltaTheta, double velocity) {
  if (isFinalized) {
    throw std::logic_error("Modifying path after finalized");
  }

  segments.push_back(new FixedWingWaycircle::ArcSegment(center, radius, theta, deltaTheta, velocity));
}

void FixedWingWaycircleModel::finalize() {
  if (isFinalized) {
    throw std::logic_error("finalize() but already finalized");
  }

  // Set everyone's start time
  double time_at = 0;
  for (auto it : segments) {
    it->startTime = time_at;
    time_at += it->getLength() / it->velocity;
  }
  totalTime = time_at;

  isFinalized = true;
}

const PathSegment* FixedWingWaycircleModel::getSegment(void) const {
  //double realTime = Simulator::Now().GetSeconds() % totalTime;
  double realTime = fmod(Simulator::Now().GetSeconds(), totalTime);

  for (auto it : segments) {
    if (it->startTime <= realTime && realTime < it->startTime + it->getCycleTime()) {
      return it;
    }
  }

  // Format an error
  std::stringstream errMsg;
  errMsg << "Outside of time bounds for getting a segment t=" << realTime << " of " << totalTime;
  throw std::runtime_error(errMsg.str().c_str());
}

Vector FixedWingWaycircleModel::DoGetPosition(void) const {
  if (!isFinalized) {
    throw std::logic_error("FixedWingWaycircleModel must be finalized before use");
  }

  Time now = Simulator::Now();
  // TODO optimize
  const PathSegment *segment = getSegment();

  double realTime = fmod(Simulator::Now().GetSeconds(), totalTime);
  double dTime = realTime - segment->startTime;

  return segment->getPosition(dTime);
}

Vector FixedWingWaycircleModel::DoGetVelocity(void) const {
  return Vector(0, 0, 0);
}

void FixedWingWaycircleModel::DoSetPosition(const Vector &position) {
  // Nah
}

}