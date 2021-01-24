#ifndef FIXED_WING_WAYCIRCLE_MODEL_H
#define FIXED_WING_WAYCIRCLE_MODEL_H

#include "ns3/nstime.h"
#include "ns3/mobility-model.h"

namespace ns3 {

namespace FixedWingWaycircle {
  class PathSegment {
  public:
    PathSegment(double velocity);
    virtual ~PathSegment();

    const double velocity;

    virtual double getLength() const = 0;
    virtual Vector getPosition(double dTime) const = 0;
    double getCycleTime() const;

    // To be set by finalize
    double startTime = 0;
  };

  class LineSegment : public PathSegment {
  public:
    LineSegment(Vector p0, Vector p1, double velocity);

    const Vector p0, p1, dp;

    virtual double getLength() const;
    virtual Vector getPosition(double dTime) const;
  };

  class ArcSegment : public PathSegment {
  public:
    ArcSegment(Vector center, double radius, double theta, double deltaTheta, double velocity);

    const Vector center;
    const double radius, theta, deltaTheta;

    virtual double getLength() const;
    virtual Vector getPosition(double dTime) const;
  };
}

/**
 * \ingroup mobility
 * 
 * \brief Mobility model for a fixed wing aircraft under @@jarmillemich's 
 * "waycircle" scheme.
 */
class FixedWingWaycircleModel : public MobilityModel {
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // Really, on this end, all we care about are lines and arcs and velocity
  // Lines are (x0, y0, z0, x1, y1, z1, v)
  // Arcs are (cx, cy, cz, r, theta, dtheta, v)
  // Coincidentally, these have the same number of elements...

  FixedWingWaycircleModel();
  virtual ~FixedWingWaycircleModel();

  /** Add a line from p0 to p1 */
  void addLineSegment(Vector p0, Vector p1, double velocity);

  /** Add an arc around center with the given radius, starting theta, and delta theta */
  void addArcSegment(Vector center, double radius, double theta, double deltaTheta, double velocity);

  /** Finalize the path, no further changes can be made */
  void finalize();

private:
  virtual Vector DoGetPosition(void) const;
  virtual Vector DoGetVelocity(void) const;

  virtual void DoSetPosition (const Vector &position);

  const FixedWingWaycircle::PathSegment* getSegment(void) const;

  // These should all be set up before actual use...
  bool isFinalized = false;
  std::vector<FixedWingWaycircle::PathSegment*> segments;

  // To be set by finalize
  double totalTime = 0;
};

} // namespace ns3

#endif /* FIXED_WING_WAYCIRCLE_MODEL_H */