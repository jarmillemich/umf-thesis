#ifndef FIXED_WING_WAYCIRCLE_MODEL_H
#define FIXED_WING_WAYCIRCLE_MODEL_H

#include "ns3/nstime.h"
#include "mobility-model.h"

namespace ns3 {

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

  FixedWingWaycircleModel(double wayCircles[][4], double alphas[], double phaseOffset = 0.0);
  virtual ~FixedWingWaycircleModel();

private:
  virtual Vector DoGetPosition(void) const;
  virtual Vector DoGetVelocity(void) const;
}

} // namespace ns3

#endif /* FIXED_WING_WAYCIRCLE_MODEL_H */