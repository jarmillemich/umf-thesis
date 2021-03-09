import sys
if 'thesis.Aircraft' in sys.modules: del sys.modules['thesis.Aircraft']
from thesis.Aircraft import Aircraft

# Just some figuring out how heavy we are
# https://www.doubleradius.com/site/stores/baicells/baicells-nova-233-gen-2-enodeb-outdoor-base-station-datasheet.pdf
# Unsure how massive antenna are, guessing 1 kg
# Potentially unreliable sources say that the Starlink terminal is about 12 Ibs (https://www.reddit.com/r/Starlink/comments/jqck07/how_much_does_dishy_weigh/)
m_payload = 6  # kg
# Starlink power figure https://arstechnica.com/information-technology/2020/11/spacex-starlink-beta-tester-takes-user-terminal-into-forest-gets-120mbps/
# This figure may be overstated as it is for that users "whole system" (router, other equipment included?)
# Others say 50-100 W https://www.reddit.com/r/Starlink/comments/kf8ajt/power_consumption_of_starlink/
# We compromise here with 80
P_payload = (45 + 100) * 0.75       # W

# Main contributor to performance, both from an efficiency and solar area perspective
# --Estimating 8-24 m is sufficient (summer vs winter at 45 N latitude)--
# Zephyr 7 (22.5 m span) is around 50 kg

# We are now using the Matlab code from Oettershagen to determine good/semioptimal
# parameters. Our latitude is now the top of NC, USA (the "hurricane demarcation line")
# And our reference time for sustainable flight is the end of november ("hurricane season" end)
# Summertime will, of course, be better
# We are now slightly overestimating Oettershagen (45% MSoC us vs 25% MSoC them), but
# we are using different wings and calculating the polars is a hassle. Flight power is pretty
# close (257 W us vs 221 W them). There's probably some error from our garbage integration, as well
# as our angled panels (theirs are horizontal in early design)
wingSpan = 21 # m
wingChord = wingSpan / 20.5 # m
wingArea = wingSpan * wingChord # m^2
solarFill = 0.85 # % of wing surface with solar PV
solarArea = wingArea * solarFill # Solar panel area m^2

P_solar_cap = 3000 # Maximum solar power in (W)
P_prop = 1500      # Maximum propellor power (W)

# Determined using the Oettershagen code
m_struct = 50.16


m_prop = 0.0011 * P_prop
m_solar = 0.59 * solarArea
m_mppt = 0.422 * 0.422e-3 * P_solar_cap
m_av = 1.22
bat_Wh_cap = 15.5 * 650
# From Oettershagen
#m_bat = bat_Wh_cap / 251
# From hypothetical Licerion/Sionpower Lithium-metal cells
m_bat = bat_Wh_cap / 650

mass = m_struct + m_prop + m_solar + m_mppt + m_av + m_payload + m_bat
print('Mass is %.2f kg (%.2f kg struct, %.2f kg bat)' % (mass, m_struct, m_bat))

# From NS3 defaults
radioParams = {
    'xmitPower': 30, # dBm
    'B': 180e3 * 25, # 25 180kHz RBs = 4.5 MHz
    'N0': -174       # See lte-spectrum-value-helper.cc kT_dBm_Hz
}

craft = Aircraft(mass = mass, wingSpan = wingSpan, e0 = 0.92, chord = wingChord)