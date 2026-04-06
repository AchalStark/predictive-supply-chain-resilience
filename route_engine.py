"""
Route Engine — Defines all trade routes, geographic corridors,
country mappings, and commodity-specific adjustments for GRV calculation.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# ROUTE DEFINITIONS
# Each route has:
#   - name, description
#   - waypoints (lat, lon) for map rendering
#   - bounding_boxes: list of (lat_min, lat_max, lon_min, lon_max) for event matching
#   - corridor_radius_km: buffer around waypoints for matching events
#   - countries: ISO numeric country codes along the route (for sanctions)
#   - chokepoints: key strategic chokepoints
# ─────────────────────────────────────────────────────────────

ROUTES = {
    "europe_india_suez": {
        "name": "Europe → India (Suez Canal)",
        "short_name": "Suez Route",
        "description": "Mediterranean → Suez Canal → Red Sea → Arabian Sea → India",
        "color": "#FF6B6B",
        "icon": "🚢",
        "waypoints": [
            (43.0, 5.0),      # Mediterranean (France)
            (38.0, 15.0),     # Central Med
            (35.0, 24.0),     # Crete
            (31.5, 32.3),     # Suez Canal
            (27.0, 34.0),     # Red Sea North
            (20.0, 38.5),     # Red Sea Central
            (12.5, 43.5),     # Bab-el-Mandeb
            (14.0, 50.0),     # Gulf of Aden
            (15.0, 58.0),     # Arabian Sea
            (19.0, 72.8),     # Mumbai, India
        ],
        "bounding_boxes": [
            (30.0, 35.0, 30.0, 35.0),     # Suez Canal zone
            (12.0, 30.0, 32.0, 44.0),     # Red Sea
            (10.0, 15.0, 42.0, 52.0),     # Gulf of Aden / Bab-el-Mandeb
            (10.0, 25.0, 50.0, 75.0),     # Arabian Sea
        ],
        "countries": [250, 380, 300, 818, 682, 887, 356, 512],  # France, Italy, Greece, Egypt, Saudi, Yemen, India, Oman
        "chokepoints": ["Suez Canal", "Bab-el-Mandeb Strait"],
        "distance_nm": 6200,
    },

    "europe_india_cape": {
        "name": "Europe → India (Cape of Good Hope)",
        "short_name": "Cape Route",
        "description": "Atlantic → Cape of Good Hope → Indian Ocean → India",
        "color": "#4ECDC4",
        "icon": "🌊",
        "waypoints": [
            (43.0, 5.0),       # Mediterranean (France)
            (36.0, -6.0),      # Strait of Gibraltar
            (28.0, -15.0),     # Canary Islands
            (15.0, -17.5),     # West Africa
            (0.0, -5.0),       # Gulf of Guinea
            (-15.0, 5.0),      # South Atlantic
            (-34.5, 18.5),     # Cape of Good Hope
            (-30.0, 35.0),     # Mozambique Channel
            (-15.0, 50.0),     # Madagascar
            (0.0, 60.0),       # Central Indian Ocean
            (10.0, 72.0),      # Arabian Sea
            (19.0, 72.8),      # Mumbai, India
        ],
        "bounding_boxes": [
            (35.0, 37.0, -7.0, -5.0),     # Gibraltar
            (-36.0, -30.0, 15.0, 25.0),    # Cape of Good Hope
            (-35.0, 0.0, 25.0, 55.0),      # East African coast / Mozambique
            (0.0, 20.0, 50.0, 75.0),       # Arabian Sea approach
        ],
        "countries": [250, 724, 710, 508, 450, 356],  # France, Spain, S.Africa, Mozambique, Madagascar, India
        "chokepoints": ["Strait of Gibraltar", "Cape of Good Hope"],
        "distance_nm": 10800,
    },

    "usa_west_east_asia": {
        "name": "USA West Coast → East Asia",
        "short_name": "Transpacific",
        "description": "Los Angeles/Long Beach → Pacific → China/Japan/Korea/Taiwan",
        "color": "#45B7D1",
        "icon": "🏗️",
        "waypoints": [
            (33.7, -118.2),    # LA/Long Beach
            (35.0, -140.0),    # Mid Pacific
            (35.0, -160.0),    # Central Pacific
            (35.0, 140.0),     # Japan approach
            (31.2, 121.5),     # Shanghai
            (22.3, 114.2),     # Hong Kong
            (25.0, 121.5),     # Taiwan
        ],
        "bounding_boxes": [
            (20.0, 45.0, 120.0, 145.0),   # East Asian waters
            (15.0, 30.0, 105.0, 125.0),    # South China Sea
        ],
        "countries": [840, 392, 156, 410, 158],  # USA, Japan, China, S.Korea, Taiwan
        "chokepoints": ["Pacific Shipping Lanes"],
        "distance_nm": 6500,
    },

    "usa_east_europe": {
        "name": "USA East Coast → Europe",
        "short_name": "Transatlantic",
        "description": "New York / Norfolk → North Atlantic → Rotterdam / Hamburg",
        "color": "#96CEB4",
        "icon": "🗽",
        "waypoints": [
            (40.7, -74.0),     # New York
            (42.0, -55.0),     # Mid Atlantic
            (48.0, -30.0),     # Central Atlantic
            (50.0, -5.0),      # English Channel approach
            (51.9, 4.5),       # Rotterdam
        ],
        "bounding_boxes": [
            (38.0, 55.0, -75.0, -5.0),    # North Atlantic corridor
            (48.0, 55.0, -5.0, 10.0),      # English Channel / North Sea
        ],
        "countries": [840, 826, 528, 276, 250],  # USA, UK, Netherlands, Germany, France
        "chokepoints": ["English Channel"],
        "distance_nm": 3500,
    },

    "usa_east_india_suez": {
        "name": "USA East Coast → India (Suez)",
        "short_name": "USA-India Suez",
        "description": "New York → Atlantic → Mediterranean → Suez → India",
        "color": "#F7DC6F",
        "icon": "🌎",
        "waypoints": [
            (40.7, -74.0),     # New York
            (36.0, -6.0),      # Gibraltar
            (35.0, 15.0),      # Central Med
            (31.5, 32.3),      # Suez
            (12.5, 43.5),      # Bab-el-Mandeb
            (15.0, 58.0),      # Arabian Sea
            (19.0, 72.8),      # Mumbai
        ],
        "bounding_boxes": [
            (30.0, 40.0, -75.0, -5.0),    # Atlantic crossing
            (30.0, 37.0, -6.0, 35.0),      # Mediterranean
            (30.0, 32.0, 30.0, 35.0),      # Suez Canal
            (12.0, 30.0, 32.0, 44.0),      # Red Sea
            (10.0, 15.0, 42.0, 52.0),      # Bab-el-Mandeb
            (10.0, 25.0, 50.0, 75.0),      # Arabian Sea
        ],
        "countries": [840, 724, 818, 682, 887, 356],  # USA, Spain, Egypt, Saudi, Yemen, India
        "chokepoints": ["Strait of Gibraltar", "Suez Canal", "Bab-el-Mandeb Strait"],
        "distance_nm": 8500,
    },

    "east_asia_india_malacca": {
        "name": "East Asia → India (Malacca Strait)",
        "short_name": "Malacca Route",
        "description": "China/Japan/Korea → South China Sea → Malacca Strait → Indian Ocean → India",
        "color": "#E74C3C",
        "icon": "⚓",
        "waypoints": [
            (31.2, 121.5),     # Shanghai
            (22.3, 114.2),     # Hong Kong
            (10.0, 110.0),     # South China Sea
            (1.3, 103.8),      # Singapore / Malacca
            (5.0, 95.0),       # Andaman Sea
            (10.0, 80.0),      # Sri Lanka
            (13.0, 80.3),      # Chennai
        ],
        "bounding_boxes": [
            (0.0, 25.0, 100.0, 125.0),    # South China Sea & approaches
            (-2.0, 6.0, 95.0, 108.0),     # Malacca Strait
            (0.0, 15.0, 75.0, 95.0),      # Bay of Bengal / Andaman
        ],
        "countries": [156, 392, 410, 702, 458, 360, 356, 764],  # China, Japan, S.Korea, Singapore, Malaysia, Indonesia, India, Thailand
        "chokepoints": ["Strait of Malacca", "South China Sea"],
        "distance_nm": 4600,
    },

    "persian_gulf_india_hormuz": {
        "name": "Persian Gulf → India (Strait of Hormuz)",
        "short_name": "Hormuz Route",
        "description": "Persian Gulf → Strait of Hormuz → Arabian Sea → Western India",
        "color": "#FF9F43",
        "icon": "🛢️",
        "waypoints": [
            (29.0, 48.0),      # Kuwait / Persian Gulf
            (26.5, 52.0),      # Central Gulf
            (26.0, 56.5),      # Strait of Hormuz
            (24.0, 58.5),      # Gulf of Oman
            (22.0, 63.0),      # Arabian Sea
            (19.0, 72.8),      # Mumbai
        ],
        "bounding_boxes": [
            (24.0, 30.0, 48.0, 57.0),     # Persian Gulf
            (25.0, 27.0, 55.0, 58.0),      # Strait of Hormuz
            (18.0, 26.0, 56.0, 73.0),      # Arabian Sea to India
        ],
        "countries": [364, 414, 634, 784, 512, 356],  # Iran, Kuwait, Qatar, UAE, Oman, India
        "chokepoints": ["Strait of Hormuz"],
        "distance_nm": 1500,
    },

    "iran_india_chabahar": {
        "name": "Iran → India (Chabahar)",
        "short_name": "Chabahar Route",
        "description": "Chabahar Port → Arabian Sea → Western India",
        "color": "#A29BFE",
        "icon": "🏭",
        "waypoints": [
            (25.3, 60.6),      # Chabahar Port
            (24.0, 63.0),      # Arabian Sea
            (22.0, 66.0),      # Mid Arabian Sea
            (19.0, 72.8),      # Mumbai
        ],
        "bounding_boxes": [
            (24.0, 26.0, 58.0, 62.0),     # Chabahar area
            (18.0, 25.0, 60.0, 73.0),      # Arabian Sea corridor
        ],
        "countries": [364, 356],  # Iran, India
        "chokepoints": ["Chabahar Port"],
        "distance_nm": 800,
    },

    "red_sea_corridor": {
        "name": "Red Sea Corridor",
        "short_name": "Red Sea",
        "description": "Suez Canal → Red Sea → Bab-el-Mandeb → Gulf of Aden",
        "color": "#FD79A8",
        "icon": "🔴",
        "waypoints": [
            (31.5, 32.3),      # Suez Canal
            (27.0, 34.0),      # Northern Red Sea
            (22.0, 37.0),      # Central Red Sea
            (16.0, 41.0),      # Southern Red Sea
            (12.5, 43.5),      # Bab-el-Mandeb
            (12.0, 45.0),      # Gulf of Aden entrance
        ],
        "bounding_boxes": [
            (12.0, 32.0, 32.0, 44.0),     # Entire Red Sea corridor
            (10.0, 14.0, 42.0, 52.0),      # Bab-el-Mandeb + Gulf of Aden
        ],
        "countries": [818, 682, 736, 232, 262, 887],  # Egypt, Saudi, Sudan, Eritrea, Djibouti, Yemen
        "chokepoints": ["Suez Canal", "Bab-el-Mandeb Strait"],
        "distance_nm": 1200,
    },

    "india_sri_lanka": {
        "name": "India ↔ Sri Lanka (Palk Strait)",
        "short_name": "Palk Strait",
        "description": "Southeast India → Palk Strait → Sri Lanka",
        "color": "#00B894",
        "icon": "🏝️",
        "waypoints": [
            (13.0, 80.3),      # Chennai
            (10.5, 79.8),      # Palk Strait North
            (9.5, 79.5),       # Palk Strait
            (7.0, 79.8),       # Colombo
        ],
        "bounding_boxes": [
            (6.0, 14.0, 78.0, 82.0),      # Palk Strait & surroundings
        ],
        "countries": [356, 144],  # India, Sri Lanka
        "chokepoints": ["Palk Strait"],
        "distance_nm": 350,
    },

    "india_coastal": {
        "name": "India Coastal Shipping",
        "short_name": "India Coast",
        "description": "Western & Eastern Indian coastal shipping routes",
        "color": "#6C5CE7",
        "icon": "🇮🇳",
        "waypoints": [
            (8.5, 76.9),       # Kochi
            (12.9, 74.8),      # Mangalore
            (15.4, 73.8),      # Goa
            (19.0, 72.8),      # Mumbai
            (22.5, 70.0),      # Kandla
            (21.0, 87.0),      # Paradip
            (17.7, 83.3),      # Visakhapatnam
            (13.0, 80.3),      # Chennai
        ],
        "bounding_boxes": [
            (7.0, 24.0, 68.0, 88.0),      # Indian coastal waters
        ],
        "countries": [356],  # India
        "chokepoints": ["Indian Coastal Waters"],
        "distance_nm": 2500,
    },
}

# ─────────────────────────────────────────────────────────────
# COMMODITY DEFINITIONS
# Each commodity has route-specific risk multipliers
# reflecting supply chain sensitivity
# ─────────────────────────────────────────────────────────────

COMMODITIES = {
    "ev_batteries": {
        "name": "EV Batteries",
        "icon": "🔋",
        "description": "Lithium-ion batteries, cathode/anode materials, battery cells & packs",
        "key_origins": ["China (CATL, BYD)", "South Korea (LG, Samsung SDI)", "Japan (Panasonic)"],
        "key_materials": ["Lithium", "Cobalt", "Nickel", "Graphite", "Manganese"],
        "applicable_routes": [
            "east_asia_india_malacca",
            "usa_west_east_asia",
            "europe_india_suez",
            "red_sea_corridor",
            "usa_east_europe",
            "india_coastal",
        ],
        "route_sensitivity": {
            "east_asia_india_malacca": 2.8,
            "usa_west_east_asia": 2.2,
            "europe_india_suez": 1.4,
            "red_sea_corridor": 1.6,
            "usa_east_europe": 1.0,
            "india_coastal": 1.3,
        },
    },
    "semiconductors": {
        "name": "Semiconductors",
        "icon": "💽",
        "description": "Integrated circuits, chipsets, wafers, semiconductor manufacturing equipment",
        "key_origins": ["Taiwan (TSMC)", "South Korea (Samsung)", "Japan", "USA (Intel, Qualcomm)"],
        "key_materials": ["Silicon Wafers", "Rare Earth Metals", "Photoresists", "EUV Equipment"],
        "applicable_routes": [
            "usa_west_east_asia",
            "east_asia_india_malacca",
            "usa_east_europe",
            "usa_east_india_suez",
            "europe_india_suez",
            "red_sea_corridor",
        ],
        "route_sensitivity": {
            "usa_west_east_asia": 2.8,
            "east_asia_india_malacca": 2.5,
            "usa_east_europe": 2.0,
            "usa_east_india_suez": 1.4,
            "europe_india_suez": 1.0,
            "red_sea_corridor": 1.0,
        },
    },
    "solar_panels": {
        "name": "Solar Panels & PV Modules",
        "icon": "☀️",
        "description": "Photovoltaic cells, solar modules, inverters, mounting systems",
        "key_origins": ["China (LONGi, JA Solar, Trina)", "Vietnam", "Malaysia", "India (Adani)"],
        "key_materials": ["Polysilicon", "Silver Paste", "EVA Film", "Tempered Glass", "Junction Boxes"],
        "applicable_routes": [
            "east_asia_india_malacca",
            "usa_west_east_asia",
            "india_coastal",
            "red_sea_corridor",
            "europe_india_suez",
        ],
        "route_sensitivity": {
            "east_asia_india_malacca": 3.0,
            "usa_west_east_asia": 1.8,
            "india_coastal": 1.8,
            "red_sea_corridor": 1.2,
            "europe_india_suez": 1.0,
        },
    },
    "power_transformers": {
        "name": "Power Transformers & Grid Equipment",
        "icon": "⚡",
        "description": "High-voltage transformers, switchgear, circuit breakers, substations",
        "key_origins": ["Germany (Siemens)", "Switzerland (ABB/Hitachi)", "France (Schneider)", "China", "India (BHEL)"],
        "key_materials": ["CRGO Steel", "Copper Windings", "Transformer Oil", "Bushings", "Insulators"],
        "applicable_routes": [
            "europe_india_suez",
            "europe_india_cape",
            "red_sea_corridor",
            "usa_east_india_suez",
            "usa_east_europe",
            "east_asia_india_malacca",
            "persian_gulf_india_hormuz",
            "iran_india_chabahar",
            "india_coastal",
        ],
        "route_sensitivity": {
            "europe_india_suez": 2.8,
            "europe_india_cape": 2.0,
            "red_sea_corridor": 2.2,
            "usa_east_india_suez": 1.8,
            "usa_east_europe": 1.5,
            "east_asia_india_malacca": 1.2,
            "persian_gulf_india_hormuz": 1.6,
            "iran_india_chabahar": 1.4,
            "india_coastal": 1.5,
        },
    },
    "electric_motors": {
        "name": "Electric Motors & Drives",
        "icon": "⚙️",
        "description": "AC/DC motors, variable frequency drives, servo motors, traction motors",
        "key_origins": ["China (Nidec, Wolong)", "Japan (Nidec, Yaskawa)", "Germany (Siemens, Bosch)", "India"],
        "key_materials": ["Permanent Magnets", "Copper Wire", "Electrical Steel", "Bearings", "Encoders"],
        "applicable_routes": [
            "east_asia_india_malacca",
            "europe_india_suez",
            "usa_west_east_asia",
            "persian_gulf_india_hormuz",
            "iran_india_chabahar",
            "india_coastal",
            "red_sea_corridor",
            "india_sri_lanka",
        ],
        "route_sensitivity": {
            "east_asia_india_malacca": 2.2,
            "europe_india_suez": 2.0,
            "india_coastal": 2.0,
            "iran_india_chabahar": 1.8,
            "usa_west_east_asia": 1.5,
            "red_sea_corridor": 1.6,
            "persian_gulf_india_hormuz": 1.4,
            "india_sri_lanka": 1.2,
        },
    },
    "cables_wiring": {
        "name": "Cables & Wiring Harnesses",
        "icon": "🔌",
        "description": "Power cables, fiber optics, wiring harnesses, copper conductors, submarine cables",
        "key_origins": ["China", "India (Polycab, Havells)", "Turkey", "Italy (Prysmian)", "Japan (Sumitomo)"],
        "key_materials": ["Copper", "Aluminum", "PVC/XLPE Insulation", "Optical Fiber", "Steel Armor"],
        "applicable_routes": [
            "india_coastal",
            "europe_india_cape",
            "europe_india_suez",
            "east_asia_india_malacca",
            "india_sri_lanka",
            "persian_gulf_india_hormuz",
            "red_sea_corridor",
        ],
        "route_sensitivity": {
            "india_coastal": 2.8,
            "europe_india_cape": 2.2,
            "europe_india_suez": 1.8,
            "east_asia_india_malacca": 1.6,
            "india_sri_lanka": 1.5,
            "persian_gulf_india_hormuz": 1.2,
            "red_sea_corridor": 1.4,
        },
    },
    "led_lighting": {
        "name": "LED & Lighting Equipment",
        "icon": "💡",
        "description": "LED chips, luminaires, smart lighting systems, drivers, display panels",
        "key_origins": ["China (Cree, MLS)", "Taiwan (Epistar)", "South Korea (Seoul Semi)", "Japan (Nichia)"],
        "key_materials": ["GaN Wafers", "Phosphor", "Heat Sinks", "LED Drivers", "Sapphire Substrates"],
        "applicable_routes": [
            "east_asia_india_malacca",
            "usa_west_east_asia",
            "europe_india_suez",
            "red_sea_corridor",
            "india_coastal",
            "usa_east_europe",
        ],
        "route_sensitivity": {
            "east_asia_india_malacca": 2.8,
            "usa_west_east_asia": 1.8,
            "india_coastal": 1.4,
            "red_sea_corridor": 1.2,
            "europe_india_suez": 1.0,
            "usa_east_europe": 1.0,
        },
    },
    "crude_oil": {
        "name": "Crude Oil",
        "icon": "🛢️",
        "description": "Sour crude oil (high sulfur), heavy crude from Middle East, Africa, Russia",
        "key_origins": ["Saudi Arabia (Aramco)", "Iraq", "UAE", "Kuwait", "Nigeria", "Russia"],
        "key_materials": ["Heavy Crude", "Sour Crude", "Bitumen", "Condensate"],
        "applicable_routes": [
            "persian_gulf_india_hormuz",
            "europe_india_suez",
            "europe_india_cape",
            "red_sea_corridor",
            "east_asia_india_malacca",
            "india_coastal",
            "iran_india_chabahar",
        ],
        "route_sensitivity": {
            "persian_gulf_india_hormuz": 3.0,
            "europe_india_cape": 2.2,
            "red_sea_corridor": 2.5,
            "europe_india_suez": 2.0,
            "east_asia_india_malacca": 1.8,
            "india_coastal": 1.5,
            "iran_india_chabahar": 1.6,
        },
    },
    "sweet_crude_oil": {
        "name": "Sweet Crude Oil",
        "icon": "🏗️",
        "description": "Low-sulfur crude oil, premium grade (Brent, WTI, Bonny Light)",
        "key_origins": ["Nigeria (Bonny Light)", "Norway (Brent)", "USA (WTI)", "Libya", "Algeria"],
        "key_materials": ["Light Sweet Crude", "Brent Blend", "WTI", "Bonny Light"],
        "applicable_routes": [
            "europe_india_suez",
            "europe_india_cape",
            "red_sea_corridor",
            "usa_east_europe",
            "usa_east_india_suez",
            "india_coastal",
        ],
        "route_sensitivity": {
            "europe_india_cape": 2.8,
            "europe_india_suez": 2.5,
            "red_sea_corridor": 2.2,
            "usa_east_india_suez": 1.8,
            "usa_east_europe": 1.6,
            "india_coastal": 1.2,
        },
    },
    "lpg": {
        "name": "LPG (Liquefied Petroleum Gas)",
        "icon": "🔥",
        "description": "Propane, butane, LPG mix for domestic and industrial use",
        "key_origins": ["Saudi Arabia", "Qatar", "UAE", "USA", "Kuwait", "Iran"],
        "key_materials": ["Propane", "Butane", "Isobutane", "LPG Mix"],
        "applicable_routes": [
            "persian_gulf_india_hormuz",
            "red_sea_corridor",
            "europe_india_suez",
            "usa_east_india_suez",
            "india_coastal",
            "iran_india_chabahar",
        ],
        "route_sensitivity": {
            "persian_gulf_india_hormuz": 3.0,
            "red_sea_corridor": 2.0,
            "europe_india_suez": 1.6,
            "usa_east_india_suez": 1.8,
            "india_coastal": 2.2,
            "iran_india_chabahar": 1.4,
        },
    },
    "petrol": {
        "name": "Petrol (Gasoline)",
        "icon": "⛽",
        "description": "Motor spirit / gasoline for automotive and industrial use",
        "key_origins": ["India (Reliance, IOCL)", "Saudi Arabia", "Singapore", "South Korea", "UAE"],
        "key_materials": ["Motor Spirit", "Reformate", "Alkylate", "MTBE", "Ethanol Blend"],
        "applicable_routes": [
            "persian_gulf_india_hormuz",
            "east_asia_india_malacca",
            "india_coastal",
            "red_sea_corridor",
            "europe_india_suez",
        ],
        "route_sensitivity": {
            "persian_gulf_india_hormuz": 2.5,
            "east_asia_india_malacca": 2.0,
            "india_coastal": 2.8,
            "red_sea_corridor": 1.8,
            "europe_india_suez": 1.4,
        },
    },
    "diesel": {
        "name": "Diesel (High Speed Diesel)",
        "icon": "🚛",
        "description": "High speed diesel, gas oil for transport, agriculture and industry",
        "key_origins": ["India (Reliance, BPCL)", "Saudi Arabia", "Kuwait", "Singapore", "Russia"],
        "key_materials": ["HSD", "Gas Oil", "ULSD", "Bio-Diesel Blend"],
        "applicable_routes": [
            "persian_gulf_india_hormuz",
            "east_asia_india_malacca",
            "india_coastal",
            "red_sea_corridor",
            "europe_india_suez",
            "europe_india_cape",
        ],
        "route_sensitivity": {
            "persian_gulf_india_hormuz": 2.8,
            "east_asia_india_malacca": 1.8,
            "india_coastal": 2.5,
            "red_sea_corridor": 2.0,
            "europe_india_suez": 1.6,
            "europe_india_cape": 1.4,
        },
    },
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km using Haversine formula."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def is_point_in_route_corridor(lat, lon, route_key, buffer_km=500):
    """Check if a point (lat, lon) falls within a route's bounding boxes or near its waypoints."""
    route = ROUTES[route_key]

    # Check bounding boxes first (fast)
    for (lat_min, lat_max, lon_min, lon_max) in route["bounding_boxes"]:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    
    # Check proximity to waypoints (slower, more precise)
    for wp_lat, wp_lon in route["waypoints"]:
        dist = haversine_km(lat, lon, wp_lat, wp_lon)
        if dist <= buffer_km:
            return True
    
    return False


def get_route_country_codes(route_key):
    """Get country codes associated with a route for sanctions lookup."""
    return ROUTES[route_key].get("countries", [])


def get_all_route_keys():
    """Get list of all route identifiers."""
    return list(ROUTES.keys())


def get_all_commodity_keys():
    """Get list of all commodity identifiers."""
    return list(COMMODITIES.keys())


def get_commodity_sensitivity(commodity_key, route_key):
    """Get the risk sensitivity multiplier for a commodity on a specific route."""
    commodity = COMMODITIES.get(commodity_key, {})
    sensitivities = commodity.get("route_sensitivity", {})
    return sensitivities.get(route_key, 1.0)
