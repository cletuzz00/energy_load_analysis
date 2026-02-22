"""
Configuration for Pecan Street data paths and circuit definitions.
"""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data paths by region
REGIONS = {
    "austin": {
        "data_path": PROJECT_ROOT / "datasets" / "15minute_data_austin" / "15minute_data_austin.csv",
        "metadata_path": PROJECT_ROOT / "datasets" / "metadata.csv",  # Austin metadata
    },
    "california": {
        "data_path": PROJECT_ROOT / "datasets" / "15minute_data_california" / "15minute_data_california.csv",
        "metadata_path": PROJECT_ROOT / "datasets" / "15minute_data_california" / "metadata.csv",
    },
    "newyork": {
        "data_path": PROJECT_ROOT / "datasets" / "15minute_data_newyork" / "15minute_data_newyork.csv",
        "metadata_path": PROJECT_ROOT / "datasets" / "15minute_data_newyork" / "metadata.csv",
    }
    #TODO puerto_rico data and metadata 
}

# Processed output
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Whole-home power column (net grid draw; can be negative with solar export)
WHOLE_HOME_COLUMN = "grid"

# All circuit columns that represent real power (kW); exclude id, time, voltage
NON_POWER_COLUMNS = {"dataid", "local_15min", "leg1v", "leg2v"}

# All circuit columns in Pecan Street 15-min data (HVAC, appliances, lights, pool, etc.)
# Not all homes have all circuits.
ALL_CIRCUIT_COLUMNS = [
    # HVAC / cooling
    "air1",
    "air2",
    "air3",
    "airwindowunit1",
    "housefan1",
    "furnace1",
    "furnace2",
    # Heating / water heating
    "heater1",
    "heater2",
    "heater3",
    "waterheater1",
    "waterheater2",
    # EV / storage / generation
    "battery1",
    "car1",
    "car2",
    "grid",
    "solar",
    "solar2",
    # Appliances
    "aquarium1",
    "clotheswasher1",
    "clotheswasher_dryg1",
    "dishwasher1",
    "disposal1",
    "drye1",
    "dryg1",
    "freezer1",
    "icemaker1",
    "microwave1",
    "oven1",
    "oven2",
    "range1",
    "refrigerator1",
    "refrigerator2",
    "venthood1",
    "winecooler1",
    # Rooms (lights/plugs)
    "bathroom1",
    "bathroom2",
    "bedroom1",
    "bedroom2",
    "bedroom3",
    "bedroom4",
    "bedroom5",
    "diningroom1",
    "diningroom2",
    "garage1",
    "garage2",
    "kitchen1",
    "kitchen2",
    "kitchenapp1",
    "kitchenapp2",
    "lights_plugs1",
    "lights_plugs2",
    "lights_plugs3",
    "lights_plugs4",
    "lights_plugs5",
    "lights_plugs6",
    "livingroom1",
    "livingroom2",
    "office1",
    "outsidelights_plugs1",
    "outsidelights_plugs2",
    "utilityroom1",
    # Pool / pumps / other
    "circpump1",
    "jacuzzi1",
    "pool1",
    "pool2",
    "poollight1",
    "poolpump1",
    "pump1",
    "security1",
    "sewerpump1",
    "shed1",
    "sprinkler1",
    "sumppump1",
    "wellpump1",
]

# Metadata columns we need after normalization (common schema)
METADATA_KEY_COLUMNS = [
    "dataid",
    "building_type",
    "total_square_footage",
    "house_construction_year",
    "pv",
    "car1",
    "car2",
    "battery1",
    "solar",
    "energy_storage_system",
    "heater1",
    "heater2",
    "heater3",
    "egauge_1min_min_time",
    "egauge_1min_max_time",
    "date_enrolled",
    "date_withdrawn",
]

# Circuits that can have negative power (generation or export)
CIRCUITS_ALLOW_NEGATIVE = {"grid", "solar", "solar2", "battery1"}
