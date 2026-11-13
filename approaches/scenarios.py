"""The three scenarios used throughout the examples and the guide."""
from __future__ import annotations

ZONES = [{"id": "red_zone", "color": "red", "capacity": 1},
         {"id": "blue_zone", "color": "blue", "capacity": 1},
         {"id": "green_zone", "color": "green", "capacity": 1}]

# 1. A blue cylinder stands on the red cube, and the blue zone is occupied by a
#    misplaced red prism. The cylinder must wait in a temporary slot.
BLOCKED = {
    "scene_id": "blocked", "scene_version": 1,
    "objects": [
        {"id": "red_cube_1", "type": "cube", "color": "red", "on": "table", "confidence": 0.96},
        {"id": "blue_cylinder_1", "type": "cylinder", "color": "blue", "on": "red_cube_1",
         "confidence": 0.91},
        {"id": "red_prism_1", "type": "prism", "color": "red", "on": "blue_zone", "confidence": 0.93},
    ],
    "zones": ZONES, "free_slots": ["tmp_1"],
}

# 2. The wrist camera cannot tell the colour of the object on top. The laptop sends
#    its class probabilities; the server decides whether it must look again.
OCCLUDED = {
    "scene_id": "occluded", "scene_version": 1,
    "objects": [
        {"id": "red_cube_1", "type": "cube", "color": "red", "on": "table", "confidence": 0.95},
        {"id": "unknown_1", "type": "cylinder", "color": None, "on": "red_cube_1", "confidence": 0.4,
         "color_probs": {"blue": 0.55, "green": 0.45}},
        {"id": "green_cube_1", "type": "cube", "color": "green", "on": "table", "confidence": 0.97},
    ],
    "zones": ZONES, "free_slots": ["tmp_1"],
}
OCCLUDED_RESOLVED = {            # what the external camera reports after REQUEST_NEW_VIEW
    **OCCLUDED, "scene_version": 2,
    "objects": [OCCLUDED["objects"][0],
                {"id": "unknown_1", "type": "cylinder", "color": "blue", "on": "red_cube_1",
                 "confidence": 0.93},
                OCCLUDED["objects"][2]],
}

# 3. During execution a person puts the red cube on the green cube.
CHANGE_BEFORE = {
    "scene_id": "change", "scene_version": 1,
    "objects": [
        {"id": "red_cube_1", "type": "cube", "color": "red", "on": "table"},
        {"id": "blue_cylinder_1", "type": "cylinder", "color": "blue", "on": "table"},
        {"id": "green_cube_1", "type": "cube", "color": "green", "on": "table"},
    ],
    "zones": ZONES, "free_slots": ["tmp_1"],
}
CHANGE_AFTER = {                 # blue cylinder already delivered, red cube now on the green cube
    **CHANGE_BEFORE, "scene_version": 2,
    "objects": [
        {"id": "red_cube_1", "type": "cube", "color": "red", "on": "green_cube_1"},
        {"id": "blue_cylinder_1", "type": "cylinder", "color": "blue", "on": "blue_zone"},
        {"id": "green_cube_1", "type": "cube", "color": "green", "on": "table"},
    ],
}


# 4. A scene whose plan has real freedom of order (used for the partial-order diagram).
DAG_EXAMPLE = {
    "scene_id": "dag", "scene_version": 1,
    "objects": [
        {"id": "blue_A", "type": "cube", "color": "blue", "on": "green_1"},
        {"id": "red_prism", "type": "prism", "color": "red", "on": "table"},
        {"id": "green_1", "type": "cube", "color": "green", "on": "red_zone"},
        {"id": "red_A", "type": "cube", "color": "red", "on": "blue_B"},
        {"id": "red_B", "type": "cube", "color": "red", "on": "table"},
        {"id": "blue_B", "type": "cube", "color": "blue", "on": "red_B"},
    ],
    "zones": [{"id": "red_zone", "color": "red", "capacity": 3},
              {"id": "blue_zone", "color": "blue", "capacity": 1},
              {"id": "green_zone", "color": "green", "capacity": 2}],
    "free_slots": ["tmp_1"],
}
