GAME_NAME = "Sekiro"

ACTION_DELAY = 0.25
REVIVE_DELAY = 2

# FIXME: change accordingly
AGENT_KEYMAP = {
    "attack": "j",
    "defense": "k",
    "dodge": "shift",
    "jump": "space",
    "left_dodge": ("a", "shift"),
    "right_dodge": ('d', "shift"),
    "back_dodge": ('s', "shift")
}

ENV_KEYMAP = {
    "pause": "esc",
    "resume": "esc",
    "revive": "j",
    "focus": "l",
    "switch_full_blood": "1",
    "switch_visible": ".",
    "stop_dqn": "t",
    "switch_invincible": "2",
}

# HACK: (left, top, right, bottom)
SCREEN_SIZE = (720, 1280)
SCREEN_ANCHOR = (1, -721, -1, -1)

FOCUS_ANCHOR = (392, 108, 892, 608)
FOCUS_SIZE = (224, 224)

AGENT_HP_ANCHOR = (75, 651, 370, 658)
BOSS_HP_ANCHOR = (75, 62, 348, 71)

AGENT_EP_ANCHOR = (641, 622, 770, 626)
BOSS_EP_ANCHOR = (641, 42, 867, 48)


SELF_BLOOD_HEIGHT = (682-31, 690-32)
SELF_BLOOD_WIDTH = (83-8, 382-8)

SELF_ENDURANCE_HEIGHT = (653-32, 658-32)
SELF_ENDURANCE_WIDTH = (653-8, 778-8)

BOSS_BLOOD_HEIGHT = (94-32, 102-32)
BOSS_BLOOD_WIDTH = (75, 358-8)

BOSS_ENDURANCE_HEIGHT = (74-32, 81-32)
BOSS_ENDURANCE_WIDTH = (653-8, 878-8)


AGENT_DEAD_DELAY = 8
ROTATION_DELAY = 1
REVIVE_DELAY = 2.2
PAUSE_DELAY = 0.8


# <------

# ------> code injection
MIN_CODE_LEN = 6
MIN_HELPER_LEN = 13
# <------

# ------> agent attributes
MAX_AGENT_HP = 800
MAX_AGENT_EP = 300

MAX_BOSS_HP = 9887
MAX_BOSS_EP = 4083

MAP_CENTER = (-110.252, 54.077, 239.538)
# <------