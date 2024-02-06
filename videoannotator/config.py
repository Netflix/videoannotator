from fractions import Fraction

PATH_DATA_BASE = "/root/data/videoannotator"

SEED = 0
SCORE_THRESHOLD = 0.5
N_SPLITS = 5
N_BOOTSTRAP = 5
N_JOBS = -1
MAX_ITER = 1_000
VALIDATION_FRACTION = Fraction(numerator=2, denominator=10)
MIN_TRAINING_POS_RATE = 0.1
CLUSTER_CNT = 10
ANNOTATOR_CNT = 3

# AVE experiments
AVE_EXPERIMENTS_NS = (25, 50, 100, 500, 1000)
AVE_EXPERIMENTS_METHODS = (
    "random",
    "zero-shot-50-random-50",
    "zero-shot",
)
SCORING = (
    "average_precision",
    # "balanced_accuracy",
)

LABEL_GROUPS = dict(
    motion={
        "zoom",
        "pan",
        "slow-motion",
        "handheld",
        # "fast-motion",
        "timelapse",
        "jump-scare",
    },
    genres={
        "action",
        "sci-fi",
        "horror",
        "fantasy",
        "drama",
        "romance",
    },
    emotions={
        "anger",
        "sad",
        "happy",
        "scared",
        "laughter",
    },
    shot_types={
        "establishing-shots",
        "static-shot",
        "shutter-shot",
        "cowboy-shot",
        "extreme-close-up",
        "extreme-wide-shot",
        "two-shot",
        "group-shot",
        "aerial",
        "eye-level",
        "medium",
        "closeup",
        "wide",
        "over-the-shoulder-shot",
        "tilt-shot",
        "dutch-angle",
        "point-of-view-shot",
        "high-angle",
        "insert-shot",
        "low-angle",
        "overhead-shot",
        "single-shot",
    },
    sensitivities={
        "alcohol",
        "smoking",
        "nudity",
        "gore",
        "drugs",
        "violence",
        "intimacy",
    },
    events_actions={
        "interview",
        "fight",
        "car-chase",
        "run",
    },
    time_location={
        "day",
        "golden-hour",
        "interior",
    },
    focus={
        "character-focus",
        "animal",
        "object",
    },
)

LABELS = frozenset(
    (
        "action",
        "smoking",
        "nudity",
        "gore",
        "sad",
        "happy",
        "intimacy",
        "establishing-shots",
        "character-focus",
        "sci-fi",
        "horror",
        "alcohol",
        "anger",
        "aerial",
        "eye-level",
        "medium",
        "closeup",
        "wide",
        "zoom",
        "pan",
        "slow-motion",
        "jump-scare",
        "timelapse",
        "interior",
        "overhead-shot",
        # "fast-motion",  # very low pos rate
        "animal",
        "group-shot",
        "drugs",
        "car-chase",
        "laughter",
        "over-the-shoulder-shot",
        "day",
        "object",
        "handheld",
        "low-angle",
        "violence",
        "drama",
        "point-of-view-shot",
        "single-shot",
        "romance",
        "golden-hour",
        "extreme-wide-shot",
        "high-angle",
        "insert-shot",
        "run",
        "fantasy",
        "static-shot",
        "shutter-shot",
        "cowboy-shot",
        "extreme-close-up",
        "two-shot",
        "fight",
        "interview",
        "scared",
        "tilt-shot",
        "dutch-angle",
    )
)
LABELS_AVE = frozenset(
    (
        "group-shot",
        "animal",
        "single-shot",
        "low-angle",
        "eye-level",
        "medium",
        "extreme-wide-shot",
        "tilt-shot",
        "high-angle",
        "over-the-shoulder-shot",
        "zoom",
        "closeup",
        "wide",
        "handheld",
        "overhead-shot",
        "insert-shot",
        "two-shot",
        "extreme-close-up",
        "aerial",
    )
)
