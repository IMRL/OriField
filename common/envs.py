import os


PREPROCESS_SCANS_TRAINING_ACC = os.getenv('PREPROCESS_SCANS_TRAINING_ACC')
PREPROCESS_SCANS_TRAINING_CROP = os.getenv('PREPROCESS_SCANS_TRAINING_CROP')

LOADER_VIS = False if os.getenv('LOADER_VIS') is None else True

DEPLOY_SEQS = os.getenv('DEPLOY_SEQS')
DEPLOY_SAVE = False if os.getenv('DEPLOY_SAVE') is None else True

FAILURE_SAVE = False if os.getenv('FAILURE_SAVE') is None else True
FAILURE_VIS = False if os.getenv('FAILURE_VIS') is None else True

DEBUG = False if os.getenv('DEBUG') is None else True
DEBUG_START = 0 if not DEBUG else 3376
DEBUG_VIS = False if not DEBUG else True

DEMO_VIS = False if os.getenv('DEMO_VIS') is None else True
DEMO_SAVE = False if os.getenv('DEMO_SAVE') is None else True

MINISET = False if os.getenv('MINISET') is None else True