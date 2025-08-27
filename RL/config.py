# Configuration for the data
DATA_INDEX_PATH = ""

# Configuration for the model
PATCH_STREAM = True
PATCH_SIZE = 16  # Patch Size
PATCH_LENGTH = 1024  # Patch Length
CHAR_NUM_LAYERS = 6  # Number of layers in the decoder
PATCH_NUM_LAYERS = 20  # Number of layers in the encoder
HIDDEN_SIZE = 1280  # Hidden Size

# Configuration for the training
BETA = 0.1  # beta in DPO's objective function
LAMBDA = 10  # lambda in DPOP's objective function
LEARNING_RATE = 1e-6
OPTIMIZATION_STEPS = 10000  # Optimization steps for DPO
WANDB_LOGGING = False  # Whether to log to wandb
WANDB_KEY = "<your_wandb_key>"

PRETRAINED_PATH = ""
EXP_TAG = ""
NAME = (
    EXP_TAG
    + "_beta_"
    + str(BETA)
    + "_lambda_"
    + str(LAMBDA)
    + "_p_size_"
    + str(PATCH_SIZE)
    + "_p_length_"
    + str(PATCH_LENGTH)
    + "_p_layers_"
    + str(PATCH_NUM_LAYERS)
    + "_c_layers_"
    + str(CHAR_NUM_LAYERS)
    + "_h_size_"
    + str(HIDDEN_SIZE)
    + "_lr_"
    + str(LEARNING_RATE)
)

WEIGHTS_PATH = "weights_notagen_" + NAME + ".pth"  # Path to save weights
WANDB_NAME = NAME
