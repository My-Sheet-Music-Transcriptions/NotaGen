import os
import json
import wandb
import torch
import random
import numpy as np
from utils import *
from config import *
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset
from transformers import GPT2Config


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

patchilizer = Patchilizer()

patch_config = GPT2Config(
    num_hidden_layers=PATCH_NUM_LAYERS,
    max_length=PATCH_LENGTH,
    max_position_embeddings=PATCH_LENGTH,
    n_embd=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=1,
)
char_config = GPT2Config(
    num_hidden_layers=CHAR_NUM_LAYERS,
    max_length=PATCH_SIZE + 1,
    max_position_embeddings=PATCH_SIZE + 1,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=128,
)

model_ref = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)
model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)


model_ref = model_ref.to(device)
model = model.to(device)


# print parameter number
print(
    "Parameter Number: "
    + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def collate_batch(input_batches):
    pos_input_patches, pos_input_masks, neg_input_patches, neg_input_masks = (
        input_batches
    )
    pos_input_patches = pos_input_patches.unsqueeze(0)
    pos_input_masks = pos_input_masks.unsqueeze(0)
    neg_input_patches = neg_input_patches.unsqueeze(0)
    neg_input_masks = neg_input_masks.unsqueeze(0)
    pos_input_patches = torch.nn.utils.rnn.pad_sequence(
        pos_input_patches, batch_first=True, padding_value=0
    )
    pos_input_masks = torch.nn.utils.rnn.pad_sequence(
        pos_input_masks, batch_first=True, padding_value=0
    )
    neg_input_patches = torch.nn.utils.rnn.pad_sequence(
        neg_input_patches, batch_first=True, padding_value=0
    )
    neg_input_masks = torch.nn.utils.rnn.pad_sequence(
        neg_input_masks, batch_first=True, padding_value=0
    )
    return (
        pos_input_patches.to(device),
        pos_input_masks.to(device),
        neg_input_patches.to(device),
        neg_input_masks.to(device),
    )


class NotaGenDataset(Dataset):
    def __init__(self, preference_dict):
        self.preference_dict = preference_dict
        self.pair_list = []
        for pos_filepath in self.preference_dict["chosen"]:
            for neg_filepath in self.preference_dict["rejected"]:
                self.pair_list.append(
                    {"chosen": pos_filepath, "rejected": neg_filepath}
                )

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        try:
            pair = self.pair_list[idx]
            pos_filepath = pair["chosen"]
            neg_filepath = pair["rejected"]

            with open(pos_filepath, "r", encoding="utf-8") as f:
                pos_abc_text = f.read()
            with open(neg_filepath, "r", encoding="utf-8") as f:
                neg_abc_text = f.read()

            pos_file_bytes = patchilizer.encode(pos_abc_text)
            pos_file_masks = [1] * len(pos_file_bytes)
            neg_file_bytes = patchilizer.encode(neg_abc_text)
            neg_file_masks = [1] * len(neg_file_bytes)

            pos_file_bytes = torch.tensor(pos_file_bytes, dtype=torch.long)
            pos_file_masks = torch.tensor(pos_file_masks, dtype=torch.long)
            neg_file_bytes = torch.tensor(neg_file_bytes, dtype=torch.long)
            neg_file_masks = torch.tensor(neg_file_masks, dtype=torch.long)

            return pos_file_bytes, pos_file_masks, neg_file_bytes, neg_file_masks
        except Exception as e:
            print(e)
            return self.__getitem__((idx + 1) % len(self.pair_list))


def process_one_batch(batch):
    pos_input_patches, pos_input_masks, neg_input_patches, neg_input_masks = batch
    pos_input_patches_ref = pos_input_patches.clone()
    pos_input_masks_ref = pos_input_masks.clone()
    neg_input_patches_ref = neg_input_patches.clone()
    neg_input_masks_ref = neg_input_masks.clone()
    policy_pos_logps = model(pos_input_patches, pos_input_masks)
    policy_neg_logps = model(neg_input_patches, neg_input_masks)
    with torch.no_grad():
        ref_pos_logps = model_ref(pos_input_patches_ref, pos_input_masks_ref).detach()
        ref_neg_logps = model_ref(neg_input_patches_ref, neg_input_masks_ref).detach()
    logits = (policy_pos_logps - policy_neg_logps) - (ref_pos_logps - ref_neg_logps)
    loss = -torch.nn.functional.logsigmoid(
        BETA * (logits - LAMBDA * max(0, ref_pos_logps - policy_pos_logps))
    )
    return loss


# train
if __name__ == "__main__":
    # Initialize wandb
    if WANDB_LOGGING:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="notagen", name=WANDB_NAME)

    # load data
    with open(DATA_INDEX_PATH, "r") as f:
        preference_dict = json.loads(f.read())

    train_set = NotaGenDataset(preference_dict)

    # Load model actor/ref
    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu")
        cpu_model = deepcopy(model)
        cpu_model.load_state_dict(checkpoint["model"])
        model.load_state_dict(cpu_model.state_dict())
        cpu_model_ref = deepcopy(model_ref)
        cpu_model_ref.load_state_dict(checkpoint["model"])
        model_ref.load_state_dict(cpu_model_ref.state_dict())
    else:
        raise Exception("No pre-trained model loaded.")

    model.train()
    total_train_loss = 0
    iter_idx = 1

    tqdm_set = tqdm(range(OPTIMIZATION_STEPS))
    for i in tqdm_set:
        idx = random.randint(0, len(train_set) - 1)
        batch = train_set[idx]
        batch = collate_batch(batch)

        loss = process_one_batch(batch)
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.zero_grad(set_to_none=True)
        tqdm_set.set_postfix({"train_loss": total_train_loss / (i + 1)})

        # Log the training loss to wandb
        if WANDB_LOGGING:
            wandb.log({"train_loss": total_train_loss / (i + 1)}, step=i + 1)

    checkpoint = {
        "model": model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    }

    torch.save(checkpoint, WEIGHTS_PATH)
