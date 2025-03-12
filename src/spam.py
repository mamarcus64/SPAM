import pytorch_lightning as pl
import sys
import torch
import pdb
import numpy as np
import os
import json
import random
from tqdm import tqdm
import scipy.ndimage
from torch.cuda.amp import autocast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../segment-anything-2')))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Spam(pl.LightningModule):
    """
    Extends SAM (Segment Anything Model) for fine-tuning on video frames.
    """

    # TODO: have better argument parser
    def __init__(self, config_file: str, original_checkpoint: str, spam_checkpoint
                #  save_folder: str
                 ):
        """
        Initializes the Spam model.
        """
        super().__init__()
        self.cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_logs = []
        self.loss_logs_history = []
        self.checkpoint_file = spam_checkpoint
        self.config_file = config_file
        self.model = build_sam2(config_file, original_checkpoint, device=self.cuda_device)
        self.predictor = SAM2ImagePredictor(self.model)
        
        self.model.load_state_dict(
            torch.load(spam_checkpoint, map_location=self.cuda_device)
        )
        
        # TODO: move num_steps counting from this class to normal PyTorchLightning setup (also evaluate on val set + report loss stats during training)
        self.num_steps = 0

        # setup save folder
        # self.save_folder = save_folder
        # self.checkpoint_save_folder = os.path.join(save_folder, "checkpoints")
        # self.log_save_folder = os.path.join(save_folder, "logs")
        # self.eval_save_folder = os.path.join(save_folder, "eval_logs")
        # os.makedirs(self.checkpoint_save_folder, exist_ok=True)
        # os.makedirs(self.log_save_folder, exist_ok=True)
        # os.makedirs(self.eval_save_folder, exist_ok=True)

        # Train mask decoder.
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.predictor.model.parameters(), lr=0.0001, weight_decay=1e-4
        )
        return optimizer

    def save(self, reset_loss_logs=True):
        # skip saving if we already saved
        # if os.path.exists(os.path.join(self.log_save_folder, f'step_{self.num_steps}.json')):
        # return
        print(f"Saving logs to {self.log_save_folder} (Step {self.num_steps})...")
        log_file = os.path.join(self.log_save_folder, f"step_{self.num_steps}.json")
        print(
            f"Saving checkpoint to {self.checkpoint_save_folder} (Step {self.num_steps})..."
        )
        checkpoint_file = os.path.join(
            self.checkpoint_save_folder, f"step_{self.num_steps}.pt"
        )

        json.dump(self.loss_logs, open(log_file, "w"), indent=4)
        torch.save(self.predictor.model.state_dict(), checkpoint_file)

        # reset log tracker
        if reset_loss_logs:
            self.loss_logs = []

    def training_step(self, batch, batch_idx):
        assert (
            len(batch) == 1
        ), "Batching currently not supported. Set to batch size of 1."

        # TODO: make the num_steps save frequency a class variable
        if self.num_steps % 1320 == 0:
            self.save()
            eval_logs = self.eval(self.eval_frames)
            eval_file = os.path.join(
                self.eval_save_folder, f"step_{self.num_steps}.json"
            )
            json.dump(eval_logs, open(eval_file, "w"), indent=4)

        for frame in batch:
            _, frame_losses, frame_logs = self.predict_frame(frame)
            total_frame_loss = torch.sum(torch.stack(frame_losses), dim=0)
            for loss_log in frame_logs:
                self.loss_logs.append(loss_log)
                self.loss_logs_history.append(loss_log)

            self.num_steps += 1

            return total_frame_loss

    def eval(self, eval_frames):
        all_logs = []
        self.predictor.model.sam_mask_decoder.train(False)
        self.predictor.model.sam_prompt_encoder.train(False)
        for frame in tqdm(eval_frames, desc="Evaluating frames"):
            predicted_masks, frame_losses, frame_logs = self.predict_frame(frame)
            all_logs.extend(frame_logs)
        # set back to train
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)
        return all_logs

    # returns mask np.ndarray's, loss Tensor and loss logs for each mask in frame.get_masks()
    def predict_frame(self, image, points):
        with autocast(dtype=torch.float16):
            predicted_masks = []
            self.predictor.set_image(image)  # shape: (512, 512, 3)
            input_label = np.ones((1))  # shape: (1,)
            # TODO: can we batch masks for the same image together?
            for point in points:

                point = np.array(point).reshape(1, 2)

                # prepare visual prompts
                # None, (1, 1, 2), (1, 1), None
                mask_input, unnorm_coords, labels, unnorm_box = (
                    self.predictor._prep_prompts(
                        point,
                        input_label,
                        box=None,
                        mask_logits=None,
                        normalize_coords=True,
                    )
                )

                # encode data
                # shapes: (1, 2, 256) and (1, 256, 64, 64)
                sparse_embeddings, dense_embeddings = (
                    self.predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels),
                        boxes=None,
                        masks=None,
                    )
                )

                # list of size 2, shapes (1, 32, 256, 256) and (1, 64, 128, 128)
                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in self.predictor._features["high_res_feats"]
                ]
                
                # import pdb; pdb.set_trace()

                # shapes: (1, 3, 256, 256) and (1, 3)
                # 3 becuase multimask_output generates 3 predicted masks
                batched_mode = unnorm_coords.shape[0] > 1
                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(
                        0
                    ),
                    image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

                # TODO: do we need to re-order masks by their prd_scores?

                # shape: (1, 3, 512, 512)
                prd_masks = self.predictor._transforms.postprocess_masks(
                    low_res_masks, self.predictor._orig_hw[-1]
                )
                
                prd_mask = torch.sigmoid(prd_masks[:, 0]).to(
                    self.cuda_device
                )  # shape: (1, 512, 512)

                predicted_masks.append(
                    (prd_mask[0].detach().to("cpu").numpy() > 0.5).astype(int)
                )

        return predicted_masks