# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict
import logging

import torch
import torch.nn.functional as F
import numpy as np
import cv2 # Needed for resizing frames if not already tensors

# Assuming these are available from the original package
from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores

# Set up logging
logger = logging.getLogger(__name__)


class SAM2StreamingPredictor(SAM2Base):
    """
    A predictor class for SAM-2 that handles single-frame (streaming) inference
    while maintaining tracking state across frames.
    """

    def __init__(
        self,
        # --- Parameters from original SAM2VideoPredictor ---
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        add_all_frames_to_correct_as_cond=False,
        # --- Parameters needed for streaming ---
        video_height: int = None, # Must be provided if normalize_coords is True in add_new_points_or_box
        video_width: int = None,  # Must be provided if normalize_coords is True in add_new_points_or_box
        offload_state_to_cpu=False,
        # --- SAM2Base parameters ---
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

        # --- State Initialization ---
        self.inference_state = self._initialize_inference_state(
            video_height, video_width, offload_state_to_cpu
        )
        self.current_frame_index = -1 # Track the index of the last processed frame

        logger.info(f"SAM2StreamingPredictor initialized. Image size: {self.image_size}, Device: {self.device}")
        if video_height is None or video_width is None:
             logger.warning("video_height/video_width not provided. Coordinate normalization in add_new_points_or_box might fail if normalize_coords=True.")


    def _initialize_inference_state(self, video_height, video_width, offload_state_to_cpu):
        """Initializes the state dictionary without loading video frames."""
        inference_state = {}
        # Store basic properties
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = self.device
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = self.device

        # State dictionaries (similar to original)
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # Cache only needed features (e.g., current/previous frame) instead of all
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        # No need for temp_output_dict_per_obj if inputs are processed directly in predict_next_frame
        # inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        # Track which frames have inputs pending processing
        inference_state["pending_inputs"] = set()

        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2StreamingPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
            model_id (str): The Hugging Face repository ID.
            **kwargs: Additional arguments to pass to the model constructor
                      (e.g., video_height, video_width, offload_state_to_cpu).

        Returns:
            (SAM2StreamingPredictor): The loaded model.
        """
        # Assuming a similar builder function exists or can be adapted
        from sam2.build_sam import build_sam2_video_predictor_hf # Reuse or adapt this builder

        # The builder needs to instantiate SAM2StreamingPredictor instead of SAM2VideoPredictor
        # This might require modifying build_sam2_video_predictor_hf or creating a new builder
        # For now, let's assume it returns the base model and we wrap it.
        # A more direct approach would modify the builder function.

        # --- Placeholder approach: Load base and wrap ---
        # sam_model_base = build_sam2_video_predictor_hf(model_id, **kwargs) # This loads the *Video* predictor
        # # Extract necessary components if possible, or modify the builder directly.
        # # This part is tricky without seeing the builder function. Let's assume
        # # the builder *can* return the right type or necessary components.
        # # If build_sam2_video_predictor_hf inherently loads video, it must be changed.

        # --- Ideal approach: Modify builder or use a dedicated one ---
        # Assuming a hypothetical streaming builder exists:
        # from sam2.build_sam import build_sam2_streaming_predictor_hf
        # sam_model = build_sam2_streaming_predictor_hf(model_id, **kwargs)
        # return sam_model

        # --- Workaround: Manually load and initialize (less clean) ---
        # This depends heavily on how build_sam2_video_predictor_hf works.
        # If it just loads weights into SAM2Base, we can do:
        logger.warning("Using a workaround for from_pretrained. Ideally, adapt or create a dedicated builder function for streaming.")
        # Extract potential SAM2Base args from kwargs, others are for StreamingPredictor
        base_kwargs = {k: v for k, v in kwargs.items() if k not in ['video_height', 'video_width', 'offload_state_to_cpu', 'fill_hole_area', 'non_overlap_masks', 'clear_non_cond_mem_around_input', 'add_all_frames_to_correct_as_cond']}
        predictor_kwargs = {k: v for k, v in kwargs.items() if k in ['video_height', 'video_width', 'offload_state_to_cpu', 'fill_hole_area', 'non_overlap_masks', 'clear_non_cond_mem_around_input', 'add_all_frames_to_correct_as_cond']}

        # Load the base model structure with weights
        sam_model_base = SAM2Base.from_pretrained(model_id, **base_kwargs) # Assuming SAM2Base has from_pretrained

        # Create the streaming predictor instance
        predictor = cls(**predictor_kwargs)

        # Transfer state/weights from the loaded base model to the predictor
        predictor.__dict__.update(sam_model_base.__dict__)
        predictor.to(predictor.device) # Ensure model parts are on the correct device

        return predictor
        # raise NotImplementedError("Loading from pretrained requires adapting the build_sam2_video_predictor_hf function or creating a new one for SAM2StreamingPredictor.")


    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
        # Delegate to internal state
        obj_idx = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # Add new object
        obj_idx = len(self.inference_state["obj_id_to_idx"])
        self.inference_state["obj_id_to_idx"][obj_id] = obj_idx
        self.inference_state["obj_idx_to_id"][obj_idx] = obj_id
        self.inference_state["obj_ids"].append(obj_id) # Keep list updated

        # Initialize state structures for the new object
        self.inference_state["point_inputs_per_obj"][obj_idx] = {}
        self.inference_state["mask_inputs_per_obj"][obj_idx] = {}
        self.inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        # No temp dict needed here
        self.inference_state["frames_tracked_per_obj"][obj_idx] = {}
        logger.info(f"Added new object ID: {obj_id} mapped to index: {obj_idx}")
        return obj_idx

    def _obj_idx_to_id(self, obj_idx):
        """Map model-side object index to client-side object id."""
        return self.inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self):
        """Get the total number of unique object ids."""
        return len(self.inference_state["obj_idx_to_id"])

    def _preprocess_frame(self, frame):
        """
        Preprocesses a single frame (numpy HWC uint8 or tensor CHW float)
        to the format expected by the model (tensor B C H W, normalized).
        Returns the processed tensor on the correct device.
        """
        if isinstance(frame, np.ndarray):
            # Assume HWC, BGR uint8 if numpy (standard cv2 format)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Input numpy frame must be HWC with 3 channels.")
            if frame.dtype != np.uint8:
                raise ValueError("Input numpy frame must be uint8.")
            # Convert HWC BGR to CHW RGB float tensor
            frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous() # HWC -> CHW
            frame = frame[[2, 1, 0], :, :] # BGR -> RGB
            frame = frame.float()

        elif isinstance(frame, torch.Tensor):
            # Assume CHW float tensor
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0) # Remove batch if present
            if frame.ndim != 3:
                 raise ValueError(f"Input tensor frame must be CHW, got shape {frame.shape}")
            # Ensure float
            frame = frame.float()
        else:
            raise TypeError("Input frame must be a numpy array (HWC BGR uint8) or torch tensor (CHW float RGB)")

        # Add batch dimension
        frame = frame.unsqueeze(0) # B C H W
        device = self.inference_state["device"]
        frame = frame.to(device)

        # Resize to model's expected input size
        b, c, h, w = frame.shape
        target_size = self.image_size
        if h != target_size or w != target_size:
            frame = F.interpolate(
                frame,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        # Normalize (assuming standard ImageNet normalization used by SAM)
        # Make sure pixel_mean and pixel_std are attributes of SAM2Base or defined here
        if not hasattr(self, 'pixel_mean') or not hasattr(self, 'pixel_std'):
             # Provide default values or ensure they are set in SAM2Base.__init__
             self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(-1, 1, 1)
             self.pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(-1, 1, 1)
             logger.warning("Using default pixel_mean/std for normalization.")

        frame = (frame - self.pixel_mean) / self.pixel_std

        return frame # Shape: (1, C, image_size, image_size)


    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        frame_idx: int, # The index this input corresponds to
        obj_id,
        points=None,
        labels=None,
        box=None, # Box coordinates [x1, y1, x2, y2]
        clear_old_points=True,
        normalize_coords=True, # Normalize from video_height/width if True
    ):
        """
        Stores new point or box prompts for a specific object and frame index.
        Inference is deferred until predict_next_frame processes this frame_idx.
        """
        if frame_idx <= self.current_frame_index:
             logger.warning(f"Adding input for frame {frame_idx} which has already been processed (current={self.current_frame_index}). This input may be ignored unless reprocessing is supported.")
        if self.inference_state["video_height"] is None and normalize_coords:
             raise ValueError("Cannot normalize coordinates because video_height/width were not provided during initialization.")

        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        # --- Input validation and processing (similar to original) ---
        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0) # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0) # add batch dimension

        if box is not None:
            if not clear_old_points:
                raise ValueError("Cannot add box without clearing old points. Use clear_old_points=True.")
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            # Box format [x1, y1, x2, y2] -> SAM format [[x1, y1], [x2, y2]]
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device).reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = self.inference_state["video_height"]
            video_W = self.inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H], device=points.device)

        # Scale to model's internal image size
        points = points * self.image_size
        points = points.to(self.inference_state["device"]) # Ensure points/labels on compute device
        labels = labels.to(self.inference_state["device"])

        # Store the processed points
        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None) # Clear mask input for this frame if points are added
        self.inference_state["pending_inputs"].add(frame_idx)
        logger.info(f"Stored point/box input for obj_id {obj_id} at frame {frame_idx}.")
        # NOTE: No inference happens here.


    @torch.inference_mode()
    def add_new_mask(
        self,
        frame_idx: int, # The index this input corresponds to
        obj_id,
        mask, # Input mask (H, W) boolean or float tensor/numpy array
    ):
        """
        Stores a new mask prompt for a specific object and frame index.
        Inference is deferred until predict_next_frame processes this frame_idx.
        """
        if frame_idx <= self.current_frame_index:
             logger.warning(f"Adding input for frame {frame_idx} which has already been processed (current={self.current_frame_index}). This input may be ignored unless reprocessing is supported.")

        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)

        if mask.dtype == torch.bool:
            mask = mask.float() # Convert boolean to float

        assert mask.dim() == 2 # Expect H, W
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask.unsqueeze(0).unsqueeze(0) # Add batch and channel dims (1, 1, H, W)
        mask_inputs_orig = mask_inputs_orig.to(self.inference_state["device"])

        # Resize mask to model's input size if needed
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = F.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float() # Threshold resized mask
        else:
            mask_inputs = mask_inputs_orig

        # Store the processed mask
        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None) # Clear point input for this frame
        self.inference_state["pending_inputs"].add(frame_idx)
        logger.info(f"Stored mask input for obj_id {obj_id} at frame {frame_idx}.")
        # NOTE: No inference happens here.


    def _get_orig_video_res_output(self, any_res_masks):
        """
        Resize predicted masks to the original video resolution and optionally
        apply non-overlapping constraints.
        Input: masks tensor (B, 1, H, W) - usually at model's low-res output size.
        Returns: Tuple (low_res_masks, video_res_masks)
                 low_res_masks: Input masks, potentially moved to CPU if offloading.
                 video_res_masks: Masks resized to video H/W, potentially non-overlapped.
        """
        if any_res_masks is None:
            return None, None

        device = self.inference_state["device"]
        storage_device = self.inference_state["storage_device"]
        video_H = self.inference_state["video_height"]
        video_W = self.inference_state["video_width"]

        if video_H is None or video_W is None:
             logger.warning("Cannot resize output to video resolution, video_height/width unknown.")
             video_res_masks = any_res_masks.clone() # Return low-res if size unknown
        elif any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks.clone() # Already correct size
        else:
            # Ensure on compute device for interpolation
            any_res_masks_dev = any_res_masks.to(device, non_blocking=True)
            video_res_masks = F.interpolate(
                any_res_masks_dev,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
                # antialias=True, # Usually not needed for upsampling logits
            )

        # Apply non-overlapping constraints if needed (on video resolution masks)
        if self.non_overlap_masks and video_res_masks.shape[0] > 1: # Only if multiple objects
            # Ensure on compute device
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks.to(device))

        # Apply hole filling if needed (on video resolution masks)
        # Note: fill_holes expects scores, not binary masks yet
        if self.fill_hole_area > 0:
             # Move back to storage device before potentially slow CPU operations
             video_res_masks = fill_holes_in_mask_scores(video_res_masks.to(storage_device), self.fill_hole_area)

        # Move original resolution masks to storage device for return
        video_res_masks = video_res_masks.to(storage_device, non_blocking=True)
        # Also move the input low-res masks to storage device
        low_res_masks_stored = any_res_masks.to(storage_device, non_blocking=True)

        return low_res_masks_stored, video_res_masks

    # Placeholder for non-overlapping logic if needed (likely exists in SAM2Base or utils)
    def _apply_non_overlapping_constraints(self, masks):
        # This function needs to exist, potentially in SAM2Base or utils.
        # It takes (B, 1, H, W) mask logits and ensures only one object has max score per pixel.
        if masks.shape[0] <= 1:
            return masks
        # Add a background score channel
        bg_score = torch.full_like(masks[:, :1], NO_OBJ_SCORE) # Use NO_OBJ_SCORE as background baseline
        masks_with_bg = torch.cat([bg_score, masks], dim=1) # Shape (B, N_obj+1, H, W)
        # Find the argmax along the object dimension
        indices = torch.argmax(masks_with_bg, dim=1, keepdim=True) # Shape (B, 1, H, W)
        # Create one-hot encoding based on argmax
        one_hot_masks = torch.zeros_like(masks_with_bg).scatter_(1, indices, 1)
        # Remove the background channel and create binary output (0 or 1, keep float for consistency)
        # Convert back to logits-like values: 0 -> large negative, 1 -> 0 (or keep original scores where max)
        # Simplified: return the original scores where they were max, else NO_OBJ_SCORE
        is_max = one_hot_masks[:, 1:] # Shape (B, N_obj, H, W)
        constrained_masks = torch.where(is_max.bool(), masks, torch.full_like(masks, NO_OBJ_SCORE))
        # logger.debug("Applied non-overlapping constraints.")
        return constrained_masks


    @torch.inference_mode()
    def predict_next_frame(self, frame):
        """
        Processes a single incoming frame, updates the tracking state, and returns predictions.

        Args:
            frame: The video frame as a numpy array (HWC, BGR, uint8) or
                   torch tensor (CHW, RGB, float).

        Returns:
            tuple: (frame_idx, obj_ids, video_res_masks)
                frame_idx (int): The index of the processed frame.
                obj_ids (list): List of object IDs being tracked.
                video_res_masks (torch.Tensor): Predicted masks for each object,
                    resized to original video resolution (B, 1, H, W) on the storage device.
                    Returns None if no objects are being tracked.
        """
        self.current_frame_index += 1
        frame_idx = self.current_frame_index
        logger.info(f"Processing frame {frame_idx}...")

        num_objs = self._get_obj_num()
        if num_objs == 0:
            logger.warning(f"No objects added yet. Call add_new_points_or_box/add_new_mask first. Skipping frame {frame_idx}.")
            return frame_idx, [], None

        # --- Preprocess Frame and Get Features ---
        # Preprocess (resize, normalize, to device)
        processed_frame = self._preprocess_frame(frame) # (1, C, H, W) tensor on self.device

        # Get image features (assuming this method exists in SAM2Base)
        # This might involve caching logic internally in _get_image_feature
        # We pass the *processed* frame tensor here
        # Note: _get_image_feature in original code took frame_idx and used stored images.
        # Here, we adapt it conceptually to take the tensor directly.
        # The actual implementation of _get_image_feature needs to handle this.
        try:
             # Assuming _get_image_feature can optionally take the preprocessed tensor
             # It should store the features in self.inference_state['cached_features'][frame_idx]
             # If it *only* works with indices, we need to store the processed frame first.
             # Let's assume it caches based on frame_idx internally for simplicity here.
             _ = self._get_image_feature(self.inference_state, frame_idx, batch_size=1, frame_tensor=processed_frame)
             logger.debug(f"Computed/Cached features for frame {frame_idx}")
        except Exception as e:
             logger.error(f"Failed to get image features for frame {frame_idx}: {e}", exc_info=True)
             # Decide how to handle: maybe cache the processed frame?
             # self.inference_state['cached_processed_frames'][frame_idx] = processed_frame
             raise RuntimeError(f"Feature extraction failed. Ensure _get_image_feature handles frame_tensor input or adapt caching. Error: {e}")


        # --- Process Each Object ---
        all_pred_masks_low_res = []
        obj_ids = self.inference_state["obj_ids"] # Get current list of object IDs

        for obj_idx in range(num_objs):
            obj_id = self._obj_idx_to_id(obj_idx) # Get obj_id for logging/debugging
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            point_inputs = self.inference_state["point_inputs_per_obj"][obj_idx].get(frame_idx)
            mask_inputs = self.inference_state["mask_inputs_per_obj"][obj_idx].get(frame_idx)
            frames_tracked_dict = self.inference_state["frames_tracked_per_obj"][obj_idx]

            # Determine if this frame has direct input for this object
            has_input = (point_inputs is not None) or (mask_inputs is not None)
            # An initial conditioning frame is one with input that hasn't been tracked before
            is_init_cond_frame = has_input and (frame_idx not in frames_tracked_dict)
            # Default to forward tracking (reverse=False) for streaming
            reverse_tracking = False
            # Should we run the memory encoder? Yes, unless it's an input frame where
            # consolidation might happen later (original code deferred this).
            # For streaming, it's simpler to run it unless performance is critical.
            # Let's align with original logic: don't run mem encoder if processing direct input now.
            run_mem_encoder = not has_input

            # Determine storage key based on input and config flags
            is_cond = is_init_cond_frame or (has_input and self.add_all_frames_to_correct_as_cond)
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

            # Get previous mask logits if correcting an existing prediction
            prev_sam_mask_logits = None
            if has_input and not is_init_cond_frame:
                # Look up previous output *for this frame* if it exists (being corrected)
                prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
                if prev_out is None:
                   prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
                if prev_out is not None and prev_out.get("pred_masks") is not None:
                     device = self.inference_state["device"]
                     prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
                     # Clamp the scale
                     prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
                     logger.debug(f"Using previous mask logits for correction on frame {frame_idx}, obj {obj_id}")


            # --- Run Single Frame Inference ---
            # This requires SAM2Base._run_single_frame_inference to exist and function correctly
            # It uses the internal state (output_dict) to fetch memory from previous frames.
            try:
                 current_out, pred_masks_low_res = self._run_single_frame_inference(
                    inference_state=self.inference_state,
                    output_dict=obj_output_dict, # Pass the specific object's state slice
                    frame_idx=frame_idx,
                    batch_size=1, # Process one object at a time
                    is_init_cond_frame=is_init_cond_frame,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    reverse=reverse_tracking, # Usually False for streaming
                    run_mem_encoder=run_mem_encoder, # Defer if input provided now
                    prev_sam_mask_logits=prev_sam_mask_logits,
                    # Maybe need to pass precomputed features if _run_single_frame_inference doesn't use cache?
                    # current_features = self.inference_state['cached_features'][frame_idx] ?
                )
            except Exception as e:
                logger.error(f"Inference failed for obj_id {obj_id} on frame {frame_idx}: {e}", exc_info=True)
                # Handle failure: maybe predict NO_OBJ_SCORE mask?
                h = w = self.image_size // 4 # Model's low-res mask output size
                pred_masks_low_res = torch.full((1, 1, h, w), NO_OBJ_SCORE, dtype=torch.float32, device=self.inference_state["storage_device"])
                # Create a dummy 'current_out' to avoid crashing state update? Or propagate error?
                # For now, create dummy output so state update doesn't fail
                current_out = {
                     "pred_masks": pred_masks_low_res.clone(),
                     "object_score_logits": torch.tensor([NO_OBJ_SCORE], device=pred_masks_low_res.device), # Check actual score name/shape
                     "maskmem_features": None, # Encoder wasn't run or failed
                     "maskmem_pos_enc": None,
                     # Add other expected keys with None/default values if necessary
                }
                # Continue processing other objects if desired


            # If input was provided and memory encoder wasn't run, run it now
            if has_input and not run_mem_encoder and current_out.get("maskmem_features") is None:
                try:
                    # Need high-res masks for memory encoder
                    high_res_masks = F.interpolate(
                        current_out["pred_masks"].to(self.inference_state["device"]),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                        inference_state=self.inference_state,
                        frame_idx=frame_idx,
                        batch_size=1, # Single object
                        high_res_masks=high_res_masks,
                        object_score_logits=current_out.get("object_score_logits"), # Pass scores if available
                        is_mask_from_pts=True, # Indicate it came from explicit input
                    )
                    current_out["maskmem_features"] = maskmem_features
                    current_out["maskmem_pos_enc"] = maskmem_pos_enc
                    logger.debug(f"Ran memory encoder for input frame {frame_idx}, obj {obj_id}")
                except Exception as e:
                    logger.error(f"Memory encoder failed for input frame {frame_idx}, obj {obj_id}: {e}", exc_info=True)
                    # Features will remain None, potentially impacting future frames


            # --- Update State ---
            # Store results (potentially offloaded to CPU within current_out by _run_single_frame_inference)
            obj_output_dict[storage_key][frame_idx] = current_out

            # If this output is now 'conditioning', remove any old 'non-conditioning' output for this frame
            if is_cond:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

            # Mark frame as tracked for this object
            frames_tracked_dict[frame_idx] = {"reverse": reverse_tracking}

            # Clear surrounding non-conditioning memory if configured and input was added
            if has_input and self.clear_non_cond_mem_around_input:
                self._clear_obj_non_cond_mem_around_input(self.inference_state, frame_idx, obj_idx)

            # Collect the low-res prediction for final processing
            all_pred_masks_low_res.append(pred_masks_low_res)

        # --- Consolidate and Finalize ---
        if not all_pred_masks_low_res:
             logger.warning(f"No predictions generated for frame {frame_idx}.")
             return frame_idx, obj_ids, None

        # Concatenate predictions from all objects
        # Ensure they are on the same device before cat (should be storage_device)
        device = all_pred_masks_low_res[0].device
        try:
            concatenated_masks = torch.cat([m.to(device) for m in all_pred_masks_low_res], dim=0)
        except Exception as e:
             logger.error(f"Failed to concatenate masks on device {device} for frame {frame_idx}: {e}. Individual mask devices: {[m.device for m in all_pred_masks_low_res]}", exc_info=True)
             # Fallback or error
             return frame_idx, obj_ids, None

        # Resize to original video resolution and apply constraints/filling
        _, video_res_masks = self._get_orig_video_res_output(concatenated_masks)

        # Clean up pending input marker for this frame
        self.inference_state["pending_inputs"].discard(frame_idx)

        # Optional: Clean up old cached features to save memory
        self._cleanup_cached_features(frame_idx)

        logger.info(f"Finished processing frame {frame_idx}. Returning {num_objs} masks.")
        return frame_idx, obj_ids, video_res_masks

    def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
        """Clears non-conditioning memory around an input frame for a specific object."""
        # This logic likely exists in the original SAM2VideoPredictor or SAM2Base
        # It needs access to inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        # and potentially modifies it based on frame_idx and model's memory window.
        # Re-implement or ensure it's available from the base class.
        logger.debug(f"Requested clearing non-cond memory around frame {frame_idx} for obj {obj_idx} (Implementation may be needed).")
        # Example placeholder logic (adjust based on actual implementation):
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        frames_to_clear = []
        # Define a window (e.g., based on model's temporal context)
        window = getattr(self, 'memory_window_size', 5) # Example window size
        for f_idx in range(max(0, frame_idx - window), min(self.current_frame_index + 1, frame_idx + window + 1)):
             if f_idx != frame_idx and f_idx in obj_output_dict["non_cond_frame_outputs"]:
                 frames_to_clear.append(f_idx)

        if frames_to_clear:
             logger.info(f"Clearing non-cond memory for frames {frames_to_clear} for obj {obj_idx} due to input at {frame_idx}.")
             for f_idx in frames_to_clear:
                 # Option 1: Just delete the entry
                 # del obj_output_dict["non_cond_frame_outputs"][f_idx]
                 # Option 2: Clear memory features but keep mask prediction?
                 if "maskmem_features" in obj_output_dict["non_cond_frame_outputs"][f_idx]:
                     obj_output_dict["non_cond_frame_outputs"][f_idx]["maskmem_features"] = None
                     obj_output_dict["non_cond_frame_outputs"][f_idx]["maskmem_pos_enc"] = None


    def _cleanup_cached_features(self, current_frame_idx):
        """Removes old cached image features to save memory."""
        # Determine which frames' features are safe to remove
        # Depends on the memory lookback window of the model
        cache = self.inference_state.get("cached_features", {})
        if not cache:
            return
        # Example: Keep features for the last 'N' frames (e.g., N=10)
        keep_window = getattr(self, 'feature_cache_window', 10)
        frames_to_remove = [fidx for fidx in cache.keys() if fidx < current_frame_idx - keep_window]
        for fidx in frames_to_remove:
            del cache[fidx]
            logger.debug(f"Removed cached features for frame {fidx}")


    def reset_state(self, video_height=None, video_width=None):
        """Resets the predictor state to start tracking a new sequence."""
        logger.info("Resetting predictor state.")
        # Use provided dimensions or keep existing if not provided
        h = video_height if video_height is not None else self.inference_state["video_height"]
        w = video_width if video_width is not None else self.inference_state["video_width"]
        offload = self.inference_state["offload_state_to_cpu"]
        self.inference_state = self._initialize_inference_state(h, w, offload)
        self.current_frame_index = -1
        # Call any potential reset methods in the base class if needed
        if hasattr(super(), 'reset_state'):
             super().reset_state()


    # --- Ensure necessary internal methods are available ---
    # These methods are called above and assumed to exist in SAM2Base or be implemented here.
    # Their exact signature and behavior might need adjustment based on SAM2Base.

    def _get_image_feature(self, inference_state, frame_idx, batch_size, frame_tensor=None):
        # This needs to be implemented in SAM2Base or here.
        # It should take the frame_tensor (if provided) or load if necessary (not applicable here),
        # run the image encoder, and store results in inference_state['cached_features'].
        # Placeholder implementation:
        if frame_idx in inference_state["cached_features"]:
             return inference_state["cached_features"][frame_idx]
        if frame_tensor is None:
            # In streaming, frame_tensor should always be provided to this function
            # If it needs to be loaded from disk/memory based on index, that logic goes here.
            raise ValueError(f"_get_image_feature called for frame {frame_idx} without frame_tensor.")

        logger.debug(f"Computing image features for frame {frame_idx}...")
        # Assume self.image_encoder exists from SAM2Base
        # frame_tensor is already preprocessed (1, C, H, W) on self.device
        features = self.image_encoder(frame_tensor) # This likely returns multiple feature levels
        inference_state["cached_features"][frame_idx] = features
        return features
        # raise NotImplementedError("_get_image_feature needs implementation in SAM2Base or here.")

    def _run_memory_encoder(self, inference_state, frame_idx, batch_size, high_res_masks, object_score_logits, is_mask_from_pts):
        # This needs to be implemented in SAM2Base or here.
        # Takes high-res masks and encodes them into memory features.
        # Placeholder implementation:
        logger.debug(f"Running memory encoder for frame {frame_idx} (Placeholder - needs implementation)...")
        # Assume self.memory_encoder and self.mask_feature_proj exist
        # Needs image features for the frame
        img_features = inference_state["cached_features"].get(frame_idx)
        if img_features is None:
             logger.error(f"Cannot run memory encoder: image features for frame {frame_idx} not found.")
             return None, None
        # Memory encoder might need specific feature level
        # Placeholder call - adapt based on actual memory encoder signature
        # maskmem_features = self.memory_encoder(img_features, high_res_masks)
        # maskmem_pos_enc = self.mask_feature_proj(maskmem_features) # Example projection
        # return maskmem_features, maskmem_pos_enc
        return torch.randn(batch_size, 256, 64, 64, device=high_res_masks.device), \
               torch.randn(batch_size, 256, 64, 64, device=high_res_masks.device) # Dummy tensors
        # raise NotImplementedError("_run_memory_encoder needs implementation in SAM2Base or here.")

    def _run_single_frame_inference(self, inference_state, output_dict, frame_idx, batch_size, is_init_cond_frame, point_inputs, mask_inputs, reverse, run_mem_encoder, prev_sam_mask_logits=None):
        # This is the core inference logic, assumed to be in SAM2Base.
        # It uses image features, memory features from 'output_dict', point/mask inputs,
        # runs the mask decoder, and returns the results ('current_out') and low-res masks.
        # It MUST handle fetching memory features from previous frames based on 'output_dict'.
        # Placeholder implementation:
        logger.debug(f"Running single frame inference for frame {frame_idx} (Placeholder - needs implementation)...")
        # Fetch required inputs (image features, memory features)
        # Run mask decoder
        # Update memory
        # Return results
        h = w = self.image_size // 4 # Example low-res mask size
        device = inference_state["storage_device"] # Return results on storage device
        pred_masks = torch.randn(batch_size, 1, h, w, device=device)
        # Simulate applying inputs (if any) - just modify random output slightly
        if point_inputs is not None or mask_inputs is not None:
             pred_masks += 0.5
        if prev_sam_mask_logits is not None:
             pred_masks += 0.2

        current_out = {
            "pred_masks": pred_masks.clone(), # Store a copy
            "object_score_logits": torch.randn(batch_size, device=device),
            "maskmem_features": torch.randn(batch_size, 256, 64, 64, device=device) if run_mem_encoder else None,
            "maskmem_pos_enc": torch.randn(batch_size, 256, 64, 64, device=device) if run_mem_encoder else None,
            # Add other fields expected by the state management logic
        }
        return current_out, pred_masks # Return dict and the tensor separately
        # raise NotImplementedError("_run_single_frame_inference needs implementation in SAM2Base.")