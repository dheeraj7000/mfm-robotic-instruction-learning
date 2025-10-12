#!/usr/bin/env python3
"""
inference_ai2thor.py

Run inference using a fine-tuned MultimodalFusionModel on live AI2-THOR scenes.

✅ Loads the fine-tuned weights
✅ Captures the current frame
✅ Feeds it + user instruction to the model
✅ Prints the predicted action or text
✅ Executes the action safely inside AI2-THOR
"""

import os
import random
import torch
from PIL import Image
import ai2thor.controller
from transformers import AutoTokenizer
from multimodal_network import MultimodalFusionModel  # your fusion model

# =========================================================
# CONFIG
# =========================================================
MODEL_CKPT = "./checkpoints/mmfusion_epoch3.pt"   # path to your fine-tuned weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENE_NAME = "FloorPlan1"
INSTRUCTION = "Pick up the bowl"
MODE = "description"   # can also test 'description' or 'vqa'

# =========================================================
# INITIALIZE MODEL + TOKENIZER
# =========================================================
print("🚀 Loading fine-tuned model...")
model = MultimodalFusionModel()
model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# INITIALIZE AI2-THOR
# =========================================================
controller = ai2thor.controller.Controller(
    width=640,
    height=480,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    renderClassImage=False,
)

controller.reset(SCENE_NAME)
event = controller.step("GetReachablePositions")
reachable_positions = event.metadata["actionReturn"]
print(f"Scene '{SCENE_NAME}' loaded with {len(reachable_positions)} reachable positions.")

# Teleport randomly
pos = random.choice(reachable_positions)
controller.step(action="Teleport", position=pos)
print(f"📍 Teleported to: {pos}")

# =========================================================
# CAPTURE CURRENT FRAME
# =========================================================
frame = controller.last_event.frame
image = Image.fromarray(frame).convert("RGB")

# =========================================================
# RUN INFERENCE
# =========================================================
print(f"\n🧠 Running inference with mode = {MODE}")
with torch.no_grad():
    outputs = model([image], [INSTRUCTION], mode=MODE)

if MODE == "action":
    logits = outputs["logits"]
    pred_id = torch.argmax(logits, dim=-1).item()
    pred_action = model.action_name(pred_id)
    print(f"🤖 Predicted action: {pred_action}")
elif MODE in ("description", "vqa"):
    print(f"🗣️ Generated text:\n{outputs['texts'][0]}")
else:
    print(outputs)

# =========================================================
# EXECUTE ACTION IN AI2-THOR (safe argument handling)
# =========================================================
def first_pickable(objects):
    for o in objects:
        if o.get("pickupable", False):
            return o
    return None

if MODE == "action":
    print(f"\n🎬 Executing {pred_action} in AI2-THOR...")
    visible_objs = [o for o in controller.last_event.metadata["objects"] if o["visible"]]

    try:
        if pred_action == "PickupObject":
            obj = first_pickable(visible_objs)
            if obj:
                print(f"👉 Targeting object: {obj['objectType']} ({obj['objectId']})")
                event = controller.step(action="PickupObject", objectId=obj["objectId"])
            else:
                print("⚠️ No pickupable objects visible.")
                event = controller.last_event

        elif pred_action in ["OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff"]:
            if visible_objs:
                obj = visible_objs[0]
                print(f"👉 Targeting interactable object: {obj['objectType']} ({obj['objectId']})")
                event = controller.step(action=pred_action, objectId=obj["objectId"])
            else:
                print("⚠️ No interactable objects visible.")
                event = controller.last_event

        else:
            # Simple movement actions
            event = controller.step(action=pred_action)

        print(f"✅ Success: {event.metadata['lastActionSuccess']}")
        if not event.metadata["lastActionSuccess"]:
            print(f"Reason: {event.metadata.get('errorMessage', 'Unknown error')}")
        else:
            visible_objs_after = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
            print(f"👁️ Visible objects now: {visible_objs_after}")

    except Exception as e:
        print(f"❌ Error executing action: {e}")

controller.stop()
print("🏁 Inference complete.")
