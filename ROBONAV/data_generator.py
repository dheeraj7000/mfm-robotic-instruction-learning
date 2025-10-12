#!/usr/bin/env python3
"""
synthetic_data_generator_litellm.py

Generates a multimodal fine-tuning dataset for MultimodalFusionModel
using AI2-THOR frames + text from a LiteLLM-compatible Mistral model.

Dependencies:
    pip install ai2thor litellm pillow tqdm
"""
import time
import os
import json
import random
from datetime import datetime
from PIL import Image
from tqdm import tqdm

import ai2thor.controller
from litellm import completion  # unified chat API

os.environ['MISTRAL_API_KEY'] = "cXD8hvQnBRWPLXIAhtb9KpYlwGOTQi0l"

# ============================
# CONFIGURATION
# ============================

OUTPUT_DIR = "./synthetic_dataset"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "data.jsonl")

SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3"]
NUM_FRAMES_PER_SCENE = 30  # increase for larger dataset

# Use any LiteLLM-supported Mistral model (e.g. "mistralai/mistral-7b-instruct")
MODEL_NAME = "mistral/mistral-small-latest"
TEMPERATURE = 0.6

AI2THOR_ACTIONS = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "DropHandObject",
    "SliceObject",
]

os.makedirs(FRAMES_DIR, exist_ok=True)

# ============================
# HELPER FUNCTIONS
# ============================

def capture_random_frame(controller, scene_name, frame_idx):
    """Teleport to a random reachable position and capture a frame."""
    controller.reset(scene_name)
    positions = controller.step("GetReachablePositions").metadata["actionReturn"]
    pos = random.choice(positions)
    event = controller.step(action="Teleport", position=pos)
    image = Image.fromarray(event.frame)
    frame_path = os.path.join(FRAMES_DIR, f"{scene_name}_{frame_idx:04d}.png")
    image.save(frame_path)
    return frame_path, event.metadata


def llm_complete(system_prompt, user_prompt):
    """Query LiteLLM-compatible Mistral model and return text response."""
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=256,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LiteLLM error] {e}")
        return None


def query_llm_for_sample(scene_name, metadata, mode):
    """Ask LLM to generate synthetic label and text based on scene metadata."""
    object_list = [obj["objectType"] for obj in metadata["objects"] if obj["visible"]]
    obj_str = ", ".join(object_list[:10]) if object_list else "nothing visible"

    if mode == "action":
        user_prompt = f"You are controlling a household robot in {scene_name}. Visible objects: {obj_str}. What should the robot do next?"
        system_prompt = "Respond with a single valid AI2-THOR action from the set: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, PickupObject, PutObject, OpenObject, CloseObject, ToggleObjectOn, ToggleObjectOff, DropHandObject, SliceObject."
    elif mode == "description":
        user_prompt = f"In {scene_name}, the robot sees: {obj_str}. Describe concisely what it should do next."
        system_prompt = "Provide a short natural-language instruction for the robot's next action."
    elif mode == "vqa":
        user_prompt = f"In {scene_name}, the visible objects are: {obj_str}. Generate a question about this scene and its answer."
        system_prompt = "Provide a short question about the scene and a concise answer."
    elif mode == "progress":
        user_prompt = f"In {scene_name}, the robot is midway through a household task seeing: {obj_str}. Estimate progress between 0 and 1."
        system_prompt = "Respond with only a float number in [0,1]."
    else:
        raise ValueError(f"Unknown mode: {mode}")

    content = llm_complete(system_prompt, user_prompt)
    time.sleep(3)  
    if not content:
        return None

    result = {
        "scene_id": scene_name,
        "mode": mode,
        "prompt": user_prompt,
    }

    # Parse per mode
    if mode == "action":
        result["target_text"] = content
        action_label = next((a for a in AI2THOR_ACTIONS if a.lower() in content.lower()), "MoveAhead")
        result["action_label"] = action_label
        result["action_id"] = AI2THOR_ACTIONS.index(action_label)
    elif mode == "description":
        result["target_text"] = content
    elif mode == "vqa":
        if "?" in content:
            q, a = content.split("?", 1)
            result["question"] = q.strip() + "?"
            result["answer_text"] = a.strip()
            result["target_text"] = f"Q: {q.strip()}? A: {a.strip()}"
        else:
            result["question"] = content
            result["answer_text"] = "Unknown"
            result["target_text"] = f"Q: {content}? A: Unknown"
    elif mode == "progress":
        try:
            val = float(content.split()[0])
            result["progress_value"] = min(max(val, 0.0), 1.0)
        except ValueError:
            result["progress_value"] = random.uniform(0.0, 1.0)

    return result


# ============================
# MAIN PIPELINE
# ============================

def main():
    controller = ai2thor.controller.Controller(
        width=640, height=480,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        renderClassImage=False,
    )

    dataset = []
    modes = ["action", "description", "vqa", "progress"]

    print("🔄 Starting synthetic dataset generation...")
    for scene in SCENES:
        print(f"\n🧩 Scene: {scene}")
        for i in tqdm(range(NUM_FRAMES_PER_SCENE)):
            frame_path, metadata = capture_random_frame(controller, scene, i)
            mode = random.choice(modes)
            entry = query_llm_for_sample(scene, metadata, mode)
            if entry is None:
                continue

            entry["frame_path"] = frame_path
            entry["timestamp"] = i
            entry["split"] = "train"
            dataset.append(entry)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(dataset)} samples to {OUTPUT_JSONL}")
    print(f"🖼️ Frames stored in: {FRAMES_DIR}")

    controller.stop()


if __name__ == "__main__":
    main()
