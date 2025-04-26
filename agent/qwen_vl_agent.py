from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch
import json
import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NavHistory(object):
    def __init__(self, history_window=5):
        self.history = [{}]
        self.history_window = history_window

    def add(self, message):

        self.history.append(message)

    def current_history_length(self):

        return len(self.history)

    def retrieve(self):

        if len(self.history) > self.history_window:
            return self.history[-self.history_window :]
        return self.history

    def get_last_reflection(self):
        if len(self.history) == 0:
            return "Not Yet"
        return self.history[-1].get("reflection", "Not Yet")

    def get_all_history(self):
        return self.history

    def clear(self):
        self.history = []


class QwenVLAgent(object):
    """
    QwenVL agent for navigation task
    """
    def __init__(
        self,
        model_path: str = "./model/Qwen2.5-VL-7B-Instruct",
        nav_graph=None,
        vision_interpreter=None,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 28 * 28,
        agent_specs={},
    ):
        self.nav_graph = nav_graph
        self.vision_interpreter = vision_interpreter
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        assert type(agent_specs) == dict, "agent_specs should be a dict"

        if agent_specs is not None:
            self.agent_specs = agent_specs
        else:
            self.agent_specs = {
                "action_space": ["move_forward", "turn_left", "turn_right", "stop"],
                "forward_step_size": 0.25,
                "turn_angle": 15,
            }
        self.history = NavHistory(history_window=5)
        self.last_master_instruction = None

    def prompt_builder(self, img_str_list:list[str], history: NavHistory, master_instruction):
        # build the prompt for the agent based on the messages

        if self.nav_graph:
            # get the nav graph from the nav graph module

            pass
        elif self.vision_interpreter:
            # get the vision interpreter from the vision interpreter module

            pass
        else:
            # build the prompt based on the image and master instruction
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"You are a planning robot for navigation task. You have a FOV 120 degree camera on your head."
                                "You see one FPV 2-frame video form your head. Based on the current visual input and your last visual input image, decide your next action. "
                                "You can see the world in front of you. You have to understand the 3D structure based on the image you see."
                                "In your world, all doors are open."
                                f"You have one a master instruction: {master_instruction}."
                                f"You can move in one directions: forward {self.agent_specs['forward_step_size']}m also can turn left {self.agent_specs['turn_angle']} degree and right {self.agent_specs['turn_angle']} degree. "
                                f"You action space is: {self.agent_specs['action_space']}."
                                "FISRT TURN YOUR HEAD TO SEE the path YOU need to walk, THEN MOVE FORWARD TO THE TARGET."
                                "IF NO VISIBLE PATH IN FRONT OF YOU, JUST MOVE FORWARD TO EXPLORE THE WORLD!!!!"
                                "BE COURAGE !!!!, move_forward TO EXPLOR THE WORLD !!!!"
                                "YOU NEED TO BALLANCE TURN LEFT AND TURN RIGHT, DO NOT TURN LEFT OR TURN RIGHT TOO MUCH."
                                "You can also stop, You can only stop when you think you finished ALL task discribed by the master instruction"
                                "The goal position of your master instruction may not in the same room with you"
                                "If you see an DOOR need to check, you need plan an way to check at the door First"
                                "If you see an obstacle in front of you, you can tern left or turn right, then go forward in the path to pass the obstacle."
                                "If you see an Path in front of you, you can go forward in the path."
                                "If you cannot see the target you want to see, you need to turn left or turn right couple of time to find your goal first, then move forward to the goal. also, the goal may on your back side turn around and check the back side first"
                                f"Last {history.history_window} steps you have done are: "
                                f"{history.retrieve()}"
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        # {
                        #     "type": "image",
                        #     "image": f"data:image;base64,{img_str_list[0]}",
                        #     "image": f"data:image;base64,{img_str_list[1]}",
                        # },
                        {
                            "type": "video",
                            "video": [
                                f"data:image;base64,{img_str_list[0]}",
                                f"data:image;base64,{img_str_list[1]}",
                            ],
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Your are at step: {history.current_history_length()}."
                                "You see one FPV video form your head at last state and current state.The first frame is the last state, the second frame is the current state. Based on the current visual input and your history, decide your next action. "
                                # f"Your last reflection was: ['{history.get_last_reflection()}']. "
                                f"Respond with a summary reflection base on your current state and history reflections, what do you see?"
                                "What part of the master instruction have you done? Any obstacles? How to pass them? Your next plan?  And one action name pick from your (action space) for your current state. "
                                "Do you achiece the reflection plan goal? If not, what should you do?"
                                "NOT include word: addCriterion"
                                'MUST Respond as one JSON object. Your JSON object must have exactly two keys: "action" and "reflection" : '
                                "{'action': your action , 'reflection' : your reflection }"
                            ),
                        },

                    ],
                },
            ]

        return messages

    def get_action(self, img_str_list: list[str], instruction: str):
        # get the action from the agent based on the messages
        # The action should be one of the actions in the action space
        action = None
        if not self.last_master_instruction:
            self.last_master_instruction = instruction
        if self.last_master_instruction != instruction:
            self.history.clear()
            self.last_master_instruction = instruction

        messages = self.prompt_builder(
            img_str_list=img_str_list, history=self.history, master_instruction=instruction
        )
        act_str = self.generate(messages)[0].strip("```").strip("json")

        try:
            act_str = json.loads(act_str)
            action = act_str.get("action", None)
            reflection = act_str.get("reflection", None)

            self.history.add(
                {
                    "step": self.history.current_history_length(),
                    "action": action,
                    "reflection": reflection,
                }
            )
            return action
        
        except json.JSONDecodeError as e:
            action = "stay"
            return action

        

    def generate(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            # images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            top_p = 0.93,
            temperature = 0.2,
            batch_size=1,
        )
        # clear cache
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return output_text
