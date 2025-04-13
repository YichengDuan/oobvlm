from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch


class NavHistory(object):
    def __init__(self, history_window=5):
        self.history = [{}]
        self.history_window = history_window

    def add(self, message):

        self.history.append(message)

    def current_heistory_length(self):
        return len(self.history)

    def retrieve(self):

        if len(self.history) > self.history_window:
            return self.history[-self.history_window :]
        return self.history

    def get_last_reflection(self):
        if len(self.history) == 0:
            return "Not Yet"
        return self.history[-1].get("reflection","Not Yet")
    
    def get_all_history(self):
        return self.history

    def clear(self):
        self.history = []


class QwenVLAgent(object):
    def __init__(
        self,
        model_path: str = "./model/Qwen2.5-VL-3B-Instruct",
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

    def history_management(self, messages):
        # manage the history of messages for agent movement in the environment

        return

    def prompt_builder(self, img_str, history:NavHistory, master_instruction):
        # build the prompt for the agent based on the messages

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a robot for navigation task. You have a FOV 120 degree camera on your head."
                            "You can see the world in front of you."
                            f"You have a master instruction now is: {master_instruction}."
                            f"You can move in one directions: forward {self.agent_specs['forward_step_size']}m "
                            f"You can also turn left {self.agent_specs['turn_angle']} degree and right {self.agent_specs['turn_angle']} degree. "
                            "You can also stop. "
                            f"You action space is: {self.agent_specs['action_space']}."
                            "Think before you act. walk like a human. constently reflect on your actions."
                            "When you think you finished the task discribed by the master instruction, you need stop"
                            "You need to make an high level plan first, like a human. for example, you need to check the door first, then go to the door, then go through the door."
                            "If you see an door need to check, you need go to check the door first."
                            "If you cannot see the target you want to see, you need to turn left or turn right couple of time to find your goal first."
                            "If you cannot see the target you want to see directly, the goal may on your back side turn around and check the back side first."
                            "If you want to see both side, you need to turn left first couple of times then turn right couple of times."
                            "If you see an obstacle in front of you, you can tern left or turn right, then go forward to pass the obstacle."
                            "If you feel stuck: mutiple same early actions in history without different reflections, you neet turn left or turn right to find the right path you can walk!!!!, then go forward, then repeat."
                            "If your early action contains looking an direction, after you trun to other direction for passing the obstacle, you need to turn back to the direction you want to look at."
                            f"You have Your last {history.history_window} History Summary:"
                            f"History: {history.retrieve()}"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{img_str}",
                    },
                    {
                        "type": "text",
                        "text": f"You see an FPV image form your head at current state. Your last reflection on your act is {history.get_last_reflection()}. You want to finish the task discribed by the master instruction, what do you do for next new action?. Only respond with one action name from your action space and one sentience reflection what your current state are, things you want to find and what you will do next 4 step, split by ':'.",
                    },
                ],
            },
        ]

        if self.nav_graph:
            # get the nav graph from the nav graph module

            pass
        elif self.vision_interpreter:
            # get the vision interpreter from the vision interpreter module

            pass
        else:
            # build the prompt based on the image and master instruction
            pass

        return messages

    def get_action(self, img_str: str, instruction: str):
        # get the action from the agent based on the messages
        # The action should be one of the actions in the action space
        action = None
        if not self.last_master_instruction:
            self.last_master_instruction = instruction
        if self.last_master_instruction != instruction:
            self.history.clear()
            self.last_master_instruction = instruction

        messages = self.prompt_builder(
            img_str=img_str, history=self.history, master_instruction=instruction
        )
        act_str = self.generate(messages)
        print("act_str", act_str)
        action = act_str[0].split(":")[0].strip()
        reflection = act_str[0].split(":")[1].strip()
        self.history.add(
            {   
                "step": self.history.current_heistory_length(),
                "action": action,
                "reflection": reflection,
            }
        )
        return action

    def generate(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
