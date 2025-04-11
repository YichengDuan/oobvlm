from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "./model/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained("./model/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 640*28*28
processor = AutoProcessor.from_pretrained("./model/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "You are a robot. You see an FPV image form your head. You want to navigate youself to the dog position, what do you do next? Only return the option form ['move_left','move_front','move_back','move_right']"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("mps")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


class NavHistory(object):
    def __init__(self, history_window=5):
        self.history = []
        self.history_window = history_window

    def add(self, message):
        self.history.append(message)

    def current_heistory_length(self):
        return len(self.history)

    def retrieve(self):
        if len(self.history) > self.history_window:
            return self.history[-self.history_window:]
        return self.history
        
    def get_history(self):
        return self.history
    
    def clear(self):
        self.history = []

class QwenVLAgent(object):
    def __init__(self, nav_graph=None, vision_interpreter=None, min_pixels=256*28*28, max_pixels=640*28*28, agent_specs=None):
        self.nav_graph = nav_graph
        self.vision_interpreter = vision_interpreter
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" 
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "./model/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            "./model/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
        )
        
        self.agent_specs = {
            "action_space": ["move_left", "move_front", "move_back", "move_right"],
            "forward_step_size": 0.25,
            "turn_angle": 15
        }
        self.history = NavHistory(history_window=5)
        self.master_instruction = None

    def history_management(self, messages):
        # manage the history of messages for agent movement in the environment

        return

    def prompt_builder(self, image, last_action, master_instruction):
        # build the prompt for the agent based on the messages

        return messages
    
    def get_action(self, image, instruction):
        action = None
        # get the action from the agent based on the messages
        # The action should be one of the actions in the action space

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
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text