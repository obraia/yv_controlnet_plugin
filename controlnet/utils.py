import gc
import os
import random
import numpy as np
import torch
import controlnet_aux
from diffusers import ControlNetModel, schedulers, pipelines
from transformers import pipeline
from PIL import Image
from utils.pillow import PillowUtils


class ControlNet:
    def __init__(self, models_path: str, controlnet_models_path: str, device: torch.device):
        self.device = device
        self.models_path = models_path
        self.controlnet_models_path = controlnet_models_path

    def load_scheduler(self, sampler, config):
        match sampler:
            case 'euler_a':
                scheduler = schedulers.EulerAncestralDiscreteScheduler.from_config(config)
            case 'euler':
                scheduler = schedulers.EulerDiscreteScheduler.from_config(config)
            case 'ddim':
                scheduler = schedulers.DDIMScheduler.from_config(config)
            case 'ddpm':
                scheduler = schedulers.DDPMScheduler.from_config(config)
            case 'uni_pc':
                scheduler = schedulers.UniPCMultistepScheduler.from_config(config)
            case _:
                raise ValueError("Invalid sampler type")

        return scheduler
    
    def load_model(self, model, controlnet_model):
        pipe = pipelines.StableDiffusionControlNetPipeline.from_pretrained(
            os.path.join(self.models_path, model),
            torch_dtype=torch.float16,
            controlnet=self.load_controlnet(controlnet_model),
            safety_checker=None,
        )
            
        return pipe.to(self.device)
    
    def load_controlnet(self, model):
        match model:
            case 'normal':
                model = 'lllyasviel/control_v11p_sd15_normalbae'
            case 'depth':
                model = 'lllyasviel/control_v11f1p_sd15_depth'
            case 'canny':
                model = 'lllyasviel/control_v11p_sd15_canny'
            case 'mlsd':
                model = 'lllyasviel/control_v11p_sd15_mlsd'
            case 'scribble':
                model = 'lllyasviel/control_v11p_sd15_scribble'
            case 'openpose':
                model = 'lllyasviel/control_v11p_sd15_openpose'
            case 'seg':
                model = 'lllyasviel/control_v11p_sd15_seg'
            case 'tile':
                model = 'lllyasviel/control_v11f1e_sd15_tile'
            case 'shuffle':
                model = 'lllyasviel/control_v11e_sd15_shuffle'
            case 'p2p':
                model = 'lllyasviel/control_v11e_sd15_ip2p'
            case _:
                raise ValueError("Invalid controlnet type")

        controlnet = ControlNetModel.from_pretrained(
            model,
            torch_dtype=torch.float16
        )

        return controlnet
    
    def load_generator(self, seed):
        return torch.Generator(self.device).manual_seed(seed)
    
    def latents_callback(self, step, emit_progress):
        emit_progress({ "step": step })

    def random_seed(self):
        return random.randint(-9999999999, 9999999999)

    def text2image(self, properties, controlnet_properties, emit_progress):
        pipe = self.load_model(properties['model'], controlnet_properties['model'])
        pipe.scheduler = self.load_scheduler(properties['sampler'], pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        properties['seed'] = properties['seed'] if properties['seed'] != -1 else self.random_seed()
        generator = self.load_generator(properties['seed'])
        image = PillowUtils.from_base64(controlnet_properties['image'])
        image = pre_processor_image(image, controlnet_properties['model'])
        image = resize_image(image, controlnet_properties['resize_mode'], properties['width'], properties['height'])
        outputs = []

        try:
            for i in range(properties['images']):
                output = pipe(
                    image=image,
                    prompt=properties['positive'],
                    negative_prompt=properties['negative'],
                    num_inference_steps=properties['steps'],
                    guidance_scale=properties['cfg'],
                    num_images_per_prompt=1,
                    width=properties['width'],
                    height=properties['height'],
                    generator=generator,
                    callback_steps=1,
                    callback=lambda s, _, l: self.latents_callback(((s + 1) + (i * properties['steps'])), emit_progress),
                    eta=0.0,
                ).images[0]

                outputs.append(output)
        except Exception as e:
            print(e)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            del pipe
            del generator
            gc.collect()

        return outputs

def pre_processor_image(image, pre_processor):
    model = 'lllyasviel/Annotators'

    match pre_processor:
        case 'normal':
            return controlnet_aux.NormalBaeDetector.from_pretrained(model)(image)
        case 'depth':
            return depth_detector(image)
        case 'canny':
            return controlnet_aux.CannyDetector()(image)
        case 'mlsd':
            return controlnet_aux.MLSDdetector.from_pretrained(model)(image)
        case 'hed':
            return controlnet_aux.HEDdetector.from_pretrained(model)(image)
        case 'scribble':
            return controlnet_aux.PidiNetDetector.from_pretrained(model)(image, safe=True)
        case 'openpose':
            return controlnet_aux.OpenposeDetector.from_pretrained(model)(image, hand_and_face=True)
        case 'seg':
            return controlnet_aux.LineartDetector.from_pretrained(model)(image, coarse=True)
        case 'shuffle':
            return controlnet_aux.ContentShuffleDetector()(image)
        case 'tile':
            return resize_for_condition_image(image, 1024)
        case _:
            return image
 
def resize_image(image, mode, width, height):
    match mode:
        case 'cover':
            return PillowUtils.resize_cover(image, width, height)
        case 'contain':
            return PillowUtils.resize(image, width, height)
        case 'contain_start':
            return PillowUtils.resize_start(image, width, height)
        case 'contain_end':
            return PillowUtils.resize_end(image, width, height)
        case 'fill':
            return PillowUtils.resize_fill(image, width, height)

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def depth_detector(image):
    depth_estimator = pipeline('depth-estimation')
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image