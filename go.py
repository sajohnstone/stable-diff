from diffusers import DiffusionPipeline
import torch

if __name__ == "__main__":
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        use_safetensors=True, 
    )

    # If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda"):
    pipe.enable_model_cpu_offload

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    prompt = "Image of Cat, gazing wistfully out a window, dressed in a black panther suit, bathed in the soft glow of a natural evening light from a front-angle view. close-up RAW photo, detailed textures, sharp focus, ultra-high pixel detail, intricate, realistic, movie scene, cinematic, high-quality, full colors, incredibly detailed, 4k, 8k, 16k, hyper-realistic, RAW photo, masterpiece, ultra-detailed, professionally color graded, professional photography"
    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
    ).images[0]

    image.save("geeks.jpg")

    print("end")