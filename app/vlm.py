import io
import os

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen2.5-vl-7b")
MAX_PIXELS = int(os.environ.get("VLM_MAX_PIXELS", str(1280 * 28 * 28)))
DEFAULT_SYSTEM = "You are a helpful assistant."


class VLM:
    """Qwen2.5-VL held resident on the GPU; answers text-only or text+image prompts.

    The prompt is assembled by hand rather than through AutoProcessor: the processor also builds a video
    processor that needs torchvision, which the mandated base image does not ship. The image processor
    used here is the PIL/numpy one, so torch and PIL are all this needs.
    """

    def __init__(self):
        self.ok = False
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
            self.torch = torch
            dev = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
            self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.improc = Qwen2VLImageProcessor.from_pretrained(MODEL_DIR, min_pixels=64 * 28 * 28, max_pixels=MAX_PIXELS)
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=dtype)
            except TypeError:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=dtype)
            self.model.to(dev).eval()
            self.dev = dev
            self.ok = True
            self.generate("Reply with OK.", max_new=4)
            print(f"[vlm] loaded on {dev}", flush=True)
        except Exception as e:
            self.ok = False
            print(f"[vlm] unavailable: {e!r}", flush=True)

    @staticmethod
    def load_image(src):
        from PIL import Image
        im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        return im.convert("RGB")

    def generate(self, prompt, images=(), max_new=200, system=None):
        images = list(images)
        vision, extra = "", {}
        if images:
            feats = self.improc(images=images, return_tensors="pt")
            merge = self.improc.merge_size ** 2
            for thw in feats["image_grid_thw"]:
                vision += "<|vision_start|>" + "<|image_pad|>" * (int(thw.prod()) // merge) +"<|vision_end|>"
            extra = {"pixel_values": feats["pixel_values"].to(self.dev, self.model.dtype),
                     "image_grid_thw": feats["image_grid_thw"].to(self.dev)}
        text = (f"<|im_start|>system\n{system or DEFAULT_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{vision}{prompt}<|im_end|>\n<|im_start|>assistant\n")
        enc = self.tok(text, return_tensors="pt").to(self.dev)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, **extra, max_new_tokens=max_new, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
