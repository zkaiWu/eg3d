import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

img_path = "/home2/zhongkaiwu/data/dreamfusion_data/eg3d_generation_data/seed0001/images/199.png"
raw_image = Image.open(img_path).convert('RGB')

question = "Write a very detailed description that goes along this photo"
# question = ""
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)


