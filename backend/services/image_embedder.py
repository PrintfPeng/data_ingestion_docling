import torch
from PIL import Image
import open_clip
import os
import logging

class ImageEmbedder:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        """
        โหลดโมเดล OpenCLIP สำหรับแปลงรูปภาพเป็น Vector
        """
        self.logger = logging.getLogger(__name__)
        print(f"🔄 Loading Image Embedding Model: {model_name}...")
        
        # ตรวจสอบ Device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        try:
            # โหลดโมเดล
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            
            # ✅ FIX: ถ้าใช้ CPU ให้บังคับแปลงเป็น float32 (แก้ปัญหา BFloat16 Crash)
            if self.device == "cpu":
                self.model.float()

            self.model.eval() # ปรับเป็นโหมด Evaluate
            print(f"✅ Model Loaded on {self.device} (Force Float32)")
            
        except Exception as e:
            print(f"❌ Error loading OpenCLIP model: {e}")
            raise e

    def embed_image(self, image_path):
        """
        รับ Path ของรูปภาพ -> ส่งกลับมาเป็น Vector (List of floats)
        """
        try:
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                return None

            # 1. เปิดและแปลงภาพ
            # สำคัญ: ต้องไม่ส่ง device เข้าไปใน preprocess ถ้าเป็น CPU 
            # แต่ปกติ preprocess จะคืนค่าเป็น Tensor CPU อยู่แล้ว
            image_tensor = self.preprocess(Image.open(image_path)).unsqueeze(0)
            
            # ส่งเข้า Device ที่ถูกต้อง
            image_tensor = image_tensor.to(self.device)

            # 2. คำนวณ Vector
            with torch.no_grad():
                # ถ้าเป็น CPU ไม่ต้องใช้ autocast (เพราะเราแปลงเป็น float32 แล้ว)
                if self.device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        image_features = self.model.encode_image(image_tensor)
                else:
                    image_features = self.model.encode_image(image_tensor)
                
                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # 3. Return
            return image_features.cpu().numpy().tolist()[0]

        except Exception as e:
            print(f"❌ Error embedding image {Path(image_path).name}: {e}")
            return None

    def embed_text(self, text):
        """
        แปลง Text เป็น Vector (ใช้สำหรับ Search)
        """
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_input = tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            if self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    text_features = self.model.encode_text(text_input)
            else:
                text_features = self.model.encode_text(text_input)
                
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy().tolist()[0]