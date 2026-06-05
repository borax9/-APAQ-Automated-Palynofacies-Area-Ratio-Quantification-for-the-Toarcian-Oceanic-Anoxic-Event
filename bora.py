import cv2
import os
import random
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# --- AYARLAR ---
IMAGE_DIR = Path(r"C:\dataset_split_3\valid\images")
LABEL_DIR = Path(r"C:\dataset_split_3\valid\labels")

OUTPUT_IMG_DIR = Path(r"C:\dataset_split_3\images\valid_augmented")
OUTPUT_LAB_DIR = Path(r"C:\dataset_split_3\labels\valid_augmented")

# Orijinal 500 fotonun yanına her fotodan 4 yeni varyasyon üretilir (Toplam 2500 foto)
AUGMENT_COUNT_PER_IMAGE = 4 

# --- Sadece Geometriyi Koruyan ve Dijital Çeşitlilik Sağlayan Pipeline ---
transform = A.Compose([
    # 1. Geometriyi ve Dikdörtgen Yapıyı Kesinlikle Bozmayan Aynalama ve Tam Dönüşler
    A.HorizontalFlip(p=0.5),      # Sağ-sol aynalama (Mirror)
    A.VerticalFlip(p=0.5),        # Yukarı-aşağı aynalama (Mirror)
    A.RandomRotate90(p=1.0),      # Sadece 90, 180, 270 derece tam tur döndürür (Dikdörtgeni bozmaz!)

    # 2. Siyah Şeyl (Black Shale) ve AOM/Phytoclast İçin Gelişmiş Dijital Artırımlar (Pattern Recognition)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6), # Parlaklık ve Kontrast
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.4),  # Exposure/Pozlama çeşitliliği
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),                                    # Mikroskop odak fluluğu (Blur)
    A.Sharpen(alpha=(0.1, 0.4), p=0.3),                                          # Phytoclast sınır netleştirme
    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),                      # Doku (texture) belirginleştirme
    A.ISONoise(color_noise_ratio=0.1, p=0.2)                                     # Kamera sensör gürültüsü
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))


def custom_mosaic(img_paths, output_size=(1024, 1024)):
    """AOM ve Phytoclast'ların yan yana karmaşık dizilimlerini öğretmek için 4'lü Mozaikleme"""
    selected = random.sample(img_paths, 4)
    imgs = [cv2.imread(str(p)) for p in selected]
    
    # Tüm resimleri standart mozaik boyutunun çeyreğine (512x512) boyutlandır (bozulma olmaması için kare)
    h, w = output_size[0] // 2, output_size[1] // 2
    resized_imgs = [cv2.resize(img, (w, h)) for img in imgs]
    
    # 4 resmi birleştir
    top = np.hstack((resized_imgs[0], resized_imgs[1]))
    bottom = np.hstack((resized_imgs[2], resized_imgs[3]))
    mosaic_img = np.vstack((top, bottom))
    
    # Bu mozaik resim için etiketleri de birleştirmek gerekir. Ancak mozaiklerin etiket koordinat
    # hesaplaması çok karmaşık olduğundan, bu fonksiyonu sadece görsel arka plan zenginliği 
    # veya YOLO'nun kendi içindeki mosaic=1.0 parametresine bırakmak daha sağlıklıdır.
    # Bu yüzden fiziksel üretimde veri bütünlüğü adına YOLO'nun dahili mozaiğini kullanacağız.
    return mosaic_img


def read_yolo_labels(label_path):
    bboxes = []
    class_ids = []
    if not label_path.exists():
        return bboxes, class_ids
        
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Segmentasyon poligonu kontrolü (4 değerden fazlaysa box'a dönüştür)
            if len(coords) > 4:
                xs = coords[0::2]
                ys = coords[1::2]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bw = xmax - xmin
                bh = ymax - ymin
                bx = xmin + bw/2
                by = ymin + bh/2
            else:
                bx, by, bw, bh = coords
                
            # Albumentations sınır taşma koruması
            bw = min(0.99, max(0.01, bw))
            bh = min(0.99, max(0.01, bh))
            bx = min(0.99, max(0.01, bx))
            by = min(0.99, max(0.01, by))
            
            bboxes.append([bx, by, bw, bh])
            class_ids.append(cls_id)
            
    return bboxes, class_ids


def save_yolo_labels(output_path, bboxes, class_ids):
    with open(output_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def main():
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LAB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Küçük/büyük harf uyumluluğu için tüm uzantıları tara
    img_paths = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        img_paths.extend(list(IMAGE_DIR.glob(ext)))
        
    print(f"📦 {len(img_paths)} adet orijinal fotoğraf bulundu. Güvenli çoğaltma başlıyor...")

    if len(img_paths) == 0:
        print(f"❌ HATA: {IMAGE_DIR} klasörünün içinde fotoğraf bulunamadı! Lütfen yolu kontrol edin.")
        return

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        lbl_path = LABEL_DIR / f"{img_path.stem}.txt"
        bboxes, class_ids = read_yolo_labels(lbl_path)
        
        # 1. Orijinal dosyaları yeni klasöre güvenle kopyala
        cv2.imwrite(str(OUTPUT_IMG_DIR / img_path.name), img)
        if lbl_path.exists():
            import shutil
            shutil.copy(lbl_path, OUTPUT_LAB_DIR / lbl_path.name)
            
        # 2. Geometriyi bozmayan dijital türetme döngüsü
        for i in range(AUGMENT_COUNT_PER_IMAGE):
            try:
                augmented = transform(image=img, bboxes=bboxes, class_ids=class_ids)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_ids = augmented['class_ids']
                
                new_base_name = f"{img_path.stem}_aug_{i}"
                
                cv2.imwrite(str(OUTPUT_IMG_DIR / f"{new_base_name}{img_path.suffix}"), aug_img)
                save_yolo_labels(OUTPUT_LAB_DIR / f"{new_base_name}.txt", aug_bboxes, aug_class_ids)
            except Exception:
                continue

    print(f"\n✅ İşlem başarıyla tamamlandı!")
    print(f"📁 Görüntüler (Orijinal + Yeni): {OUTPUT_IMG_DIR}")
    print(f"📁 Etiketler  (Orijinal + Yeni): {OUTPUT_LAB_DIR}")


if __name__ == "__main__":
    main()