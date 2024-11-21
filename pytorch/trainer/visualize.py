import matplotlib.pyplot as plt
import numpy as np
import os

def normalize_image(img):
    """
    이미지 정규화: [0, 1] 범위로 변환
    """
    img = img - img.min()
    img = img / (img.max() + 1e-7)
    return img

def visualize_and_save_images(data_loader, class_names, save_dir=None, max_visualizations=3):
    """
    증강된 이미지와 마스크를 덮은 이미지를 각각 저장.

    Args:
        data_loader: DataLoader 객체
        class_names: 클래스 이름 리스트
        save_dir: 저장할 디렉토리 경로 (None이면 저장하지 않음)
        max_visualizations: 최대 시각화 횟수
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True) 

    visualization_count = 0  

    for idx, (img, mask) in enumerate(data_loader):
        if visualization_count >= max_visualizations:
            break 

        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        mask_np = mask[0].cpu().numpy() 

        if img_np.shape[:2] != mask_np.shape[1:]:
            raise ValueError("이미지와 마스크의 크기가 다릅니다!")
        
        img_np = normalize_image(img_np)

        overlay = np.zeros_like(img_np, dtype=np.float32)
        colors = plt.cm.get_cmap('tab10', len(class_names)) 

        for i in range(mask_np.shape[0]):
            overlay += np.expand_dims(mask_np[i], axis=-1) * np.array(colors(i)[:3])

        overlay = np.clip(overlay, 0, 1)

        if save_dir:
            original_path = os.path.join(save_dir, f"augmented_image_{idx}.png")
            plt.imsave(original_path, img_np)
            print(f"Saved augmented image: {original_path}")

            overlay_path = os.path.join(save_dir, f"augmented_with_mask_{idx}.png")
            plt.figure(figsize=(10, 5))
            plt.imshow(img_np, alpha=0.7) 
            plt.imshow(overlay, alpha=0.3)
            plt.axis("off")
            plt.savefig(overlay_path, bbox_inches="tight")
            plt.close()
            print(f"Saved augmented image with mask: {overlay_path}")

        visualization_count += 1
