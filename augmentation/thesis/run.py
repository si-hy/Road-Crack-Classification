import os
import json
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from collections import Counter, defaultdict
import random
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys

# matplotlib 한글 폰트 설정을 위한 font_manager 임포트
import matplotlib.font_manager as fm

# scipy 임포트 (AdvancedBlending 클래스에서 사용)
from scipy.ndimage import distance_transform_edt

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 한글 폰트 설정 함수 (이전과 동일) ---
def set_korean_font():
    """Matplotlib에 한글 폰트를 설정합니다."""
    try:
        if sys.platform == "win32":  # Windows
            font_name = None
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if 'Malgun Gothic' in available_fonts:
                font_name = 'Malgun Gothic'
            else:
                font_path_win = "c:/Windows/Fonts/malgun.ttf"
                if os.path.exists(font_path_win):
                    try:
                        font_prop = fm.FontProperties(fname=font_path_win)
                        font_name = font_prop.get_name()
                    except Exception as e:
                        logger.warning(f"Windows malgun.ttf 파일에서 폰트 이름 가져오기 실패: {e}")
                else:
                    logger.warning(f"Windows에서 'Malgun Gothic' 폰트를 찾을 수 없습니다. 경로: {font_path_win}")
            
            if font_name:
                plt.rc("font", family=font_name)
                logger.info(f"Windows에서 '{font_name}' 폰트를 설정했습니다.")
            else:
                logger.error("Windows에서 한글 폰트를 설정하지 못했습니다. 시각화 시 한글이 깨질 수 있습니다.")

        elif sys.platform == "darwin":  # macOS
            font_name = 'AppleGothic'
            try:
                plt.rc("font", family=font_name)
                logger.info(f"macOS에서 '{font_name}' 폰트를 설정했습니다.")
            except RuntimeError:
                logger.warning(f"macOS에서 '{font_name}' 폰트를 찾을 수 없습니다. 다른 한글 폰트를 확인해주세요.")

        elif sys.platform.startswith("linux"):  # Linux
            font_path_linux = None
            nanum_fonts = [f for f in fm.fontManager.ttflist if 'NanumGothic' in f.name]
            if nanum_fonts:
                font_path_linux = nanum_fonts[0].fname
                font_name = fm.FontProperties(fname=font_path_linux).get_name()
                plt.rc("font", family=font_name)
                logger.info(f"Linux에서 '{font_name}' (경로: {font_path_linux}) 폰트를 설정했습니다.")
            else:
                font_paths_linux_fallback = [
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                    "/usr/share/fonts/nanum/NanumGothic.ttf",
                ]
                for path_option in font_paths_linux_fallback:
                    if os.path.exists(path_option):
                        font_path_linux = path_option
                        break
                if font_path_linux:
                    font_name = fm.FontProperties(fname=font_path_linux).get_name()
                    plt.rc("font", family=font_name)
                    logger.info(f"Linux에서 '{font_name}' (경로: {font_path_linux}) 폰트를 설정했습니다.")
                else:
                    logger.error("Linux에서 NanumGothic 폰트를 찾을 수 없습니다. 'sudo apt-get install fonts-nanum*'으로 설치해주세요.")
        else:
            logger.warning(f"지원되지 않는 OS 플랫폼({sys.platform})입니다. 한글 폰트가 제대로 설정되지 않을 수 있습니다.")
        
        plt.rc("axes", unicode_minus=False)
        
    except Exception as e:
        logger.error(f"한글 폰트 설정 중 오류 발생: {e}")
        logger.warning("기본 폰트로 시도합니다. 한글이 깨질 수 있습니다.")

class AdvancedBlending:
    """정교한 블렌딩 기법을 적용한 Copy-Paste"""
    
    @staticmethod
    def feather_edges(mask, feather_amount=10):
        """마스크 가장자리를 부드럽게 페더링"""
        if mask is None or mask.size == 0:
            logger.warning("페더링할 마스크가 비어있습니다.")
            return mask
        if np.all(mask == 0):
            return mask.astype(float)
            
        binary_mask = (mask > 0).astype(np.uint8)
        dist_transform = distance_transform_edt(binary_mask)
        feather_amount_safe = max(feather_amount, 1e-5)
        feathered = np.minimum(dist_transform / feather_amount_safe, 1.0)
        return feathered
    
    @staticmethod
    def poisson_blend(obj_img, background, obj_binary_mask, center_coords):
        """포아송 블렌딩 (Seamless Cloning)"""
        if obj_img is None or obj_img.size == 0: return background
        if background is None or background.size == 0: return background
        if obj_binary_mask is None or obj_binary_mask.size == 0: return background

        mask_for_poisson = (obj_binary_mask > 0).astype(np.uint8) * 255
        
        if np.sum(mask_for_poisson) == 0:
            logger.warning("포아송 블렌딩을 위한 마스크가 비어있습니다. 원본 배경을 반환합니다.")
            return background.copy()

        try:
            # seamlessClone은 입력 이미지와 마스크의 크기가 같아야 함
            if obj_img.shape[:2] != mask_for_poisson.shape[:2]:
                logger.warning(f"Poisson Blend: 객체 이미지({obj_img.shape[:2]})와 마스크({mask_for_poisson.shape[:2]}) 크기가 다릅니다. 마스크를 객체 크기로 조정합니다.")
                mask_for_poisson = cv2.resize(mask_for_poisson, (obj_img.shape[1], obj_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # center_coords가 이미지 경계 내에 있는지 확인 및 조정
            h_bg, w_bg = background.shape[:2]
            h_obj, w_obj = obj_img.shape[:2]
            
            if not (0 <= center_coords[0] < w_bg and 0 <= center_coords[1] < h_bg):
                logger.error(f"Poisson Blend: 중심점 {center_coords}이 배경 크기 {background.shape[:2]} 밖에 있습니다.")
                return background.copy()

            result = cv2.seamlessClone(
                obj_img,
                background,
                mask_for_poisson,
                center_coords,
                cv2.NORMAL_CLONE
            )
            return result
        except cv2.error as e:
            logger.error(f"포아송 블렌딩 오류: {e}. 객체 크기: {obj_img.shape}, 마스크 크기: {mask_for_poisson.shape}, 배경 크기: {background.shape}, 중심: {center_coords}")
            return background.copy()

    @staticmethod
    def multiband_blend(background_roi, obj_img_aligned, obj_mask_aligned_0_1_float_3ch, levels=4):
        """멀티밴드 블렌딩 (Laplacian Pyramid)"""
        if background_roi is None or obj_img_aligned is None or obj_mask_aligned_0_1_float_3ch is None or \
           background_roi.size == 0 or obj_img_aligned.size == 0 or obj_mask_aligned_0_1_float_3ch.size == 0:
            logger.warning("멀티밴드 블렌딩 입력값이 유효하지 않습니다.")
            return background_roi
        
        if background_roi.shape != obj_img_aligned.shape or background_roi.shape != obj_mask_aligned_0_1_float_3ch.shape:
            logger.warning("멀티밴드 블렌딩: 입력 이미지/마스크 크기가 일치하지 않습니다.")
            return background_roi

        gpA = [background_roi.astype(np.float32)]
        gpB = [obj_img_aligned.astype(np.float32)]
        gpM = [obj_mask_aligned_0_1_float_3ch.astype(np.float32)]

        current_levels = 0
        for i in range(levels):
            if gpA[i].shape[0] < 2 or gpA[i].shape[1] < 2 or \
               gpB[i].shape[0] < 2 or gpB[i].shape[1] < 2 or \
               gpM[i].shape[0] < 2 or gpM[i].shape[1] < 2:
                logger.warning(f"멀티밴드 블렌딩 중 피라미드 레벨 {i+1}에서 이미지 크기가 너무 작아 현재 레벨({i})까지만 처리합니다.")
                levels = i
                break
            gpA.append(cv2.pyrDown(gpA[i]))
            gpB.append(cv2.pyrDown(gpB[i]))
            gpM.append(cv2.pyrDown(gpM[i]))
            current_levels +=1
        
        if current_levels == 0 and levels > 0 :
            logger.warning("멀티밴드 블렌딩: 이미지 크기가 너무 작아 피라미드를 생성할 수 없습니다. 단순 알파 블렌딩으로 대체합니다.")
            blended_roi_content = background_roi * (1 - obj_mask_aligned_0_1_float_3ch) + obj_img_aligned * obj_mask_aligned_0_1_float_3ch
            return np.clip(blended_roi_content, 0, 255).astype(np.uint8)

        lpA = [gpA[levels]]
        lpB = [gpB[levels]]
        for i in range(levels, 0, -1):
            size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
            lpA.append(cv2.subtract(gpA[i-1], cv2.pyrUp(gpA[i], dstsize=size)))
            lpB.append(cv2.subtract(gpB[i-1], cv2.pyrUp(gpB[i], dstsize=size)))
        
        LS = []
        for i in range(levels + 1):
            la_current = lpA[i]
            lb_current = lpB[i]
            gm_current = gpM[levels-i]
            
            if la_current.shape != gm_current.shape or lb_current.shape != gm_current.shape:
                logger.warning(f"멀티밴드 블렌드 중 레벨 {levels-i}에서 크기 불일치. 마스크 크기 조정 시도.")
                gm_current = cv2.resize(gm_current, (la_current.shape[1], la_current.shape[0]), interpolation=cv2.INTER_LINEAR)
                if gm_current.ndim == 2 and la_current.ndim == 3:
                    gm_current = np.stack([gm_current]*3, axis=-1)

            ls = la_current * (1.0 - gm_current) + lb_current * gm_current
            LS.append(ls)
        
        ls_ = LS[0]
        for i in range(1, levels + 1):
            size = (LS[i].shape[1], LS[i].shape[0])
            ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size), LS[i])
        
        return np.clip(ls_, 0, 255).astype(np.uint8)
    
    # 이 함수는 AdvancedBlending 클래스 내부에 위치해야 합니다.
    def blend_object_onto_background(self, background_orig, obj_img_transformed, obj_mask_transformed_binary,
                                        obj_points_transformed_abs, paste_x, paste_y, new_w, new_h,
                                        blend_mode='advanced_alpha'):
        output_image = background_orig.copy()
        y_start, y_end = int(paste_y), int(paste_y + new_h)
        x_start, x_end = int(paste_x), int(paste_x + new_w)

        h_bg, w_bg = output_image.shape[:2]
        if y_start < 0 or x_start < 0 or y_end > h_bg or x_end > w_bg:
            logger.error(f"블렌딩 ROI가 이미지 경계를 벗어납니다. ROI: ({x_start},{y_start})-({x_end},{y_end}), BG: ({w_bg},{h_bg})")
            return output_image, obj_points_transformed_abs

        roi_background = output_image[y_start:y_end, x_start:x_end]

        if obj_img_transformed is None or obj_img_transformed.size == 0 or \
        obj_mask_transformed_binary is None or obj_mask_transformed_binary.size == 0:
            logger.warning("블렌딩할 객체 이미지 또는 마스크가 비어있습니다.")
            return output_image, obj_points_transformed_abs

        if roi_background.shape[:2] != obj_img_transformed.shape[:2]:
            logger.debug(f"블렌딩 전 ROI({roi_background.shape[:2]})와 객체({obj_img_transformed.shape[:2]}) 크기 불일치. 객체/마스크를 ROI 크기로 조정.")
            obj_img_transformed = cv2.resize(obj_img_transformed, (roi_background.shape[1], roi_background.shape[0]), interpolation=cv2.INTER_LINEAR)
            obj_mask_transformed_binary = cv2.resize(obj_mask_transformed_binary, (roi_background.shape[1], roi_background.shape[0]), interpolation=cv2.INTER_NEAREST)

        harmonized_obj_img = obj_img_transformed.copy()
        if blend_mode in ['advanced_alpha', 'color_match_alpha', 'poisson_harmonized', 'multiband_harmonized']:
            try:
                obj_lab = cv2.cvtColor(harmonized_obj_img, cv2.COLOR_BGR2LAB).astype(np.float32)
                roi_lab = cv2.cvtColor(roi_background, cv2.COLOR_BGR2LAB).astype(np.float32)
                obj_pixels_lab = obj_lab[obj_mask_transformed_binary > 0]
                if obj_pixels_lab.size > 0:
                    obj_mean = np.mean(obj_pixels_lab, axis=0); obj_std = np.std(obj_pixels_lab, axis=0)
                    roi_mean = np.mean(roi_lab, axis=(0, 1)); roi_std = np.std(roi_lab, axis=(0, 1))
                    for i in range(3):
                        obj_lab[:, :, i] = np.clip(
                            (obj_lab[:, :, i] - obj_mean[i]) * (roi_std[i] / (obj_std[i] + 1e-5)) + roi_mean[i],
                            0, 255
                        )
                    harmonized_obj_img = cv2.cvtColor(obj_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                else: logger.debug("색상 조화를 위한 객체 픽셀이 없습니다.")
            except cv2.error as e: logger.warning(f"색상 조화 중 OpenCV 오류: {e}")

            if blend_mode in ['advanced_alpha', 'poisson_harmonized', 'multiband_harmonized']:
                try:
                    roi_gray = cv2.cvtColor(roi_background, cv2.COLOR_BGR2GRAY)
                    obj_gray = cv2.cvtColor(harmonized_obj_img, cv2.COLOR_BGR2GRAY)
                    obj_pixels_gray = obj_gray[obj_mask_transformed_binary > 0]
                    if obj_pixels_gray.size > 0 and np.mean(obj_pixels_gray) > 1e-5 :
                        brightness_ratio = np.mean(roi_gray) / (np.mean(obj_pixels_gray) + 1e-5)
                        brightness_ratio = np.clip(brightness_ratio, 0.7, 1.5)
                        harmonized_obj_img = cv2.convertScaleAbs(harmonized_obj_img, alpha=brightness_ratio, beta=0)
                    else: logger.debug("조명 조화를 위한 객체 픽셀이 없거나 평균 밝기가 0에 가깝습니다.")
                except cv2.error as e: logger.warning(f"조명 조화 중 OpenCV 오류: {e}")
        
        if blend_mode == 'poisson' or blend_mode == 'poisson_harmonized':
            center_in_bg_abs = (x_start + new_w // 2, y_start + new_h // 2)
            output_image = self.poisson_blend(harmonized_obj_img, output_image, obj_mask_transformed_binary, center_in_bg_abs)
        
        elif blend_mode == 'multiband' or blend_mode == 'multiband_harmonized':
            mask_0_1_float_3ch = np.stack([obj_mask_transformed_binary.astype(float)/255.0]*3, axis=-1)
            blended_roi_content = self.multiband_blend(roi_background, harmonized_obj_img, mask_0_1_float_3ch)
            output_image[y_start:y_end, x_start:x_end] = blended_roi_content

        else:
            if blend_mode == 'simple_alpha':
                alpha_mask_0_1_float = cv2.GaussianBlur(obj_mask_transformed_binary, (5,5), 0).astype(float) / 255.0
            else:
                feather_amount = max(3, int(min(new_h, new_w) * 0.03))
                mask_feathered = self.feather_edges(obj_mask_transformed_binary, feather_amount)
                blur_ksize = max(3, 2 * int(min(new_h, new_w) * 0.02) + 1)
                mask_blur = cv2.GaussianBlur(mask_feathered, (blur_ksize, blur_ksize), 0)
                final_alpha_mask_0_1_float = np.clip(mask_blur, 0, 1)
                if blend_mode == 'advanced_alpha':
                    grad_x = cv2.Sobel(mask_blur, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(mask_blur, cv2.CV_64F, 0, 1, ksize=3)
                    gradient = np.sqrt(grad_x**2 + grad_y**2)
                    if np.max(gradient) > 1e-5:
                        gradient = gradient / np.max(gradient)
                        final_alpha_mask_0_1_float = final_alpha_mask_0_1_float * (1 - gradient * 0.2)
                        final_alpha_mask_0_1_float = np.clip(final_alpha_mask_0_1_float, 0, 1)
                alpha_mask_0_1_float = final_alpha_mask_0_1_float

            alpha_mask_3ch = np.stack([alpha_mask_0_1_float] * 3, axis=-1)
            blended_roi_content = roi_background * (1 - alpha_mask_3ch) + harmonized_obj_img * alpha_mask_3ch
            output_image[y_start:y_end, x_start:x_end] = blended_roi_content.astype(np.uint8)

            if blend_mode == 'advanced_alpha':
                try:
                    shadow_kernel_size = max(3, int(min(new_h, new_w) * 0.08))
                    shadow_kernel_size = shadow_kernel_size if shadow_kernel_size % 2 != 0 else shadow_kernel_size + 1
                    dilated_mask = cv2.dilate(obj_mask_transformed_binary, np.ones((shadow_kernel_size//2, shadow_kernel_size//2), np.uint8), iterations=1)
                    shadow_alpha_mask = cv2.GaussianBlur(dilated_mask, (shadow_kernel_size, shadow_kernel_size), 0)
                    shadow_alpha_mask = shadow_alpha_mask.astype(float) / 255.0 * 0.15
                    shadow_region_float = output_image[y_start:y_end, x_start:x_end].astype(float)
                    effective_shadow_alpha = np.clip(shadow_alpha_mask - (obj_mask_transformed_binary.astype(float)/255.0), 0, 1)
                    for c in range(3):
                        shadow_region_float[:,:,c] *= (1 - effective_shadow_alpha * 0.7)
                    output_image[y_start:y_end, x_start:x_end] = np.clip(shadow_region_float, 0, 255).astype(np.uint8)
                except Exception as e: logger.warning(f"그림자 효과 적용 중 오류: {e}")
                
        return output_image, obj_points_transformed_abs

    def calculate_optimized_weights(self):
        if not self.original_class_counts:
            logger.warning("원본 클래스 카운트가 없어 가중치를 계산할 수 없습니다. 모든 클래스 동일 가중치(1.0)를 사용합니다.")
            return {cn: 1.0 for cn in self.class_names}
        total_objects = sum(self.original_class_counts.values())
        # 객체가 있는 클래스 수만 계산
        num_classes_with_objects = len([c_name for c_name, count in self.original_class_counts.items() if count > 0])

        if total_objects == 0 or num_classes_with_objects == 0:
            logger.warning("객체가 없거나 객체가 있는 클래스가 없어 유효한 가중치를 계산할 수 없습니다. 모든 클래스 동일 가중치(1.0)를 사용합니다.")
            return {cn: 1.0 for cn in self.class_names}
        
        weights = {}
        for class_name in self.class_names: # 모든 정의된 클래스에 대해
            count = self.original_class_counts.get(class_name, 0)
            if count > 0:
                weight = np.sqrt(total_objects / (num_classes_with_objects * count))
            else:
                weight = 0 # 객체 없는 클래스는 일단 0
            weights[class_name] = weight
        
        # 객체가 없는 클래스에 대한 가중치 후처리 (다른 클래스 최대 가중치의 1.5배)
        valid_weights = [w for w in weights.values() if w > 0]
        max_calculated_weight = max(valid_weights) if valid_weights else 1.0 # 유효한 가중치가 없으면 기본값 1.0
        
        for class_name in self.class_names:
            if weights[class_name] == 0:
                weights[class_name] = max_calculated_weight * 1.5
        
        logger.info(f"최적화된 증강 가중치: {weights}")
        return weights

'''
    def apply_geometric_transform(self, image, shapes, transform_prob=0.8):
        """
        이미지와 모든 shape의 폴리곤 좌표에 동일한 기하 변형(Elastic, Grid)을 적용합니다.
        shapes: [{'label': 'name', 'points': [[x1,y1], ...], ...}, ...]
        """
        if random.random() >= transform_prob: 
            return image, shapes

        transform_pipeline = A.Compose([
            A.ElasticTransform(alpha=100, sigma=100 * 0.06, alpha_affine=100 * 0.04, p=0.7, 
                               border_mode=cv2.BORDER_REFLECT_101),
            A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.7, 
                               border_mode=cv2.BORDER_REFLECT_101),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['shape_indices'], remove_invisible=False))

        all_keypoints_flat = []
        keypoint_shape_indices = [] 
        points_per_shape_count = [] 

        for idx, shape_dict in enumerate(shapes):
            points = shape_dict.get('points', [])
            if points and len(points) >=3 : 
                all_keypoints_flat.extend(points) 
                keypoint_shape_indices.extend([idx] * len(points)) 
                points_per_shape_count.append(len(points))
            else:
                points_per_shape_count.append(0) 

        if not all_keypoints_flat: 
            img_only_transform_pipeline = A.Compose([
                A.ElasticTransform(alpha=100, sigma=100 * 0.06, alpha_affine=100 * 0.04, p=0.7,
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.7,
                                   border_mode=cv2.BORDER_REFLECT_101),
            ])
            transformed_image = img_only_transform_pipeline(image=image)['image']
            return transformed_image, shapes 

        try:
            transformed_data = transform_pipeline(image=image, keypoints=all_keypoints_flat, shape_indices=keypoint_shape_indices)
            transformed_image = transformed_data['image']
            transformed_keypoints_flat = transformed_data['keypoints']

            new_shapes = []
            current_kp_idx = 0
            for shape_idx, original_shape_dict in enumerate(shapes):
                num_points_for_this_shape = points_per_shape_count[shape_idx]
                new_shape = original_shape_dict.copy()
                if num_points_for_this_shape > 0:
                    shape_keypoints = transformed_keypoints_flat[current_kp_idx : current_kp_idx + num_points_for_this_shape]
                    new_shape['points'] = np.array(shape_keypoints, dtype=np.int32).tolist()
                    current_kp_idx += num_points_for_this_shape
                else: 
                    new_shape['points'] = [] 
                new_shapes.append(new_shape)
            
            return transformed_image, new_shapes

        except Exception as e:
            logger.error(f"기하 변형 중 오류 발생: {e}. 이미지 변형만 시도합니다.")
            img_only_transform_pipeline = A.Compose([
                A.ElasticTransform(alpha=100, sigma=100 * 0.06, alpha_affine=100 * 0.04, p=1.0, 
                                   border_mode=cv2.BORDER_REFLECT_101),
                A.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0, 
                                   border_mode=cv2.BORDER_REFLECT_101),
            ])
            transformed_image = img_only_transform_pipeline(image=image)['image']
            return transformed_image, shapes 
'''
# 기존의 apply_geometric_transform 함수를 아래 함수로 교체하세요.
# 이름도 역할에 맞게 apply_combined_augmentations로 변경했습니다.

    def apply_combined_augmentations(self, image, shapes, transform_prob=0.8):
        """
        이미지와 폴리곤 좌표에 기존 증강(Elastic, Grid)과 YOLO 스타일 증강을 함께 적용합니다.
        """
        if random.random() >= transform_prob:
            return image, shapes

        # 🚀 기존 파이프라인에 YOLO 스타일 증강 기법을 추가
        transform_pipeline = A.Compose([
            # --- 🆕 추가된 YOLO 스타일 증강 ---
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.7,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),

            # --- ✅ 유지되는 기존 증강 ---
            A.ElasticTransform(alpha=100, sigma=100 * 0.06, alpha_affine=100 * 0.04, p=0.7,
                            border_mode=cv2.BORDER_REFLECT_101),
            A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.7,
                            border_mode=cv2.BORDER_REFLECT_101),

        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['shape_indices'], remove_invisible=False))


        # --- 아래 로직은 기존과 동일하게 유지됩니다 ---
        all_keypoints_flat = []
        keypoint_shape_indices = []
        points_per_shape_count = []

        for idx, shape_dict in enumerate(shapes):
            points = shape_dict.get('points', [])
            if points and len(points) >= 3:
                all_keypoints_flat.extend(points)
                keypoint_shape_indices.extend([idx] * len(points))
                points_per_shape_count.append(len(points))
            else:
                points_per_shape_count.append(0)

        if not all_keypoints_flat:
            img_only_transform_pipeline = A.Compose([t for t in transform_pipeline.transforms if not isinstance(t, A.KeypointParams)])
            transformed_image = img_only_transform_pipeline(image=image)['image']
            return transformed_image, shapes

        try:
            transformed_data = transform_pipeline(image=image, keypoints=all_keypoints_flat, shape_indices=keypoint_shape_indices)
            transformed_image = transformed_data['image']
            transformed_keypoints_flat = transformed_data['keypoints']

            new_shapes = []
            current_kp_idx = 0
            for shape_idx, original_shape_dict in enumerate(shapes):
                num_points_for_this_shape = points_per_shape_count[shape_idx]
                new_shape = original_shape_dict.copy()
                if num_points_for_this_shape > 0:
                    shape_keypoints = transformed_keypoints_flat[current_kp_idx : current_kp_idx + num_points_for_this_shape]
                    new_shape['points'] = np.array(shape_keypoints, dtype=np.int32).tolist()
                    current_kp_idx += num_points_for_this_shape
                else:
                    new_shape['points'] = []
                new_shapes.append(new_shape)
            
            return transformed_image, new_shapes

        except Exception as e:
            logger.error(f"결합된 증강 중 오류 발생: {e}. 이미지 변형만 시도합니다.")
            img_only_transform_pipeline = A.Compose([t for t in transform_pipeline.transforms if not isinstance(t, A.KeypointParams)])
            transformed_image = img_only_transform_pipeline(image=image)['image']
            return transformed_image, shapes

    def calculate_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if intersection_area == 0: return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def extract_objects_from_dataset(self):
        logger.info("고품질 객체 추출 중...")
        self.class_objects.clear()
        extracted_count = 0
        json_files = list(self.labels_dir.glob('*.json'))
        if not json_files:
            logger.warning(f"{self.labels_dir} 에서 JSON 라벨 파일을 찾을 수 없습니다. 객체 추출을 건너뜁니다.")
            return
        for idx, json_file in enumerate(json_files):
            if idx % 50 == 0: logger.info(f"객체 추출 진행: {idx}/{len(json_files)}")
            try:
                base_name = json_file.stem
                img_file, _ = self._find_image_file(base_name)
                if not img_file:
                    logger.warning(f"객체 추출을 위한 이미지 파일 없음: {self.images_dir / base_name}")
                    continue
                image = cv2.imread(str(img_file))
                if image is None:
                    logger.warning(f"이미지 로드 실패: {img_file}")
                    continue
                with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
                if 'shapes' not in data: continue
                for shape in data['shapes']:
                    if shape.get('shape_type') != 'polygon' or 'label' not in shape: continue
                    label = shape['label']
                    if label not in self.class_to_idx:
                        logger.debug(f"객체 추출 중 정의되지 않은 라벨 '{label}' 발견: {json_file.name}")
                        continue
                    points_list = shape.get('points', [])
                    if not points_list or len(points_list) < 3: continue
                    points = np.array(points_list, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(points)
                    if not (self.min_object_size <= w < image.shape[1] * self.max_object_ratio and \
                            self.min_object_size <= h < image.shape[0] * self.max_object_ratio):
                        continue
                    if h == 0: continue
                    aspect_ratio = w / h
                    if not (0.2 < aspect_ratio < 5.0): continue
                    obj_region_mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(obj_region_mask_full, [points], 255)
                    obj_img_cropped = image[y:y+h, x:x+w].copy()
                    obj_mask_cropped = obj_region_mask_full[y:y+h, x:x+w].copy() 
                    obj_img_masked = cv2.bitwise_and(obj_img_cropped, obj_img_cropped, mask=obj_mask_cropped)
                    relative_points = points - np.array([x, y])
                    self.class_objects[label].append({
                        'image': obj_img_masked,    
                        'mask': obj_mask_cropped,   
                        'points': relative_points,  
                    })
                    extracted_count += 1
            except Exception as e: logger.exception(f"객체 추출 중 오류: {json_file} - {e}")
        logger.info(f"총 {extracted_count}개의 객체 추출 완료.")
        for class_name, objects in self.class_objects.items():
            logger.info(f"   - {class_name}: {len(objects)}개")

    def _find_image_file(self, base_name):
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', 
                               '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
        for ext in possible_extensions:
            potential_file = self.images_dir / f"{base_name}{ext}"
            if potential_file.exists():
                return potential_file, ext
        return None, None

    def intelligent_copy_paste_with_advanced_blending(self, background_orig, 
                                                  class_weights, 
                                                  num_pastes_range=(1, 4), 
                                                  difficulty_level='medium',
                                                  blend_mode='advanced_alpha'):
        if not any(self.class_objects.values()):
            logger.warning("Copy-Paste를 위한 추출된 객체가 없습니다.")
            return background_orig, [] 

        output_image = background_orig.copy()
        h_bg, w_bg = output_image.shape[:2]
        pasted_shapes_info = [] 

        min_pastes, max_pastes = num_pastes_range
        if difficulty_level == 'easy':
            num_pastes_actual = random.randint(min_pastes, max(min_pastes, (min_pastes + max_pastes) // 3))
        elif difficulty_level == 'medium':
            num_pastes_actual = random.randint(min_pastes, max_pastes)
        else: 
            num_pastes_actual = random.randint(max_pastes, int(max_pastes * 1.5))
            num_pastes_actual = min(num_pastes_actual, 8) 
        
        if not class_weights or not any(v > 0 for v in class_weights.values()):
            logger.warning("유효한 클래스 가중치가 없어 Copy-Paste를 건너뜁니다.")
            return output_image, []

        classes_with_objects_and_weights = [cn for cn in class_weights if class_weights.get(cn, 0) > 0 and self.class_objects.get(cn)]
        if not classes_with_objects_and_weights:
            logger.warning("붙여넣을 수 있는 객체가 있는 클래스가 없거나 가중치가 없습니다.")
            return output_image, []
        weights_for_choice = [class_weights[cn] for cn in classes_with_objects_and_weights]

        occupied_bboxes = [] 
        successfully_pasted_count = 0

        # --- 🎯 핵심 수정: 양옆 25% 제외 영역 계산 ---
        left_margin_25_percent = int(w_bg * 0.25)   # 왼쪽 25%
        right_margin_25_percent = int(w_bg * 0.25)  # 오른쪽 25%
        available_width = w_bg - left_margin_25_percent - right_margin_25_percent
        
        logger.info(f"Copy-Paste 영역: 양옆 {left_margin_25_percent}px(25%) 제외, 사용 가능 폭: {available_width}px")

        for _ in range(num_pastes_actual):
            try:
                selected_class = random.choices(classes_with_objects_and_weights, weights=weights_for_choice)[0]
            except IndexError:
                logger.warning("가중치 기반 클래스 선택 실패. 건너뜁니다.")
                continue
            
            if not self.class_objects[selected_class]: continue

            obj_data = random.choice(self.class_objects[selected_class])
            obj_img_to_paste = obj_data['image'] 
            obj_mask_to_paste = obj_data['mask'] 
            obj_points_relative = obj_data['points'].copy() 

            if obj_img_to_paste is None or obj_img_to_paste.size == 0 or \
            obj_mask_to_paste is None or obj_mask_to_paste.size == 0:
                logger.warning(f"선택된 객체 '{selected_class}'의 이미지 또는 마스크가 비어있습니다.")
                continue

            h_obj_orig, w_obj_orig = obj_img_to_paste.shape[:2]

            current_scale = random.uniform(0.6, 1.4)
            current_rotation = 0
            if random.random() < 0.4: current_rotation = random.uniform(-20, 20)
            
            transform_center = (w_obj_orig // 2, h_obj_orig // 2)
            M_transform = cv2.getRotationMatrix2D(transform_center, current_rotation, current_scale)
            
            cos_t = np.abs(M_transform[0, 0]); sin_t = np.abs(M_transform[0, 1])
            new_obj_w = int((h_obj_orig * sin_t) + (w_obj_orig * cos_t))
            new_obj_h = int((h_obj_orig * cos_t) + (w_obj_orig * sin_t))

            if new_obj_w == 0 or new_obj_h == 0: continue

            M_transform[0, 2] += (new_obj_w / 2) - transform_center[0]
            M_transform[1, 2] += (new_obj_h / 2) - transform_center[1]
            
            final_obj_img = cv2.warpAffine(obj_img_to_paste, M_transform, (new_obj_w, new_obj_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            final_obj_mask_binary = cv2.warpAffine(obj_mask_to_paste, M_transform, (new_obj_w, new_obj_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
            
            ones_homo = np.ones((obj_points_relative.shape[0], 1))
            points_homo = np.hstack([obj_points_relative, ones_homo])
            final_obj_points_relative_transformed = (M_transform @ points_homo.T).T 

            if new_obj_h >= h_bg or new_obj_w >= available_width: continue  # 수정: available_width 체크
            
            if random.random() < 0.3: 
                color_aug = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=18, val_shift_limit=12, p=1.0)
                ])
                final_obj_img = color_aug(image=final_obj_img)['image']

            # --- 🎯 핵심 수정: paste 위치 계산을 양옆 25% 제외 영역으로 제한 ---
            paste_margin = max(5, int(0.01 * min(h_bg, w_bg)))
            
            # X축: 왼쪽 25% 마진부터 시작, 오른쪽 25% 마진 전까지
            min_paste_x = left_margin_25_percent + paste_margin
            max_paste_x = w_bg - right_margin_25_percent - new_obj_w - paste_margin
            
            # Y축: 기존과 동일 (상하 제한 없음)
            min_paste_y = paste_margin
            max_paste_y = h_bg - new_obj_h - paste_margin
            
            # 유효한 paste 영역이 있는지 확인
            if max_paste_x <= min_paste_x or max_paste_y <= min_paste_y: 
                logger.debug(f"객체 크기({new_obj_w}x{new_obj_h})가 너무 커서 25% 제외 영역에 배치할 수 없습니다.")
                continue
            
            found_position = False
            for _ in range(30): 
                # 수정된 범위에서 paste 위치 선택
                current_paste_x = random.randint(min_paste_x, max_paste_x)
                current_paste_y = random.randint(min_paste_y, max_paste_y)
                current_bbox_abs = [current_paste_x, current_paste_y, current_paste_x + new_obj_w, current_paste_y + new_obj_h]
                
                # 디버그 로그 추가 (선택사항)
                logger.debug(f"Paste 시도: x={current_paste_x} (범위: {min_paste_x}-{max_paste_x}), "
                            f"y={current_paste_y} (범위: {min_paste_y}-{max_paste_y})")
                
                if any(self.calculate_iou(current_bbox_abs, occ_bbox) > 0.15 for occ_bbox in occupied_bboxes):
                    continue
                
                output_image, _ = self.blender.blend_object_onto_background(
                    output_image, final_obj_img, final_obj_mask_binary, 
                    None, 
                    current_paste_x, current_paste_y, new_obj_w, new_obj_h,
                    blend_mode=blend_mode
                )
                
                abs_points_for_label = (final_obj_points_relative_transformed + np.array([current_paste_x, current_paste_y])).astype(np.int32).tolist()
                pasted_shapes_info.append({
                    'label': selected_class,
                    'points': abs_points_for_label, 
                    'group_id': None, 'shape_type': 'polygon', 'flags': {}
                })
                occupied_bboxes.append(current_bbox_abs)
                successfully_pasted_count += 1
                found_position = True
                break

        logger.debug(f"{successfully_pasted_count}개의 객체(Advanced Blending, 양옆 25% 제외) 붙여넣기 완료 (시도: {num_pastes_actual}개).")
        return output_image, pasted_shapes_info


    def augment_dataset_pipeline(self, pipeline_type="copy_paste_first", 
                                 target_total_images=2400, 
                                 elastic_grid_prob=0.8,
                                 copy_paste_prob=0.9, 
                                 num_pastes_range=(1,3),
                                 blend_mode='advanced_alpha'):
        logger.info(f"'{pipeline_type}' (블렌드: {blend_mode}) 파이프라인으로 데이터셋 증강 시작: 목표 {target_total_images}장")

        self.analyze_dataset()
        if not self.original_class_counts:
            logger.error("원본 데이터셋 분석 실패. 증강 중단.")
            return

        class_weights = self.calculate_optimized_weights()
        if not class_weights:
            logger.error("클래스 가중치 계산 실패. 증강 중단.")
            return
        
        self.extract_objects_from_dataset()

        json_files_original = list(self.labels_dir.glob('*.json'))

        # --- 이 부분을 수정 ---
        # 원본 파일을 복사하지 않고, 0에서부터 시작하여 순수 증강 이미지만 생성합니다.
        # logger.info("원본 파일 복사 중...")
        # original_image_count = len(json_files_original)
        # for json_file_idx, json_file in enumerate(json_files_original):
        #     if json_file_idx % 100 == 0:
        #             logger.info(f"원본 파일 복사 진행: {json_file_idx}/{len(json_files_original)}")
        #     base_name = json_file.stem
        #     img_file, img_ext_found = self._find_image_file(base_name)
        #     if img_file:
        #         try:
        #             shutil.copy2(img_file, self.output_images_dir / img_file.name)
        #             shutil.copy2(json_file, self.output_labels_dir / json_file.name)
        #         except Exception as e:
        #             logger.error(f"원본 파일 복사 실패: {img_file} 또는 {json_file} - {e}")
        #     else:
        #         logger.warning(f"원본 이미지 파일을 찾지 못해 복사하지 못했습니다: {self.images_dir / base_name}")
        # current_total_images = original_image_count
        
        current_total_images = 0
        original_image_count = 0 # 원본은 복사하지 않으므로 0으로 설정
        # --- 수정 끝 ---
        
        generated_augmented_count = 0
        
        difficulty_levels = ['easy', 'medium', 'hard']
        difficulty_probs = [0.3, 0.5, 0.2]

        while current_total_images < target_total_images:
            random.shuffle(json_files_original)
            for json_file_orig in json_files_original:
                if current_total_images >= target_total_images: break

                base_name = json_file_orig.stem
                img_file_orig, img_ext = self._find_image_file(base_name)

                if not img_file_orig:
                    logger.warning(f"증강을 위한 원본 이미지 파일 없음: {self.images_dir / base_name}")
                    continue

                try:
                    image = cv2.imread(str(img_file_orig))
                    if image is None:
                        logger.warning(f"이미지 로드 실패: {img_file_orig}")
                        continue
                    
                    with open(json_file_orig, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                    
                    original_shapes = label_data.get('shapes', []) 
                    augmented_image = image.copy()
                    current_shapes = [s.copy() for s in original_shapes] 

                    difficulty = random.choices(difficulty_levels, difficulty_probs)[0]

                    # 🎯 핵심 변경: Copy-Paste를 먼저 수행하도록 순서 변경
                    if pipeline_type == "copy_paste_first": 
                        if random.random() < copy_paste_prob and any(self.class_objects.values()):
                            pasted_bg, new_pasted_shapes = self.intelligent_copy_paste_with_advanced_blending(
                                augmented_image.copy(), 
                                class_weights,
                                num_pastes_range=num_pastes_range,
                                difficulty_level=difficulty,
                                blend_mode=blend_mode
                            )
                            augmented_image = pasted_bg
                            current_shapes.extend(new_pasted_shapes) 
                            logger.debug(f"CPF: Copy-Paste 적용 후 shapes 개수: {len(current_shapes)}")
                        
                        augmented_image, current_shapes = self.apply_combined_augmentations(
                            augmented_image, current_shapes, transform_prob=elastic_grid_prob
                        )
                        logger.debug(f"CPF: Elastic/Grid 적용 후 shapes 개수: {len(current_shapes)}")

                    elif pipeline_type == "elastic_grid_first": 
                        augmented_image, current_shapes = self.apply_combined_augmentations(
                            augmented_image, current_shapes, transform_prob=elastic_grid_prob
                        )
                        logger.debug(f"EGF: Elastic/Grid 적용 후 shapes 개수: {len(current_shapes)}")

                        if random.random() < copy_paste_prob and any(self.class_objects.values()):
                            pasted_bg, new_pasted_shapes = self.intelligent_copy_paste_with_advanced_blending(
                                augmented_image.copy(), 
                                class_weights, 
                                num_pastes_range=num_pastes_range, 
                                difficulty_level=difficulty,
                                blend_mode=blend_mode
                            )
                            augmented_image = pasted_bg
                            current_shapes.extend(new_pasted_shapes) 
                            logger.debug(f"EGF: Copy-Paste 적용 후 shapes 개수: {len(current_shapes)}")
                            
                    else:
                        logger.error(f"알 수 없는 파이프라인 유형: {pipeline_type}")
                        continue
                    
                    current_shapes = [s for s in current_shapes if s.get('points') and len(s['points']) >= 3]

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    aug_base_name = f"{base_name}_{pipeline_type}_{blend_mode}_{timestamp}_{random.randint(1000,9999)}"
                    aug_img_path = self.output_images_dir / f"{aug_base_name}{img_ext}" 
                    aug_label_path = self.output_labels_dir / f"{aug_base_name}.json"

                    cv2.imwrite(str(aug_img_path), augmented_image)
                    
                    final_label_data = {
                        "version": label_data.get("version", "5.0.0"), "flags": label_data.get("flags", {}),
                        "shapes": current_shapes, 
                        "imagePath": aug_img_path.name, "imageData": None,
                        "imageHeight": augmented_image.shape[0], "imageWidth": augmented_image.shape[1]
                    }
                    with open(aug_label_path, 'w', encoding='utf-8') as f:
                        json.dump(final_label_data, f, indent=2, ensure_ascii=False)

                    generated_augmented_count += 1
                    current_total_images += 1
                    if generated_augmented_count % 20 == 0: 
                        logger.info(f"   - 생성된 증강 이미지 {generated_augmented_count}개 (총 {current_total_images}/{target_total_images}장)")

                except Exception as e:
                    logger.exception(f"증강 파이프라인 오류 ({pipeline_type}, {blend_mode}): {json_file_orig.name} - {e}")
        
        logger.info(f"'{pipeline_type}' ({blend_mode}) 파이프라인 최종 증강 완료: 원본 {original_image_count}장 + 증강 {generated_augmented_count}장 = 총 {current_total_images}장")
        self.visualize_augmentation_results()

    def visualize_augmentation_results(self):
        set_korean_font() 
        final_counts_from_output = Counter()
        output_label_files = list(self.output_labels_dir.glob('*.json'))
        if not output_label_files:
            logger.warning("출력 디렉토리에 라벨 파일이 없어 시각화를 건너뜁니다.")
            return
        logger.info(f"시각화를 위해 총 {len(output_label_files)}개의 출력 라벨 파일 분석 중...")
        for json_file in output_label_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f: data = json.load(f)
                if 'shapes' in data:
                    for shape in data['shapes']:
                        label = shape.get('label')
                        if label and label in self.class_to_idx: 
                            final_counts_from_output[label] += 1
            except Exception as e: logger.warning(f"출력 라벨 파일 분석 오류: {json_file.name} - {e}")
        logger.info(f"출력 파일 분석 기반 최종 클래스 분포: {dict(final_counts_from_output)}")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 14))
        fig.suptitle("데이터 증강 결과 분석", fontsize=20, fontweight='bold', y=0.98)
        classes = list(self.class_names)
        original_obj_counts = [self.original_class_counts.get(c, 0) for c in classes]
        final_obj_counts = [final_counts_from_output.get(c, 0) for c in classes]
        x_indices = np.arange(len(classes))
        bar_width = 0.35
        rects1 = ax1.bar(x_indices - bar_width/2, original_obj_counts, bar_width, label='원본 객체 수', color='deepskyblue', alpha=0.9)
        rects2 = ax1.bar(x_indices + bar_width/2, final_obj_counts, bar_width, label='증강 후 객체 수', color='salmon', alpha=0.9)
        ax1.set_xlabel('클래스', fontsize=13); ax1.set_ylabel('객체 수', fontsize=13)
        ax1.set_title('클래스별 객체 수 비교', fontsize=15); ax1.set_xticks(x_indices)
        ax1.set_xticklabels(classes, rotation=45, ha="right", fontsize=10); ax1.legend(fontsize=11)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        for rect in rects1 + rects2:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., h, f'{int(h)}', ha='center', va='bottom', fontsize=8)
        increase_rates = [((f - o) / o * 100) if o > 0 else (float('inf') if f > 0 else 0) for o, f in zip(original_obj_counts, final_obj_counts)]
        colors_bar = ['limegreen' if r >= 100 else 'gold' if r >= 0 else 'tomato' for r in increase_rates]
        bars = ax2.bar(classes, increase_rates, color=colors_bar)
        ax2.set_xlabel('클래스', fontsize=13); ax2.set_ylabel('객체 수 증가율 (%)', fontsize=13)
        ax2.set_title('클래스별 객체 수 증가율', fontsize=15); ax2.grid(axis='y', linestyle=':', alpha=0.6)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        for bar_idx, bar_item in enumerate(bars):
            yval = bar_item.get_height()
            ax2.text(bar_item.get_x() + bar_item.get_width()/2., yval, f'{yval:.0f}%' if yval != float('inf') else 'Inf', 
                     ha='center', va='bottom' if yval >=0 else 'top', fontsize=8)
        if sum(final_obj_counts) > 0:
            valid_labels = [classes[i] for i, v in enumerate(final_obj_counts) if v > 0]
            valid_values = [v for v in final_obj_counts if v > 0]
            ax3.pie(valid_values, labels=valid_labels, autopct='%1.1f%%', startangle=120,
                    wedgeprops={'edgecolor': 'silver', 'linewidth': 0.7}, textprops={'fontsize': 9})
            ax3.set_title('최종 클래스 분포 (증강 후 객체 기준)', fontsize=15); ax3.axis('equal')
        else:
            ax3.text(0.5, 0.5, "증강된 객체 없음", ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('최종 클래스 분포 (증강 후 객체 기준)', fontsize=15)
        def gini_coefficient_calc(values):
            vals = sorted(filter(lambda x: x > 0, values))
            if not vals or len(vals) <= 1: return 0.0
            n = len(vals); idx = np.arange(1, n + 1)
            return (np.sum((2 * idx - n - 1) * np.array(vals))) / (n * sum(vals)) if sum(vals) > 0 else 0.0
        gini_orig = gini_coefficient_calc(original_obj_counts)
        gini_final = gini_coefficient_calc(final_obj_counts)
        ax4.text(0.5, 0.85, '클래스 균형도 (Gini 계수)', ha='center', va='center', fontsize=15, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.5, 0.65, f'원본 Gini: {gini_orig:.3f}', ha='center', va='center', fontsize=13, transform=ax4.transAxes)
        ax4.text(0.5, 0.55, f'증강 후 Gini: {gini_final:.3f}', ha='center', va='center', fontsize=13, transform=ax4.transAxes)
        improvement_text_val, text_color_val = "Gini 개선율: N/A", 'dimgray'
        if gini_orig > 1e-6 : 
            improvement_val = ((gini_orig - gini_final) / gini_orig * 100)
            improvement_text_val = f'Gini 개선율: {improvement_val:.1f}%'
            text_color_val = 'forestgreen' if improvement_val > 0 else ('tomato' if improvement_val < 0 else 'darkorange')
        elif gini_orig <= 1e-6 and gini_final > 1e-6 : improvement_text_val, text_color_val = "균형 악화됨", 'tomato'
        elif gini_orig <= 1e-6 and gini_final <= 1e-6 : improvement_text_val, text_color_val = "완벽 균형 유지", 'forestgreen'
        ax4.text(0.5, 0.35, improvement_text_val, ha='center', va='center', fontsize=16, color=text_color_val, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.5, 0.15, "(Gini 계수는 0에 가까울수록 균형)", ha='center', va='center', fontsize=10, style='italic', color='gray', transform=ax4.transAxes)
        ax4.axis('off'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_fig = self.output_dir / f'augmentation_summary_{timestamp_str}.png'
        try:
            plt.savefig(save_path_fig, dpi=300, bbox_inches='tight')
            logger.info(f"시각화 요약 저장 완료: {save_path_fig}")
        except Exception as e: logger.error(f"시각화 파일 저장 실패: {e}")
        plt.close(fig)
        logger.info("\n=== 최종 통계 요약 (객체 수 기준) ===")
        logger.info(f"원본 총 객체 수: {sum(original_obj_counts)}개")
        logger.info(f"증강 후 총 객체 수 (출력 파일 분석): {sum(final_obj_counts)}개")
        for c_name_log in classes:
            o_cnt_log, f_cnt_log = self.original_class_counts.get(c_name_log,0), final_counts_from_output.get(c_name_log,0)
            inc_str_log = f"({(f_cnt_log-o_cnt_log)/o_cnt_log*100:.0f}%)" if o_cnt_log > 0 else "(원본 0)"
            if f_cnt_log > 0 and o_cnt_log == 0: inc_str_log = "(신규)"
            logger.info(f"   - {c_name_log}: {o_cnt_log} → {f_cnt_log} {inc_str_log}")

# 사용 예시
if __name__ == '__main__':
    images_dir_path = r'C:/Users/jenni/Desktop/aug again data/train/outputac25/images'
    labels_dir_path = r"C:/Users/jenni/Desktop/aug again data/train/outputac25/labels"
    # 🎯 핵심 변경: 출력 폴더명을 copy_paste_first로 변경
    output_dir_path_base = r"C:/Users/jenni/Desktop/aug again data/yolov8_training_results/outputac25yolo"
    
    class_names_config = ['ac', 'lc', 'pc', 'tc', 'ph']
    # --- 1. ✨ 목표 이미지 수 수정 (과제 요구사항 반영) ---

    # 원본이 1021장이므로, 공정한 비교를 위해 목표 증강 이미지 수도 1021장으로 설정합니다.

    target_total_images_for_experiment = 1021

    # --- 2. 핵심 파라미터 수정 ---

    # 한 번 증강할 때 더 많은 소수 클래스 객체를 붙여넣도록 범위를 늘립니다.

    # 이는 소수 클래스 객체 수를 빠르게 늘리는 데 가장 효과적인 방법입니다.

    num_pastes_range_config = (2, 6)

    # Copy-Paste가 불균형 해소의 핵심이므로, 거의 모든 증강 이미지에 적용되도록 확률을 높게 설정합니다.

    copy_paste_prob_config = 0.9
    # --- 실행할 파이프라인 및 블렌드 모드 설정 ---

    # 🎯 핵심 변경: Copy-Paste를 먼저 수행하도록 파이프라인 타입 변경

    chosen_pipeline_type = "copy_paste_first"  # 원본에서 elastic_grid_first였지만 copy_paste_first로 변경
    chosen_blend_mode = "simple_alpha"
    logger.info(f"\n\n{'='*20} 클래스 불균형 해소 증강 시작 (Copy-Paste First) {'='*20}")
    logger.info(f"목표 이미지 수: {target_total_images_for_experiment}")
    logger.info(f"붙여넣기 객체 수 범위: {num_pastes_range_config}")
    logger.info(f"붙여넣기 확률: {copy_paste_prob_config}")
    logger.info(f"파이프라인 유형: {chosen_pipeline_type}")
    logger.info(f"블렌드 모드: {chosen_blend_mode}")
    output_dir_single_exp = Path(output_dir_path_base)
    logger.info(f"출력 폴더: {output_dir_single_exp}")
    # 만약 이전에 생성된 폴더가 있다면 삭제하여 깨끗한 상태에서 시작

    if output_dir_single_exp.exists():
        logger.warning(f"기존 출력 폴더 '{output_dir_single_exp}'를 삭제합니다.")
        shutil.rmtree(output_dir_single_exp)
        
    pipeline_single = OptimizedYOLOAugmentation(
        images_dir=images_dir_path, labels_dir=labels_dir_path,
        output_dir=str(output_dir_single_exp), class_names=class_names_config
    )

    pipeline_single.augment_dataset_pipeline(
        pipeline_type=chosen_pipeline_type,
        target_total_images=target_total_images_for_experiment, # 수정된 파라미터 적용
        elastic_grid_prob=0.7,
        copy_paste_prob=copy_paste_prob_config,      # 수정된 파라미터 적용
        num_pastes_range=num_pastes_range_config, # 수정된 파라미터 적용
        blend_mode=chosen_blend_mode
    )

    logger.info(f"\n\n증강 완료. 결과를 '{output_dir_single_exp}' 폴더와 생성된 'augmentation_summary.png' 파일에서 확인하세요.")