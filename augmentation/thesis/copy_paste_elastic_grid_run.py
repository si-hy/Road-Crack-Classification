import os
import yaml
import json
import cv2
import numpy as np
import pandas as pd
import random
import shutil
import logging
import torch
from datetime import datetime
from pathlib import Path
from collections import Counter

# Third-party libraries
try:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    print("필수 라이브러리가 설치되지 않았습니다. 'pip install ultralytics matplotlib pandas'를 실행해주세요.")
    exit()


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_status():
    """GPU 상태를 확인하고 로깅합니다."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
        
        logger.info(f"✅ CUDA 사용 가능: {device_count}개 디바이스")
        logger.info(f"📱 현재 디바이스: {current_device} ({device_name})")
        logger.info(f"💾 GPU 메모리: 할당됨 {memory_allocated:.2f}GB, 예약됨 {memory_cached:.2f}GB")
        return True
    else:
        logger.warning("❌ CUDA 사용 불가능 - CPU 모드로 실행됩니다")
        return False

def clear_gpu_cache():
    """GPU 캐시를 정리합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("🧹 GPU 캐시 정리 완료")

class YOLOv8DatasetTrainer:
    """
    증강된 데이터셋을 사용하여 YOLOv8 모델을 학습, 평가, 시각화하는 전체 파이프라인을 관리합니다.
    """
    def __init__(self, train_data_dir, val_data_dir, class_names, output_dir):
        """
        클래스 초기화
        
        Args:
            train_data_dir (str): 증강된 학습 데이터 디렉토리 경로
            val_data_dir (str): 검증 데이터 디렉토리 경로  
            class_names (list): 클래스 이름 리스트 (예: ['ac', 'lc'])
            output_dir (str): 학습 결과 저장 디렉토리
        """
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU 상태 확인 및 디바이스 설정
        self.gpu_available = check_gpu_status()
        self.device = 'cuda' if self.gpu_available else 'cpu'
        
        # YOLOv8 형식 데이터셋 디렉토리 구조 정의
        self.yolo_dataset_dir = self.output_dir / "yolo_dataset"
        self.yolo_train_images = self.yolo_dataset_dir / "train" / "images"
        self.yolo_train_labels = self.yolo_dataset_dir / "train" / "labels"
        self.yolo_val_images = self.yolo_dataset_dir / "val" / "images"
        self.yolo_val_labels = self.yolo_dataset_dir / "val" / "labels"
        
        # 생성될 속성 미리 정의
        self.yaml_path = None
        self.training_results_dir = None
        self.best_model_path = None
        self.last_model_path = None

        # 🎨 개선된 색상 팔레트 (BGR 형식)
        self.colors = [
            (220, 20, 60),   # ac - Crimson Red
            (60, 179, 113),  # lc - Medium Sea Green
            (30, 144, 255),  # pc - Dodger Blue
            (255, 215, 0),   # tc - Gold
            (148, 0, 211),   # ph - Dark Violet
        ]

        # 디렉토리 생성
        for dir_path in [self.yolo_train_images, self.yolo_train_labels, self.yolo_val_images, self.yolo_val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def analyze_dataset_distribution(self):
        """데이터셋의 클래스 분포를 분석하고 로깅합니다."""
        logger.info("데이터셋 클래스 분포 분석 중...")
        
        train_class_counts = Counter()
        val_class_counts = Counter()
        
        # 학습 데이터 분석
        train_json_files = list((self.train_data_dir / "labels").glob("*.json"))
        for json_file in train_json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data.get('shapes', []):
                    if 'label' in shape and shape['label'] in self.class_names:
                        train_class_counts[shape['label']] += 1
            except Exception as e:
                logger.error(f"학습 라벨 파일 분석 오류: {json_file} - {e}")
        
        # 검증 데이터 분석
        val_json_files = list((self.val_data_dir / "labels").glob("*.json"))
        for json_file in val_json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data.get('shapes', []):
                    if 'label' in shape and shape['label'] in self.class_names:
                        val_class_counts[shape['label']] += 1
            except Exception as e:
                logger.error(f"검증 라벨 파일 분석 오류: {json_file} - {e}")
        
        logger.info(f"학습 데이터 클래스 분포: {dict(train_class_counts)}")
        logger.info(f"검증 데이터 클래스 분포: {dict(val_class_counts)}")
        
        return train_class_counts, val_class_counts

    def convert_labelme_to_yolo(self, json_file_path, image_width, image_height):
        """LabelMe JSON 형식을 YOLO 세그멘테이션 형식으로 변환합니다."""
        yolo_labels = []
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for shape in data.get('shapes', []):
                if shape.get('shape_type') != 'polygon':
                    continue
                
                label = shape.get('label', '')
                if label not in self.class_names:
                    continue
                
                class_id = self.class_names.index(label)
                points = np.array(shape.get('points', []), dtype=np.float32)
                
                if len(points) < 3:
                    continue
                
                # 폴리곤 좌표를 0~1 범위로 정규화
                normalized_points = points.copy()
                normalized_points[:, 0] /= image_width
                normalized_points[:, 1] /= image_height
                
                # YOLO 세그멘테이션 형식: class_id x1 y1 x2 y2 ... xn yn
                yolo_line = f"{class_id}"
                for point in normalized_points.flatten():
                    yolo_line += f" {point:.6f}"
                yolo_labels.append(yolo_line)
        
        except Exception as e:
            logger.error(f"라벨 변환 오류: {json_file_path} - {e}")
        
        return yolo_labels

    def prepare_yolo_dataset(self):
        """전체 데이터셋을 YOLOv8 형식으로 준비합니다."""
        logger.info("YOLOv8 형식 데이터셋 준비 중...")
        self._convert_dataset_split(self.train_data_dir, self.yolo_train_images, self.yolo_train_labels, "학습")
        self._convert_dataset_split(self.val_data_dir, self.yolo_val_images, self.yolo_val_labels, "검증")
        self._create_yaml_config()
        logger.info("YOLOv8 형식 데이터셋 준비 완료")

    def _convert_dataset_split(self, source_dir, target_images_dir, target_labels_dir, split_name):
        """데이터셋의 한 분할(train/val)을 변환하는 헬퍼 함수입니다."""
        images_source = source_dir / "images"
        labels_source = source_dir / "labels"
        
        json_files = list(labels_source.glob("*.json"))
        logger.info(f"{split_name} 데이터 변환 시작: {len(json_files)}개 파일")
        
        converted_count = 0
        error_count = 0
        
        for idx, json_file in enumerate(json_files):
            if idx > 0 and idx % 100 == 0:
                logger.info(f"{split_name} 데이터 변환 진행: {idx}/{len(json_files)}")
            
            try:
                base_name = json_file.stem
                image_file = self._find_image_file(images_source, base_name)
                
                if not image_file:
                    logger.warning(f"이미지 파일을 찾을 수 없음: {base_name}")
                    error_count += 1
                    continue
                
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.warning(f"이미지 로드 실패: {image_file}")
                    error_count += 1
                    continue
                
                height, width = image.shape[:2]
                yolo_labels = self.convert_labelme_to_yolo(json_file, width, height)
                
                shutil.copy2(image_file, target_images_dir / image_file.name)
                
                target_label_path = target_labels_dir / f"{base_name}.txt"
                with open(target_label_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_labels))
                
                converted_count += 1
            except Exception as e:
                logger.error(f"{split_name} 데이터 변환 중 오류 발생: {json_file} - {e}")
                error_count += 1
        
        logger.info(f"{split_name} 데이터 변환 완료: 성공 {converted_count}개, 실패 {error_count}개")

    def _find_image_file(self, images_dir, base_name):
        """베이스 이름으로 이미지 파일을 찾습니다."""
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']
        for ext in possible_extensions:
            potential_file = images_dir / f"{base_name}{ext}"
            if potential_file.exists():
                return potential_file
        return None

    def _create_yaml_config(self):
        """YOLOv8 학습을 위한 YAML 설정 파일을 생성합니다."""
        yaml_content = {
            'path': str(self.yolo_dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        self.yaml_path = self.yolo_dataset_dir / "dataset.yaml"
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"YAML 설정 파일 생성 완료: {self.yaml_path}")

    def train_model(self, model_name, epochs, imgsz, device, patience, disable_yolo_augmentation):
        """YOLOv8 모델을 학습합니다."""
        logger.info("YOLOv8 모델 학습 시작...")
        logger.info(f"학습 설정: 모델={model_name}, 에포크={epochs}, 이미지크기={imgsz}, YOLO기본증강비활성화={disable_yolo_augmentation}")
        
        # 학습 전 GPU 상태 재확인 및 캐시 정리
        clear_gpu_cache()
        if self.gpu_available:
            check_gpu_status()
        
        # 환경 변수 설정으로 GPU 사용 강제
        if self.gpu_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            torch.cuda.set_device(0)
        
        model = YOLO(model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"yolov8_training_{timestamp}"
        
        try:
            # 명시적으로 디바이스 설정
            actual_device = self.device if device == 'auto' else device
            logger.info(f"🎯 사용할 디바이스: {actual_device}")
            
            train_params = {
                'data': str(self.yaml_path), 
                'epochs': epochs, 
                'imgsz': imgsz,
                'device': actual_device,  # 명시적 디바이스 설정
                'project': str(self.output_dir), 
                'name': project_name,
                'patience': patience, 
                'save': True, 
                'plots': True, 
                'val': True, 
                'verbose': True,
                'workers': 4,  # 데이터 로더 워커 수 제한
                'batch': -1,   # 자동 배치 크기 조정
            }
            
            if disable_yolo_augmentation:
                logger.info("⚠️ YOLO 기본 증강을 비활성화합니다 (순수 증강 효과 측정).")
                train_params.update({
                    'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'shear': 0.0,
                    'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.0,
                    'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
                    'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
                })
            
            # 학습 실행 전 메모리 정리
            if self.gpu_available:
                clear_gpu_cache()
            
            logger.info("🚀 모델 학습을 시작합니다...")
            model.train(**train_params)
            
            logger.info("✅ 모델 학습 완료!")
            self.training_results_dir = self.output_dir / project_name
            self.best_model_path = self.training_results_dir / "weights" / "best.pt"
            self.last_model_path = self.training_results_dir / "weights" / "last.pt"
            logger.info(f"📁 학습 결과 저장 위치: {self.training_results_dir}")
            logger.info(f"🏆 최고 성능 모델: {self.best_model_path}")
            
            # 학습 후 메모리 정리
            if self.gpu_available:
                clear_gpu_cache()
            
        except Exception as e:
            logger.error(f"❌ 모델 학습 중 오류 발생: {e}")
            logger.error(f"오류 타입: {type(e).__name__}")
            
            # GPU 관련 오류인 경우 CPU로 재시도 제안
            if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                logger.warning("🔄 GPU 오류가 발생했습니다. CPU 모드로 재시도를 권장합니다.")
                logger.warning("main() 함수에서 device='cpu'로 설정하여 다시 실행해보세요.")
            
            raise

    def evaluate_model(self):
        """학습된 최고 성능 모델을 평가합니다."""
        if not self.best_model_path or not self.best_model_path.exists():
            logger.error("평가할 모델이 없습니다. 먼저 학습을 완료하세요.")
            return
        
        logger.info("최고 성능 모델 평가 시작...")
        
        # 평가 전 GPU 캐시 정리
        if self.gpu_available:
            clear_gpu_cache()
        
        try:
            model = YOLO(str(self.best_model_path))
            results = model.val(
                data=str(self.yaml_path), 
                split='val', 
                save_json=True, 
                plots=True,
                device=self.device
            )
            
            logger.info("✅ 모델 평가 완료!")
            logger.info(f"📊 mAP50-95 (Box): {results.box.map:.4f}, mAP50 (Box): {results.box.map50:.4f}")
            logger.info(f"📊 mAP50-95 (Seg): {results.seg.map:.4f}, mAP50 (Seg): {results.seg.map50:.4f}")
            
        except Exception as e:
            logger.error(f"❌ 모델 평가 중 오류 발생: {e}")
            raise

    def _draw_dashed_polyline(self, img, points, color, thickness=1, dash_len=10):
        """점선 폴리곤을 그립니다."""
        for i in range(len(points)):
            start_point = tuple(points[i])
            end_point = tuple(points[(i + 1) % len(points)])
            
            line_len = np.linalg.norm(np.array(start_point) - np.array(end_point))
            num_dashes = int(line_len / (2 * dash_len))
            
            for j in range(num_dashes):
                start = (int(start_point[0] + (end_point[0] - start_point[0]) * (2 * j) / (2 * num_dashes)),
                         int(start_point[1] + (end_point[1] - start_point[1]) * (2 * j) / (2 * num_dashes)))
                end = (int(start_point[0] + (end_point[0] - start_point[0]) * (2 * j + 1) / (2 * num_dashes)),
                       int(start_point[1] + (end_point[1] - start_point[1]) * (2 * j + 1) / (2 * num_dashes)))
                cv2.line(img, start, end, color, thickness)

    def _draw_label(self, img, text, pos, bg_color, text_color=(255, 255, 255)):
        """가독성 좋은 배경이 있는 라벨을 그립니다."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        top_left = (pos[0], pos[1] - text_h - baseline)
        bottom_right = (pos[0] + text_w, pos[1])
        
        # 반투명 배경을 위한 오버레이 생성
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img) # 투명도 조절
        
        cv2.putText(img, text, (pos[0], pos[1] - baseline // 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def visualize_predictions(self, num_samples=10, conf_threshold=0.4):
        """개선된 스타일로 예측 결과를 시각화합니다."""
        if not self.best_model_path or not self.best_model_path.exists():
            logger.error(f"모델 파일이 존재하지 않습니다: {self.best_model_path}"); return

        logger.info(f"🎨 검증 데이터셋 예측 시각화 시작 (샘플 수: {num_samples})")
        
        if self.device == 'cuda': clear_gpu_cache()
        
        try:
            model = YOLO(str(self.best_model_path))
            val_images = list(self.yolo_val_images.glob("*.*"))

            if not val_images: logger.warning("검증 이미지가 없습니다."); return

            random.shuffle(val_images)
            selected_images = val_images[:min(num_samples, len(val_images))]
            viz_dir = self.output_dir / "prediction_visualizations_improved"
            viz_dir.mkdir(exist_ok=True)

            for idx, img_path in enumerate(selected_images):
                logger.info(f"🖼️  시각화 진행: {idx+1}/{len(selected_images)} - {img_path.name}")
                
                image = cv2.imread(str(img_path))
                if image is None: logger.warning(f"이미지 로드 실패: {img_path}"); continue
                
                h, w = image.shape[:2]
                
                results = model.predict(source=str(img_path), conf=conf_threshold, save=False, verbose=False, device=self.device)
                result = results[0]
                
                # --- 예측 결과 그리기 (반투명 마스크 + 실선 테두리) ---
                if hasattr(result, 'masks') and result.masks is not None:
                    overlay = image.copy()
                    alpha = 0.4 # 마스크 투명도

                    for i, mask_data in enumerate(result.masks.data):
                        class_id = int(result.boxes.cls[i])
                        color = self.colors[class_id % len(self.colors)]
                        
                        mask = cv2.resize(mask_data.cpu().numpy(), (w, h))
                        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # 반투명 마스크 채우기
                            cv2.fillPoly(overlay, contours, color)

                    # 원본 이미지와 반투명 마스크 오버레이 합성
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                    
                    # 합성된 이미지 위에 테두리 및 라벨 다시 그리기
                    for i, mask_data in enumerate(result.masks.data):
                        class_id = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])
                        color = self.colors[class_id % len(self.colors)]
                        
                        mask = cv2.resize(mask_data.cpu().numpy(), (w, h))
                        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # 실선 테두리
                            cv2.drawContours(image, contours, -1, color, 2)
                            
                            # 가독성 좋은 라벨
                            x, y, _, _ = cv2.boundingRect(contours[0])
                            label_text = f"P: {self.class_names[class_id]} {conf:.2f}"
                            self._draw_label(image, label_text, (x, y - 5), (0,0,0))

                # --- Ground Truth 그리기 (흰색 점선) ---
                label_path = self.yolo_val_labels / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 7:
                                class_id = int(parts[0])
                                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                                coords[:, 0] *= w; coords[:, 1] *= h
                                pts = coords.astype(np.int32)
                                
                                # 점선 테두리
                                self._draw_dashed_polyline(image, pts, (255, 255, 255), 2, dash_len=5)


                output_path = viz_dir / f"improved_pred_{img_path.name}"
                cv2.imwrite(str(output_path), image)

            logger.info(f"🎉 개선된 시각화 완료. 결과 저장 위치: {viz_dir}")
        except Exception as e:
            logger.error(f"❌ 시각화 중 오류 발생: {e}", exc_info=True)
            raise

    def create_evaluation_plots(self):
        """학습 결과(results.csv)를 바탕으로 평가 플롯을 생성합니다."""
        if not self.training_results_dir or not self.training_results_dir.exists():
            logger.warning("학습 결과 디렉토리가 없어 평가 플롯을 생성할 수 없습니다.")
            return

        results_csv = self.training_results_dir / "results.csv"
        if not results_csv.exists():
            logger.warning(f"결과 CSV 파일이 없습니다: {results_csv}")
            return
            
        logger.info("평가 플롯 생성 중...")
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('YOLOv8 Training & Validation Analysis', fontsize=16, fontweight='bold')

            # Losses, mAP50, mAP50-95
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
            
            axes[0, 1].plot(df['epoch'], df['train/seg_loss'], label='Train Seg Loss')
            axes[0, 1].plot(df['epoch'], df['val/seg_loss'], label='Val Seg Loss')
            axes[0, 1].set_title('Segmentation Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)
            
            axes[0, 2].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
            axes[0, 2].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
            axes[0, 2].set_title('Classification Loss'); axes[0, 2].legend(); axes[0, 2].grid(True)
            
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (Box)')
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(M)'], label='mAP50 (Mask)')
            axes[1, 0].set_title('mAP@0.50'); axes[1, 0].legend(); axes[1, 0].grid(True)

            axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95 (Box)')
            axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(M)'], label='mAP50-95 (Mask)')
            axes[1, 1].set_title('mAP@0.50:0.95'); axes[1, 1].legend(); axes[1, 1].grid(True)

            axes[1, 2].axis('off')
            final_epoch = df.iloc[-1]
            summary_text = "Final Performance:\n\n"
            summary_text += f"mAP50 (Box): {final_epoch.get('metrics/mAP50(B)', 0):.4f}\n"
            summary_text += f"mAP50-95 (Box): {final_epoch.get('metrics/mAP50-95(B)', 0):.4f}\n"
            summary_text += f"mAP50 (Mask): {final_epoch.get('metrics/mAP50(M)', 0):.4f}\n"
            summary_text += f"mAP50-95 (Mask): {final_epoch.get('metrics/mAP50-95(M)', 0):.4f}"
            axes[1, 2].text(0.0, 0.5, summary_text, fontsize=12, va='center')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = self.output_dir / "training_evaluation_plots.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            logger.info(f"✅ 평가 플롯 저장 완료: {plot_path}")
        except Exception as e:
            logger.error(f"❌ 평가 플롯 생성 중 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    # ------------------- 사용자 설정 -------------------
    # 🎯 실험의 일관성을 위해 이 설정들을 유지하세요.
    EPOCHS = 50
    IMGSZ = 640
    MODEL_NAME = "yolov8n-seg.pt"
    # 🎯 증강 방법 효과를 정확히 측정하려면 True로 설정
    DISABLE_YOLO_AUG = True
    
    # 🔧 디바이스 설정 ('auto', 'cuda', 'cpu' 중 선택)
    # GPU 오류 시 'cpu'로 변경하여 재시도
    DEVICE = 'auto'  # GPU 오류 발생시 'cpu'로 변경

    # 🎯 데이터 및 결과 경로 설정 (사용자 환경에 맞게 수정)
    BASE_DATA_PATH = Path(r"C:/Users/jasmi/Downloads/Final data and augmented datasets")
    TRAIN_DATA_DIR = BASE_DATA_PATH / "train/outputac"
    VAL_DATA_DIR = BASE_DATA_PATH / "val"
    OUTPUT_DIR = BASE_DATA_PATH / "yolov8_training_results/elastic_grid_copy_paste_ac25"
    
    CLASS_NAMES = ['ac', 'lc', 'pc', 'tc', 'ph']
    # ----------------------------------------------------

    logger.info("=" * 60)
    logger.info(f"🚀 YOLOv8 증강 데이터셋 학습 파이프라인 시작")
    logger.info(f"📁 출력 디렉토리: {OUTPUT_DIR}")
    logger.info("=" * 60)
    
    # 초기 GPU 상태 확인
    check_gpu_status()
    
    if OUTPUT_DIR.exists():
        logger.warning(f"⚠️ 기존 출력 폴더 '{OUTPUT_DIR}'가 존재합니다. 내용을 덮어쓸 수 있습니다.")

    try:
        # 1. 트레이너 초기화
        logger.info("🔧 트레이너 초기화 중...")
        trainer = YOLOv8DatasetTrainer(
            train_data_dir=str(TRAIN_DATA_DIR), 
            val_data_dir=str(VAL_DATA_DIR),
            class_names=CLASS_NAMES, 
            output_dir=str(OUTPUT_DIR)
        )
        
        # 2. 데이터셋 분석 및 YOLO 형식으로 변환
        logger.info("📊 데이터셋 분석 및 변환 중...")
        trainer.analyze_dataset_distribution()
        trainer.prepare_yolo_dataset()
        
        # 3. 모델 학습
        logger.info("🎓 모델 학습 시작...")
        trainer.train_model(
            model_name=MODEL_NAME, 
            epochs=EPOCHS, 
            imgsz=IMGSZ,
            device=DEVICE, 
            patience=50, 
            disable_yolo_augmentation=DISABLE_YOLO_AUG
        )
        
        # 4. 모델 평가 및 결과 시각화
        logger.info("📈 모델 평가 및 시각화 중...")
        trainer.evaluate_model()
        trainer.visualize_predictions(num_samples=15)
        trainer.create_evaluation_plots()
        
        logger.info("=" * 60)
        logger.info("🎉 모든 과정 완료!")
        logger.info(f"📁 결과는 '{OUTPUT_DIR}' 폴더에서 확인하세요.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"💥 전체 파이프라인 실행 중 심각한 오류 발생: {e}")
        logger.error(f"🔍 오류 타입: {type(e).__name__}")
        
        # GPU 관련 오류 해결 가이드
        if "cuda" in str(e).lower() or "gpu" in str(e).lower() or "device" in str(e).lower():
            logger.error("=" * 60)
            logger.error("🚨 GPU 관련 오류가 발생했습니다!")
            logger.error("💡 해결 방법:")
            logger.error("1. 스크립트 상단의 DEVICE = 'auto'를 DEVICE = 'cpu'로 변경")
            logger.error("2. 아나콘다 프롬프트를 재시작 후 다시 실행")
            logger.error("3. 또는 컴퓨터를 재부팅 후 다시 실행")
            logger.error("=" * 60)
        
        raise

if __name__ == "__main__":
    main()