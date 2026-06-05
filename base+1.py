# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os
import sys

class OptimizedMeridianPoseTrainer:
    def __init__(self):
        self.model = None

    def setup_training_environment(self):
        if sys.platform.startswith('win'):
            sys.stdout.reconfigure(encoding='utf-8')
        return True

    def validate_baseline_model(self, model_path):
        print(f"🔍 验证基础模型性能...")
        try:
            model = YOLO(model_path)
            results = model.val(
                data='keypoints.yaml',
                imgsz=640,
                batch=4,
                device='cpu',
                workers=2,
                verbose=False
            )
            print(f"✅ 基础模型性能:")
            print(f"  关键点检测 - mAP50: {results.pose.map50:.3f}, mAP50-95: {results.pose.map:.3f}")
            print(f"  目标检测 - mAP50: {results.box.map50:.3f}, mAP50-95: {results.box.map:.3f}")
            return model, results
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            return None, None

    def meridian_training(self, baseline_model_path):
        """经络感知训练（论文阶段1，固定学习率0.001）"""
        print("🚀 开始经络感知训练...")
        model = YOLO(baseline_model_path)
        results = model.train(
            data='keypoints.yaml',
            epochs=100,
            imgsz=640,
            batch=8,
            device='cpu',
            workers=4,
            patience=30,
            save=True,
            project='runs/train',
            name='meridian_stage1',
            exist_ok=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=5.0,
            warmup_momentum=0.8,
            box=5.0,
            cls=0.5,
            dfl=1.0,
            pose=12.0,
            kobj=2.0,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=10.0,
            translate=0.1,
            scale=0.2,
            shear=0.0,
            perspective=0.0001,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.0,
            overlap_mask=False,
            mask_ratio=1,
            close_mosaic=0
        )
        print("✅ 经络感知训练完成!")
        return model, results

    def _extract_final_metrics(self, results):
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            return {
                'box_loss': metrics.get('train/box_loss', 0),
                'cls_loss': metrics.get('train/cls_loss', 0),
                'pose_loss': metrics.get('train/pose_loss', 0),
                'val_box_loss': metrics.get('val/box_loss', 0),
                'val_pose_loss': metrics.get('val/pose_loss', 0),
                'precision': metrics.get('metrics/precision(B)', 0),
                'recall': metrics.get('metrics/recall(B)', 0),
                'map50': metrics.get('metrics/mAP50(B)', 0),
                'map50_95': metrics.get('metrics/mAP50-95(B)', 0)
            }
        else:
            return {
                'box_loss': getattr(results, 'box_loss', 0),
                'pose_loss': getattr(results, 'pose_loss', 0),
                'precision': getattr(results, 'precision', 0),
                'recall': getattr(results, 'recall', 0)
            }

    def generate_report(self, baseline_results, trained_results):
        print("\n" + "=" * 70)
        print("📈 经络感知训练效果报告")
        print("=" * 70)
        if hasattr(baseline_results, 'pose') and hasattr(trained_results, 'pose'):
            base_map = baseline_results.pose.map
            trained_map = trained_results.pose.map
            improvement = trained_map - base_map
            print(f"基础模型 mAP50-95: {base_map:.4f}")
            print(f"训练后 mAP50-95: {trained_map:.4f}")
            print(f"性能提升: {improvement:+.4f}")
        else:
            print("⚠️ 无法获取完整的性能指标")

def train_optimized_model():
    trainer = OptimizedMeridianPoseTrainer()
    if not trainer.setup_training_environment():
        return
    print("=" * 70)
    print("基于平衡优化的精细化腧穴定位模型训练（阶段1）")
    print("=" * 70)
    try:
        baseline_model_path = 'runs/train/keypoints_model/weights/best.pt'
        if not os.path.exists(baseline_model_path):
            print(f"❌ 基础模型不存在: {baseline_model_path}")
            return
        baseline_model, baseline_results = trainer.validate_baseline_model(baseline_model_path)
        if baseline_model is None:
            print("❌ 基础模型验证失败")
            return
        print(f"✅ 基础模型验证完成，开始经络感知训练...")
        trained_model, trained_results = trainer.meridian_training(baseline_model_path)
        trainer.generate_report(baseline_results, trained_results)
        return trained_model, trained_results
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_optimized_model()