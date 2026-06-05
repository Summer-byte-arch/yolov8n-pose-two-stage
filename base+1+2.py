import os
import sys
from ultralytics import YOLO
from dynamic_pose_weight import DynamicPoseTrainer

def setup_environment():
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    return True

def validate_model(model_path, desc="model"):
    print(f"Validating {desc}: {model_path}")
    try:
        model = YOLO(model_path)
        results = model.val(data='keypoints.yaml', imgsz=640, batch=4, device='cpu', workers=2, verbose=False)
        print(f"   mAP50: {results.pose.map50:.4f}, mAP50-95: {results.pose.map:.4f}")
        return model, results
    except Exception as e:
        print(f"Validation failed: {e}")
        return None, None

def adaptive_finetuning(base_model_path):
    print("\n" + "="*70)
    print("Adaptive Fine-Tuning (Algorithm 2)")
    print("="*70)
    base_model, base_results = validate_model(base_model_path, "base model")
    if base_model is None:
        raise RuntimeError("Failed to load base model")
    base_map = base_results.pose.map
    print(f"Base model mAP50-95: {base_map:.4f}")

    model = YOLO(base_model_path)
    train_args = {
        'data': 'keypoints.yaml',
        'epochs': 20,
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',
        'workers': 4,
        'patience': 0,
        'save': True,
        'project': 'runs/train',
        'name': 'adaptive_finetune',
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': 0.0002,
        'lrf': 0.002,
        'momentum': 0.95,
        'weight_decay': 0.00005,
        'warmup_epochs': 2.0,
        'warmup_momentum': 0.8,
        'hsv_h': 0.001,
        'hsv_s': 0.1,
        'hsv_v': 0.05,
        'degrees': 1.0,
        'translate': 0.01,
        'scale': 0.02,
        'fliplr': 0.1,
        'mosaic': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'close_mosaic': 3,
    }
    model.train(trainer=DynamicPoseTrainer, **train_args)

    finetuned_path = 'runs/train/adaptive_finetune/weights/best.pt'
    if not os.path.exists(finetuned_path):
        print("Best model not found, using current model")
        final_model = model
        finetuned_results = model.val(data='keypoints.yaml', imgsz=640, batch=4, device='cpu', workers=2, verbose=False)
    else:
        final_model, finetuned_results = validate_model(finetuned_path, "fine-tuned model")
    final_map = finetuned_results.pose.map
    improvement = final_map - base_map
    print(f"\nFine-tuned mAP50-95: {final_map:.4f}  (improvement: {improvement:+.4f})")
    return final_model, final_map

def save_model(model, save_name='adaptive_finetuned_model'):
    model_path = f'{save_name}.pt'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

def main():
    if not setup_environment():
        return
    base_model_path = 'runs/train/keypoints_model/weights/best.pt'
    if not os.path.exists(base_model_path):
        print(f"Base model not found: {base_model_path}")
        return
    final_model, final_map = adaptive_finetuning(base_model_path)
    save_model(final_model)
    print(f"\nAdaptive fine-tuning completed. Final mAP50-95: {final_map:.4f}")

if __name__ == "__main__":
    main()