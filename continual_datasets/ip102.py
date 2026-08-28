import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class IP102(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = []
        self.targets = []
        self.classes = []
        
        self._load_dataset()
        
    def _find_file(self, filename, search_paths):
        for path in search_paths:
            if not os.path.exists(path):
                continue
            for root_dir, dirs, files in os.walk(path):
                if filename in files:
                    return os.path.join(root_dir, filename)
        return None

    def _load_dataset(self):
        # Determine search paths
        search_paths = [self.root, '/kaggle/input', os.environ.get('IP102_DATA_PATH', self.root)]
        
        filtered_class_path = self._find_file('filtered_class.txt', search_paths)
        if filtered_class_path is None:
            # Fallback gracefully or raise
            raise RuntimeError("Could not find filtered_class.txt")
            
        classes_txt_path = self._find_file('classes.txt', search_paths)
        if classes_txt_path is None:
            raise RuntimeError("Could not find classes.txt")
            
        # Parse filtered classes
        with open(filtered_class_path, 'r') as f:
            valid_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
            
        assert len(valid_ids) == 25, f"Must have exactly 25 classes in filtered_class.txt, found {len(valid_ids)}"
        
        # Parse class names
        class_names_map = {}
        with open(classes_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    class_names_map[int(parts[0])] = parts[1]
                    
        self.classes = [class_names_map.get(vid, str(vid)) for vid in valid_ids]
        
        # We need a contiguous mapping for valid_ids to 0..24
        valid_id_to_contiguous = {vid: i for i, vid in enumerate(valid_ids)}
        
        # Find split json
        json_name = f'{self.split}.json'
        json_path = self._find_file(json_name, search_paths)
        if json_path is None:
            # Fallback for val
            if self.split == 'val':
                json_name = 'test.json'
                json_path = self._find_file(json_name, search_paths)
        
        if json_path is None:
            raise RuntimeError(f"Could not find {json_name}")
            
        # Read json
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
            
        if isinstance(data_json, dict) and 'images' in data_json and 'annotations' in data_json:
            img_id_to_name = {img['id']: img.get('file_name', img.get('filename')) for img in data_json['images']}
            annotations = []
            for ann in data_json['annotations']:
                img_name = img_id_to_name.get(ann['image_id'])
                if img_name:
                    ann['file_name'] = img_name
                    annotations.append(ann)
        elif isinstance(data_json, dict) and 'annotations' in data_json:
            annotations = data_json['annotations']
        elif isinstance(data_json, dict) and 'images' in data_json:
            annotations = data_json['images']
        elif isinstance(data_json, list):
            annotations = data_json
        else:
            raise RuntimeError("Unknown JSON structure")
            
        # Find images dir
        if len(annotations) > 0:
            first_img_name = annotations[0].get('file_name', annotations[0].get('filename', str(annotations[0].get('id', '')) + '.jpg'))
            first_img_path = self._find_file(first_img_name, search_paths)
            if first_img_path:
                images_dir = os.path.dirname(first_img_path)
            else:
                raise RuntimeError(f"Could not find image directory containing {first_img_name}")
        else:
            raise RuntimeError("Empty annotations")
                
        for ann in annotations:
            cat_id = int(ann.get('category_id', -1))
            if cat_id in valid_ids:
                img_name = ann.get('file_name', ann.get('filename', str(ann.get('id', '')) + '.jpg'))
                img_path = os.path.join(images_dir, img_name)
                
                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.targets.append(valid_id_to_contiguous[cat_id])
                else:
                    print(f"Warning: image {img_path} not found.")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
