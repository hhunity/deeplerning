# csrnet_count.py
# PyTorchを使ったCSRNetによる物体カウント
# 基本的な使い方
#python csrnet_count.py crowd.jpg
# 学習済み重みを使う場合（推奨）
#python csrnet_count.py crowd.jpg --weights partA_pre.pth
# 結果を別ファイルに保存
#python csrnet_count.py crowd.jpg --weights partA_pre.pth --output result.png

#入力画像
#   ↓
#[フロントエンド] VGG-16 (conv1〜conv3) → 特徴マップ
 #  ↓
#[バックエンド]  Dilated Conv (dilation=2) → 密度マップ
 #  ↓
#カウント数 = 密度マップの全ピクセルの合計値



import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os


# ==================== モデル定義 ====================

class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
    論文: https://arxiv.org/abs/1802.10062
    
    構造:
      - フロントエンド: VGG-16の最初の10層（特徴抽出）
      - バックエンド: 拡張畳み込み（Dilated Conv）によるデンスマップ推定
    """

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        # フロントエンド: VGG-16 の conv1_1 〜 pool3 まで
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat)
        self.backend  = self._make_layers(self.backend_feat, in_channels=512, dilation=True)

        # 最終出力: 密度マップ（1チャンネル）
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._load_vgg_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _load_vgg_weights(self):
        """VGG-16 の事前学習済み重みをフロントエンドに転送"""
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_features = list(vgg16.features.children())
        
        # VGG-16 の最初の13層 (conv1〜conv3) をフロントエンドに対応付け
        frontend_children = list(self.frontend.children())
        
        # Conv層とBN層のみコピー（MaxPoolはスキップ）
        vgg_idx = 0
        for layer in frontend_children:
            if isinstance(layer, nn.Conv2d):
                while vgg_idx < len(vgg_features) and not isinstance(vgg_features[vgg_idx], nn.Conv2d):
                    vgg_idx += 1
                if vgg_idx < len(vgg_features):
                    layer.weight.data = vgg_features[vgg_idx].weight.data
                    layer.bias.data   = vgg_features[vgg_idx].bias.data
                    vgg_idx += 1
        print("[INFO] VGG-16 事前学習済み重みをフロントエンドに読み込みました。")


# ==================== 前処理 ====================

def get_transform():
    """ImageNetの統計量で正規化"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image_path: str):
    """
    画像を読み込み、テンソルに変換する。
    CSRNetは任意サイズの入力を受け付けるが、
    メモリ節約のため長辺を1024pxにリサイズする。
    """
    img = Image.open(image_path).convert('RGB')

    # 長辺を 1024px に制限（任意）
    max_size = 1024
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    transform = get_transform()
    tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return tensor, img


# ==================== 推論 ====================

def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    密度マップを推論し numpy 配列で返す。
    カウント数 = 密度マップの総和
    """
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        density_map = model(tensor)

    # (1, 1, H, W) → (H, W)
    density_map = density_map.squeeze().cpu().numpy()
    density_map = np.maximum(density_map, 0)  # 負値をゼロクリップ
    return density_map


# ==================== 可視化 ====================

def visualize(original_img: Image.Image,
              density_map: np.ndarray,
              count: float,
              save_path: str = None):
    """元画像と密度マップを並べて表示・保存"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 元画像 ---
    axes[0].imshow(original_img)
    axes[0].set_title("入力画像", fontsize=14)
    axes[0].axis('off')

    # --- 密度マップ ---
    im = axes[1].imshow(density_map, cmap=cm.jet, interpolation='bilinear')
    axes[1].set_title(f"密度マップ  (推定カウント: {count:.1f})", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("CSRNet — 物体カウント", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] 結果を保存しました: {save_path}")

    plt.show()


# ==================== メイン ====================

def main():
    parser = argparse.ArgumentParser(
        description="CSRNet で画像中の物体数をカウントします"
    )
    parser.add_argument('image', type=str,
                        help='入力画像のパス (JPG / PNG など)')
    parser.add_argument('--weights', type=str, default=None,
                        help='学習済み CSRNet 重みファイル (.pth/.pt)')
    parser.add_argument('--no-vgg-init', action='store_true',
                        help='VGG-16 事前学習重みを使わない')
    parser.add_argument('--output', type=str, default='output.png',
                        help='結果画像の保存先 (default: output.png)')
    parser.add_argument('--cpu', action='store_true',
                        help='CPU を強制使用')
    args = parser.parse_args()

    # --- デバイス設定 ---
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] 使用デバイス: {device}")

    # --- モデル構築 ---
    load_vgg = not args.no_vgg_init
    model = CSRNet(load_weights=load_vgg).to(device)

    # CSRNet 学習済み重みがあれば読み込む
    if args.weights:
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"重みファイルが見つかりません: {args.weights}")
        state = torch.load(args.weights, map_location=device)
        # checkpoint が 'state_dict' キーを持つ場合に対応
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        print(f"[INFO] 重みを読み込みました: {args.weights}")
    else:
        print("[WARN] CSRNet 学習済み重みが指定されていません。")
        print("       VGG-16 初期化のみで推論します（精度は低くなります）。")
        print("       公式モデルは以下から入手できます:")
        print("       https://github.com/leeyeehoo/CSRNet-pytorch")

    # --- 前処理 ---
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {args.image}")

    tensor, original_img = preprocess_image(args.image)
    print(f"[INFO] 入力画像サイズ: {original_img.size[0]}x{original_img.size[1]}")

    # --- 推論 ---
    density_map = predict(model, tensor, device)
    count = float(density_map.sum())
    print(f"\n{'='*40}")
    print(f"  推定カウント数: {count:.1f}")
    print(f"{'='*40}\n")

    # --- 可視化 ---
    visualize(original_img, density_map, count, save_path=args.output)


if __name__ == '__main__':
    main()
