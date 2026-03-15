

### 古典的画像処理

実は1μmの輝点カウントは、ディープラーニング不要で解ける可能性が高いです。 
グレースケール化 → ガウシアンフィルタ → 二値化 → 連結成分ラベリング → 個数カウント

| アルゴル | 説明 |
|-----|-----|
| findContours |  |
| SimpleBlobDetector |  |
| 手法狙いLoG / DoG | フィルタ結晶はスポット状、窪みは広域 → サイズで分離 |
| Top-hat | 変換局所的な輝点だけを抽出|
| 局所正規化（CLAHE）|窪み内の輝度ムラを補正してから検出|

または２段階

Step1: 窪み領域を検出・マスク（形状・サイズで比較的容易）
Step2: 窪み内だけを切り出して結晶を検出

### 第2候補：DLを使うなら

| アルゴル | 説明 |
|-----|-----|
|YOLOv8n (nano) |軽量・高速輝点検出→カウントに転用しやすい |
|U-Net | セグメンテーション輝点領域を分割してカウントCSRNet |
| DM-Count | 密度マップ回帰群衆カウント技術を転用|
|　Res-Net | 深さだけ増やす |
| EfficientNet-B0  | 幅、解像度、深さを増やす　シンプルな分類→回帰1〜10個分類なら十分な精度|

EfficientNet-B0が最適


### 学習の仕方

1. データの取得
    方針内容最低目標200〜300枚（回帰なら意外と少なくて済む）
    アノテーション内容各画像に対して「結晶の個数」を記録するだけでOK
    クラスバランス0個・1個・2個…10個がなるべく均等になるよう収集

2. データ拡張（Augmentation）
~~~
transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 左右反転
    transforms.RandomVerticalFlip(),     # 上下反転
    transforms.RandomRotation(90),       # 回転
    transforms.GaussianBlur(3),          # ぼかし（ラインセンサノイズ模擬）
    transforms.RandomErasing(),          # ランダム消去
])
~~~

3. 転移学習

~~~
# Stage1: ヘッドだけ学習
for param in model.features.parameters():
    param.requires_grad = False  # バックボーン凍結

optimizer = Adam(model.classifier.parameters(), lr=1e-3)

# Stage2: 全体をFine-tuning
# Stage1の結果を使用して、再学習
for param in model.features.parameters():
    param.requires_grad = True

optimizer = Adam(model.parameters(), lr=1e-5)  # lrを下げる
~~~

なぜ2段階にするのか
Stage1をスキップしていきなり全体を学習すると、ランダム初期化されたヘッドの大きな勾配がバックボーンまで伝わって、せっかくのImageNetのweightsが壊れてしまうリスクがあります。Stage1でヘッドを先に安定させることで、それを防いでいます。

- これは、Step1で大まかに移動させて、Step2で細かく制御するってこと？
  - Stage2ではバックボーンも動かすので、学習率が高いとImageNetで学習した大事なweightsを壊してしまうリスクがあります。
  
4. 過学習対策
   
| 手法 | 設定目安 |
|------|---------|
| **Dropout** | 0.3〜0.5をヘッドに追加 |
| **Weight Decay** | `Adam(..., weight_decay=1e-4)` |
| **Early Stopping** | val_lossが5epoch改善しなければ停止 |
| **K-Fold交差検証** | データが少ないときに特に有効 |

Stage1: バックボーンを凍結 → ヘッドだけ学習（10〜20epoch）
             ↓ lossが安定したら
Stage2: バックボーンを解凍 → 全体を低いlrでFine-tuning

5. 学習が収束しないときのチェックリスト
□ 学習率が高すぎる   → 1e-4 から始める
□ バッチサイズが小さすぎる → 16〜32が目安
□ 損失関数が合っていない → 回帰はMSEよりMAEかHuber Lossが安定しやすい
□ 入力の正規化忘れ   → mean/stdで正規化する

## 全体のおすすめ進め方
① 300枚収集 → EfficientNet-B0で動作確認
      ↓
② 1000枚まで増やす + Augmentation
      ↓
③ K-Fold + Early Stopping で過学習対策
      ↓
④ 精度不足なら Fine-tuning or データ追加

### 画像の増やし方
Copy-Paste Augmentation
クロップは、アノテーションの再配置の必要があるかもなので
注意が必要。
コピーする。ブラーで境界をなじませる。ノイズを付加する。

###  Grad-CAM　と　特徴マップ
Grad-CAMは重みそのものではなく、「この画素が出力にどれだけ影響したか」 を可視化するものです。

推論画像を入力
   ↓
順伝播 → 予測（3.1個）
   ↓
逆伝播 → 勾配を読む（weightsは更新しない）
   ↓
「どの画素が予測に影響したか」をヒートマップ化
   ↓
元画像に重ねて表示

入力画像 → バックボーン → 最終特徴マップ → ヘッド → 出力（3.2個）
                              ↑
                         ここに注目

pip install tensorboard

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# 学習中に記録
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('lr', lr, epoch)

# ブラウザで確認
# tensorboard --logdir=runs
```

Loss曲線・予測分布・Grad-CAMを**全部TensorBoardにまとめて見る**のが定番のやり方です。

---

## 優先順位
```
① Loss曲線         ← 必ず見る
② 予測値の分布      ← 回帰タスクなので特に重要
③ Grad-CAM         ← 精度が出ないときに見る

~~~
pip install grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=model, target_layers=[model.features[-1]])
grayscale_cam = cam(input_tensor=input_image)
~~~

フィルタの重みモデルが持っている固定のパターン「エッジ検出フィルタ」などGrad-CAMこの画像に対する出力への影響度「この画像のここを見て判断した」

||特徴マップ|Grad-CAM|
|------|---------|---------|
|何を見る|層の出力をそのまま表示|出力への影響度を計算して表示|
|逆伝播|使わない|使う|
|結果|チャンネルごとに別々の画像|1枚のヒートマップ|

|Grad-CAMで見て実際にやること||
|------|---------|
|Grad-CAMの結果|やること|
|窪みの縁を見ている|データ・Augmentationを見直す|
|結晶以外の場所を見ている|アノテーションを見直|
|すぼんやりしている|データを増やす|
|ちゃんと結晶を見ているのに精度が低い|このときだけアーキテクチャ検討|

### Loss曲線

|パターン意味||
|--|--|
|train/val lossが両方下がる|正常train|
|lossだけ下がる|過学習|
|両方下がらない|学習率・モデルが合っていない|

### 予測値の分布

|パターン|意味|
|---|----|
|対角線上に並ぶ|正常|
|全部平均値付近に集まる|特徴を学習できていない|
|特定の個数だけずれる|データ不均衡|

### 学習率スケジューラの確認

学習率を学習の途中で自動的に変化させる仕組みです！
epochが進むにつれて学習率を自動で下げていきます。

