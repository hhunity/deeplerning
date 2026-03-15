# CLAUDE.md

## プロジェクト概要
顕微鏡画像中の結晶（輝点）をカウントするディープラーニングプロジェクト。
EfficientNet-B0を使った回帰タスクがメイン。

## 環境
- Python 3.x（venv: `./venv`）
- PyTorch
- 仮想環境の有効化: `source venv/bin/activate`

## プロジェクト構成
- `efficientnet/` — EfficientNet-B0による結晶カウント（メイン）
- `crnet/` — CSRNet（密度マップ回帰）
- `unet/` — U-Net（セグメンテーション）
- `dqn/` — DQN（強化学習）
- `sam/` — Segment Anything Model
- `annotation/` — アノテーションツール
- `dataset/` — データセット

## コーディング規約
- コメントは英語で書く
- 変数名・関数名は英語（スネークケース）
- 学習スクリプトにはTensorBoardのログ出力を含める

## よく使うコマンド
```bash
source venv/bin/activate
tensorboard --logdir=runs
```

## タスクの優先度
1. Loss曲線の確認（必須）
2. 予測値の分布確認（回帰タスクなので特に重要）
3. Grad-CAMで結晶を正しく見ているか確認
