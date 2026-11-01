# フェーズ1a：GPUでの生成（MiniMax-H3＋Fun ControlNet Union 2.0）

3Dシーンから書き出した制御用の映像（奥行き、輪郭）を使って、H3で映像を生成する。
目的は「3Dで加えた編集（カメラ、物、タイミング）が、生成後の映像に残るか」を測ること（`docs/PLAN.md` 11.4、12）。

## 流れ（パターンA）

```
クラウド（済み）：テストシーン → 制御用の映像 → jobs.json を push
   ↓
GPUマシン：pull → setup_gpu.sh → run_jobs.py → pack_results.sh → 小さい結果を push
   ↓
クラウド：結果を読んで採点（次の作業）
```

## 0. 使う前に

- [ ] **MiniMax H3 Community License の原文を読む**（対象地域、生成映像の公開範囲）。`docs/PLAN.md` 9.1。

## 1. GPUマシンを用意する

| 項目 | 目安 | 理由 |
|---|---|---|
| GPU | 80GB（A100 80GB か H100） | 本体のTransformerだけでbf16で約62GB |
| メインメモリ（RAM） | 多め（できれば200GB前後） | 標準の `model_group_offload` は、使ってない層をRAMに置く。テキストエンコーダー（約62GB）もある |
| ディスク | 250GB以上 | 重み：本体＋テキストエンコーダーで約124GB、ControlNet 2.0で約13.5GB |

**この表の数字は、VideoX-Funの説明とモデルカードから見積もったもの。実際に動かして確かめてない。**
RAMが足りないときは、`--memory-mode model_cpu_offload_and_qfloat8`（Transformerを8bitにする。画質が少し落ちる可能性あり）を使う。使った設定は結果に自動で記録される。

## 2. セットアップ

```bash
git clone https://github.com/TaroKinoshita/Video-Reconstruction.git
cd Video-Reconstruction
bash gpu/h3_control/setup_gpu.sh            # Hugging Face から取得
# bash gpu/h3_control/setup_gpu.sh --source modelscope   # 繋がりにくいとき
```

- VideoX-Fun はコミット `e8c314f` に固定して `third_party/VideoX-Fun` に入れる。
- `MiniMax-H3` の中が `transformer/` じゃなくて `FL2VA/` などのフォルダ分けになってたら、実行時に `--model models/Diffusion_Transformer/MiniMax-H3/FL2VA` を付ける。

## 3. 確認だけする（GPUを使わない）

```bash
python gpu/h3_control/run_jobs.py --jobs evaluation/phase1a/jobs.json --out runs/phase1a/generated --dry-run
```

制御用の映像があるか、解像度、24fps、フレーム数（17n+5）をチェックして、実行予定の一覧を出す。

## 4. 生成する

```bash
python gpu/h3_control/run_jobs.py --jobs evaluation/phase1a/jobs.json --out runs/phase1a/generated
```

- 標準は `main` グループ：4シーン（基準＋編集3つ）× 奥行き × シード3つ ＝ **12本**。
- 輪郭（canny）の分も回すなら `--group extra`、全部なら `--group ""`。
- 1本だけ試すなら：`--only base__depth --seeds 43`
- 途中で止まっても、もう一度実行すれば終わった分は飛ばして続きから回る。
- 1本ごとに `video.mp4`（音声つき。H3は音も生成する）、`sheet.jpg`（最初・真ん中・最後のフレーム）、`meta.json`（設定、シード、かかった時間、GPUメモリの最大値、環境）を保存する。

**時間はまだ分からない。** まず1本だけ回して時間を測ってから、全部回すのがおすすめ。

## 5. 結果を戻す

```bash
bash gpu/h3_control/pack_results.sh
git add evaluation/phase1a/results && git commit -m "phase1a results" && git push
```

- `evaluation/phase1a/results/` に小さいファイル（`meta.json`、`sheet.jpg`、`results_manifest.json`）だけが入る。これをpushすれば、クラウド側で確認と採点の準備ができる。
- 動画は `runs/phase1a_videos_*.tar` にまとまる。自分のPCにダウンロードしておく（Gitには入れない）。

## 生成の設定（jobs.json）

| 項目 | 値 | 根拠 |
|---|---|---|
| 解像度 | 1280×704（高さ704、幅1280） | 32の倍数 |
| フレーム数 | 124（約5.2秒） | H3は17n+5の長さで生成する |
| fps | 24 | H3は24固定 |
| ステップ数 | 40 | VideoX-Funの例と同じ |
| guidance_scale | 1.0 | 配布されたチェックポイントは蒸留済みで、1より大きくすると画質が落ちる（VideoX-Funの注記） |
| control_context_scale | 1.0 | 制御の強さ。0で制御なし |
| シード | 43, 44, 45 | 計画書 11.5：1条件3回以上 |
