# 基準点：条件B（汎用LLM＋blender-mcp）

計画書の条件B（`docs/PLAN.md` 11.1）。補強なしの「素のAI＋Blender MCP」で映像を再構成させて、点数を取る。
条件C（自作の診断と修正ツールあり）と比べる相手になる。

## 公平にするために固定するもの

| 項目 | 値 | 理由 |
|---|---|---|
| 指示文 | `prompt_L1.txt` | CineRecon の L1（Proxy）をほぼそのまま使う。変更は1か所だけ：「CUDA GPUで描画」→「GPUがあれば使う」 |
| MCP | `mcp-for-blender` 2.2.6（blender-mcp） | 版を固定。CineRecon は 1.9.1 を使っていたので、数値を直接比べるときは注意 |
| Blender | 5.2.x（GUI版） | blender-mcp は `blender -b`（画面なし）では動かない。Linuxで画面がないときは Xvfb で仮想画面を使う（CineReconと同じ） |
| LLM | 条件Cと同じモデル | 計画書 11.1 |
| 予算 | 60分、200ターン（既定値） | 予備実験のあと、計画書 11.6 の手順で固定する |
| 外部サービス | 使わない | 指示文で禁止。テレメトリーもオフ |

`prompt_L1.txt` の元は CineRecon（MIT License、`LICENSE.CineRecon`、コミット `1d85031`）。

## 正解データを見せない仕組み（計画書 11.8）

- AIの作業フォルダには、参照映像（`reference.mp4`）だけをコピーする。正解（`gt.json`、`gt_passes/`、`meta.json`）は入れない。
- 実行後、AIのログに正解ファイルの名前（`gt.json`、`gt_passes`、`meta.json`、`/evaluation/`）が出てきたら、**その実行は無効（`valid: false`）**にする。
- 実行フォルダ（RUN_DIR）は `evaluation/` の外に作ること。

## 実行に必要なもの

- **Blender 5.2.x の GUI版**：手元のWindows PCに入ってる（`C:\Program Files\Blender Foundation\Blender 5.2`。環境変数 `BLENDER` がなく、`blender` にパスが通ってなければ、ここを自動で使う）
- `uv` / `uvx`、`claude` CLI、`ffmpeg`
- このリポジトリの `.venv`（採点に使う。`bash scripts/setup.sh`）
- Linuxで画面がない場合は `xvfb-run`

WindowsではGit Bashで実行する。手元のPCはRTX 4060があるので、CineRecon と同じくCyclesをGPUで描画できる。

実行すると `install-addon` で blender-mcp のアドオンが Blender のユーザー設定に入る可能性がある（手元の 5.2 には今 `blend_ai` などのアドオンがある）。**Windowsでの実行は、まだ一度も確かめてない。**

## 実行

```bash
bash baselines/b_llm_mcp/run_baseline.sh evaluation/synthetic/syn_01 ~/vrecon_runs/B/syn_01_r1 --model claude-opus-5-5
```

やること：
1. 参照映像を作業フォルダにコピーして、指示文の中のパスを埋める
2. Blender（GUI）を起動して、blender-mcp のサーバーを立てる（Linuxで画面がなければ Xvfb）
3. `claude -p` で、指示文をAIに渡す（時間とターン数に上限あり。ログは `logs/agent.jsonl`）
4. `finish_run.py`：成果物の有無、ターン数、費用、ツールの使用回数、正解の漏れを `run.json` に記録して、`scene.blend` を採点して `score.json` に書く

1クリップにつき3回（11.5）。シードは指定できないから、別々の実行として数える。

## 採点だけしたいとき

```bash
.venv/Scripts/vrecon score-blend evaluation/synthetic/syn_01 path/to/scene.blend --out score.json   # Linuxは .venv/bin/vrecon
```

AIが作った `.blend` は、物の名前もカメラの設定もバラバラ。だから採点の前に次のように揃える：
- 物の対応付けは、画面上の重なりで自動的に行う（名前は使わない）
- 全フレームを読み取る（キーフレームの付け方に依存しない）
- 解像度が違っても、縦横比が同じなら参照の解像度に合わせる（警告を記録）。縦横比が違ったら採点しない

## 未確認のこと

- `run_baseline.sh` は、Blender GUI と `claude` を実際に通してはまだ動かしてない（このクラウド環境では Blender GUI が入らないため）。採点の後処理（`finish_run.py`）だけ自動テスト済み。最初の1回は様子を見ながら実行してほしい。
- アドオンの自動有効化（`--python-expr` の部分）は、アドオンのモジュール名が変わると効かない。そのときは Blender で手動で有効にして、自動起動をオンにする。
