# 引き継ぎ資料（2026-11-06 更新：ローカルのWindows環境に移行）

新しいセッション（ローカルの Claude Code など）は、まずこれを読む。詳しい方針は `docs/PLAN.md`（計画書、最優先）、
設計は `docs/ARCHITECTURE.md`、座標などの取り決めは `docs/CONVENTIONS.md`。

## 1. このプロジェクトは何か（3行）

- 参考映像を、**後から編集できる Blender の3Dシーン**（カメラ、配置、動き、タイミング）に作り直す。
- その3Dから書き出した奥行きなどの映像を使って、**動画生成AI（MiniMax-H3＋Fun ControlNet Union 2.0）で最終映像**を作る。
- 研究・非営利。問いは2つ。RQ1：構造化された診断と修正の道具で、素のLLM＋MCPより再構成が良くなるか。RQ2：3Dで加えた編集が生成映像に残るか。

## 2. 今どこまで進んだか

計画書12章の段階でいうと、**段階0は完了**。段階1a・1bは準備完了で、実行待ち（GPUやGUI版Blenderが必要）。
段階2（診断と修正）は、仕組みが動くところまでできてる。

| 部分 | 状態 | 場所 |
|---|---|---|
| データの型（4層、版、ロック） | 完了 | `src/vrecon/schema/` |
| Blenderビルダー、描画、`.blend` 保存 | 完了 | `src/vrecon/scene/builder.py` |
| 人の修正の読み戻し（handoff / ingest） | 完了 | `scene/readback.py`、`scene/ingest.py` |
| 動画の取り込み、カット分割 | 完了 | `src/vrecon/video.py` |
| V2V用の制御映像（奥行き、輪郭）書き出し | 完了 | `scene/passes.py`、`control/` |
| 1a：H3テスト用シーン（基準＋編集3つ）、制御映像 | 完了 | `evaluation/phase1a/` |
| 1a：GPUでH3を回すスクリプト | 書いたけど**未実行** | `gpu/h3_control/`（手順書あり） |
| 正解つき合成映像6本 | 完了 | `evaluation/synthetic/` |
| 採点（IoU、カメラ、再投影、タイミング） | 完了 | `src/vrecon/metrics/`、`evaluate.py` |
| 1b：基準点B（素のLLM＋blender-mcp）の実行キット | 書いたけど**未実行** | `baselines/b_llm_mcp/`（手順書あり） |
| 実行管理（提案→制約→評価→採用） | 完了 | `src/vrecon/run/manager.py` |
| 数値探し（画角、位置、大きさ、タイミング） | 動く。画角は6本全部で回復。タイミングは2〜4フレーム残る | `src/vrecon/optimize/` |
| 点の追跡（OpenCV Lucas-Kanade、重みなし）と、追跡点のズレの損失 | 完了 | `src/vrecon/optimize/tracks.py` |
| 決め打ちの修理（`vrecon run auto`、MCPの `run_auto`）＝条件CのAIなし版 | 完了 | `src/vrecon/run/policy.py` |
| 条件の比較レポート（H1a・H1bの表） | 完了（偽データでテスト済み） | `scripts/compare_conditions.py` |
| 基準値ファイルのひな形（まだ下書き、固定前） | 完了 | `evaluation/thresholds.yaml` |
| CLI / MCPサーバー | 完了 | `src/vrecon/cli.py`、`src/vrecon/mcp_server.py` |
| 修理ベンチマーク | 結果あり | `evaluation/repair_benchmark/`（`NOTES.md` に考察） |

テストは88件、全部合格（`.venv/Scripts/python.exe -m pytest -q`。作業フォルダを日本語パスにした `--basetemp` でも確認済み）。

ローカルで修理ベンチマークを再実行（2026-11-06）：6本で約8分40秒、数値はクラウドの結果（`evaluation/repair_benchmark/`）とほぼ同じ。
少し差が出たのは syn_04（IoU 0.883→0.860、採用 12→11）と syn_06（IoU 0.876→0.873、採用 14→13）。原因は未確認（動画の読み込みを ffmpeg に替えた影響か、OS差の可能性）。`evaluation/repair_benchmark/` の記録はクラウド時の結果のまま。

## 3. 決まっていること（変えるなら計画書も直す）

- ゴールは「3Dの下書き」と「最終映像」の両方（計画書1章）。最初の完成地点は M1（3.1章）。
- V2Vは **MiniMax-H3＋Fun ControlNet Union 2.0**（9.1章）。ライセンス原文は確認済み。FastH3-Omni は条件A（直接生成）の候補。
- H3の入力条件：24fps固定、幅と高さは32の倍数、フレーム数は17n+5、`guidance_scale` は1.0。
- LLMはプロジェクトを直接書き換えない。「提案」を出して、実行管理部分が採用を決める（6章）。
- 正解データ（`gt.json`、`gt_passes/`）は**採点専用**。再構成側に絶対に渡さない（11.8章）。
- 基準値は、検証用の映像を見る前に `evaluation/thresholds.yaml` で固定する（11.6章、まだ作ってない）。

## 4. 分かっている課題

1. **参照映像の「物の領域」は、まだ正解マスクで代用してる。** 本物の分割ツール（SAM2など）が要る。
   ローカルでは `huggingface.co` と `download.pytorch.org` に繋がることを確認済み（2026-11-06）。
   差し替える場所：`optimize/__init__.py` の `gt_stand_in_masks`（同じ形の `MaskObservation` を返せばいい）。
2. ~~画角の推定が不安定~~ → **解決**（合成映像で）。点の追跡を足して、カメラの提案を「床・壁の輪郭と、床・壁に乗った点」だけで
   判断・採用するようにした（`evaluation/repair_benchmark/NOTES.md`）。
3. **タイミングが2〜4フレーム残る。** 一定速度の動きでは「遅れて動く」と「先にずらして置く」が見分けにくい。
4. GPU側（`gpu/h3_control/run_jobs.py`）と基準点B（`baselines/b_llm_mcp/run_baseline.sh`）は、**まだ一度も実際に動かしてない**。最初は1本だけ試す。
5. 本物のテスト映像（単一カット12本：開発用6本、検証用6本）がまだない。
6. 修理の順番（画角 → 物ごとの位置 → 大きさ → タイミング）は、今は決め打ち（`vrecon run auto`）。AIに選ばせる版（条件Cの本番）はまだ。
7. 条件Cの**最初の再構成**（修理の出発点）を作る部分がまだない。ベンチマークは正解にわざと誤差を入れたものから始めている。
   本番では、AI（＋blender-mcp）が作った最初のシーンを `score-blend` と同じ方法で取り込んで、修理に回す想定。

## 5. 次にやること（おすすめ順。計画書16章の見直しを反映）

1. ~~ローカルで環境を作って、テストが通るのを確認する~~ → 完了（6章）。
2. **観測の道具を入れる**（計画書16.3）：SAM 3（領域分割）、CoTracker3 または TAPNext++（点の追跡）。
   合成映像6本の修理ベンチマークで、正解マスクの代用と比べる。差し替え場所：`optimize/__init__.py` の `gt_stand_in_masks`、
   `optimize/tracks.py` の `track_points`（同じ形のデータを返せばいい）。
3. **最初の再構成を4D再構成モデルで作る**（計画書16.2、課題7）：MoRe か MegaSaM の出力から、カメラと物の大まかな位置を作る。
4. **基準点Bを1本回す**（`baselines/b_llm_mcp/README.md`。GUI版Blenderが要る）。
5. **AIに仮説を選ばせる**：`vrecon-mcp` を Claude Code に繋いで、`run_diagnose` → 仮説 → `run_optimize` を回させる。
6. **H3で1本だけ生成してみる**（`gpu/h3_control/README.md`。手元のGPUでは足りないのでレンタルGPUで）。奥行き・輪郭・グレー（プロキシの描画）の3種類を比べる（計画書16.4）。
7. 本物のテスト映像を集める。評価データの候補に BVB と CineRecon も加える（計画書16.1）。

**研究の新規性は RQ1（診断・数値探し・採用ルール）に置く**（計画書16.1）。RQ2 に近い先行研究（Shape-for-Motion、ProxyUp）がある。

## 6. ローカル環境

今の開発環境：Windows 11、Core i7-13700H、RAM 32GB、RTX 4060 Laptop（VRAM 8GB）、Blender 5.2 GUI版（`bpy` は5.2.3）。

必要なもの：Git、Python 3.13（`bpy` 5.2.3 は3.13専用）、`uv`、`ffmpeg`。GUI版の Blender 5.2.x は、基準点Bを回すときだけ要る。

- Windows：Git Bash で `bash scripts/setup.sh`、または README の「Windows」の手順。PowerShell で `bash` と打つと WSL が起動して、別の環境になるので注意。
- Linux / macOS：`bash scripts/setup.sh`（Mesaのインストールは Linux のときだけ行う）

Windowsで気をつけること：
- **フォルダのパスに日本語が入ってる。** OpenCVのファイル関数は日本語パスを開けないので、`control/core.py` の `load_frames`、`read_video_rgb`、`encode_mp4` を使う。
- **文字コードは必ず `encoding="utf-8"` を指定する。** 指定しないと cp932 になる。
- pytest の出力に `Windows fatal exception: code 0xc0000139` が出ることがあるけど、`import bpy` などでDLLを読むときの表示で、処理は続いてる。
- `.venv` や `runs/` が OneDrive の中にある。大きいファイルが増えると同期の負担になる（置き場所は計画書15章の未決事項）。

描画（1280×704で測定、2026-11-06）：

| 処理 | CPU | GPU（CUDA / OptiX） | 使う方 |
|---|---|---|---|
| 奥行き・領域のデータ描画（1サンプル、1フレーム） | 0.09秒 | 0.20秒 / 0.28秒 | CPUに固定（領域の値はGPUと全画素一致） |
| 見た目の描画（64サンプル、1フレーム） | 11.9秒 | 4.0秒 / 4.0秒 | GPU（環境変数 `VRECON_RENDER_DEVICE` で変更可） |

手元のPCでの実行の見込み（**VRAMの目安からの見積もりで、まだ動かしてない**）：
- 手元で試す：SAM 3、CoTracker3 / TAPNext++、基準点B。8GBに収まるかは実測で確かめる。MoRe / MegaSaM は入るかどうか不明。
- レンタルGPUが要る：MiniMax-H3（目安はVRAM 80GB）。

Gitに入れてない（作り直せる）もの：
- `runs/`（ベンチマークの途中経過）
- `evaluation/synthetic/*/gt_passes/` → `python scripts/make_synthetic_clips.py --passes-only` で作り直せる（数分）
- `evaluation/phase1a/controls/*/passes/` → `vrecon control` を再実行

## 7. よく使うコマンド

```
vrecon new 動画.mp4 --out myproj                         # 動画を取り込んでカット分割
vrecon build project.json --blend out.blend               # 3Dシーンを作る
vrecon handoff project.json --out h/  →  vrecon ingest project.json --handoff h/handoff.json
vrecon control project.json --out ctl/                    # V2V用の制御映像
vrecon score evaluation/synthetic/syn_01 project.json     # 正解と比べて採点
vrecon score-blend evaluation/synthetic/syn_01 scene.blend
vrecon run init|state|diagnose|optimize|auto|submit|history|rollback   # 修理ループ（auto = 決め打ちの順番）
python scripts/compare_conditions.py 結果フォルダ --ref B --cand C  # 条件の比較表（結果フォルダ/<条件>/<クリップ>/<回>/score.json）
python scripts/repair_benchmark.py                        # 修理ベンチマーク（6本で約15分、点の追跡つき）
```

MCPで使う場合：`claude mcp add vrecon -- <repo>\.venv\Scripts\python.exe -m vrecon.mcp_server`
（Linuxは `<repo>/.venv/bin/python`）

## 8. ユーザーとの約束

- 説明は日本語、カジュアル、結論から。専門知識がない人にも分かるように。誇張はしない。
- クレジットを節約する：まとめて作ってまとめてテスト、全体テストはコミット前の1回、大きい出力は必要なときだけ。
- コミットとpushは、ユーザーが頼んだときに `main` へ。
