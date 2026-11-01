# 引き継ぎ資料（2026-11-02 時点）

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
| 数値探し（画角、位置、大きさ、タイミング） | 動く。**課題あり** | `src/vrecon/optimize/` |
| CLI / MCPサーバー | 完了 | `src/vrecon/cli.py`、`src/vrecon/mcp_server.py` |
| 修理ベンチマーク | 結果あり | `evaluation/repair_benchmark/`（`NOTES.md` に考察） |

テストは80件、全部合格（`python -m pytest -q`）。

## 3. 決まっていること（変えるなら計画書も直す）

- ゴールは「3Dの下書き」と「最終映像」の両方（計画書1章）。最初の完成地点は M1（3.1章）。
- V2Vは **MiniMax-H3＋Fun ControlNet Union 2.0**（9.1章）。ライセンス原文は確認済み。FastH3-Omni は条件A（直接生成）の候補。
- H3の入力条件：24fps固定、幅と高さは32の倍数、フレーム数は17n+5、`guidance_scale` は1.0。
- LLMはプロジェクトを直接書き換えない。「提案」を出して、実行管理部分が採用を決める（6章）。
- 正解データ（`gt.json`、`gt_passes/`）は**採点専用**。再構成側に絶対に渡さない（11.8章）。
- 基準値は、検証用の映像を見る前に `evaluation/thresholds.yaml` で固定する（11.6章、まだ作ってない）。

## 4. 分かっている課題

1. **参照映像の「物の領域」は、まだ正解マスクで代用してる。** 本物の分割ツール（SAM2など）と点の追跡（CoTrackerなど）が要る。
   前のクラウド環境は Hugging Face に繋がらなかったので入れられなかった。ローカルなら入れられるはず。
2. **画角の推定が、輪郭だけでは安定しない**（`evaluation/repair_benchmark/NOTES.md`）。物も含めて判断すると、物を動かして辻褄を合わせてしまう。
   床・壁だけで判断すると、カメラが水平に近いとき手がかりが足りない。→ 点の追跡を観測に加えるのが次の一手（未検証）。
3. **タイミングが2フレーム前後残る。** 一定速度の動きでは「遅れて動く」と「先にずらして置く」が見分けにくい。
4. GPU側（`gpu/h3_control/run_jobs.py`）と基準点B（`baselines/b_llm_mcp/run_baseline.sh`）は、**まだ一度も実際に動かしてない**。最初は1本だけ試す。
5. 本物のテスト映像（単一カット12本：開発用6本、検証用6本）がまだない。

## 5. 次にやること（おすすめ順）

1. ローカルで環境を作って、テストが通るのを確認する（下の手順）。
2. **点の追跡と領域分割の道具を入れる**（課題1、2）。観測の提供元を `optimize/__init__.py` の `gt_stand_in_masks` と差し替える。
3. **基準点Bを1本回す**（`baselines/b_llm_mcp/README.md`。GUI版Blenderが要る）。
4. **H3で1本だけ生成してみる**（`gpu/h3_control/README.md`。80GB級のGPUを借りる）。
5. 本物のテスト映像を集める。

## 6. ローカル環境の作り方

必要なもの：Git、Python 3.13（`bpy` 5.2.3 は3.13専用）、`uv`、`ffmpeg`。GUI版の Blender 5.2.x は、基準点Bを回すときだけ要る。

- Linux / macOS / WSL：`bash scripts/setup.sh`（Mesaのインストールは Linux のときだけ行う）
- Windows（PowerShell）：README の「Windows」の手順。`scripts/*.sh` と `gpu/`、`baselines/` のシェルスクリプトは bash 用なので、
  使うときは WSL か Git Bash で動かす。

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
vrecon run init|state|diagnose|optimize|submit|history|rollback   # 修理ループ
python scripts/repair_benchmark.py                        # 修理ベンチマーク（6本で約15分）
```

MCPで使う場合：`claude mcp add vrecon -- <repo>/.venv/bin/python -m vrecon.mcp_server`
（Windowsは `<repo>\.venv\Scripts\python.exe`）

## 8. ユーザーとの約束

- 説明は日本語、カジュアル、結論から。専門知識がない人にも分かるように。誇張はしない。
- クレジットを節約する：まとめて作ってまとめてテスト、全体テストはコミット前の1回、大きい出力は必要なときだけ。
- コミットとpushは、ユーザーが頼んだときに `main` へ。
