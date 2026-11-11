# 条件B（GPT）を回すための依頼文

（以下をそのままGPTに渡す。リポジトリ：C:\Users\kinos\OneDrive\デスクトップ\MyProject\Video-Reconstruction）

---

条件Cと同じやり方で、今度は条件Bを6本（syn_01〜syn_06）、各1回実行してください。
条件Cで作った実行の仕組み（Codex CLIの独立セッション、ログ保存、監査）を流用してください。

## 条件をCとそろえる点

- モデル `gpt-6.1-sol`、推論 `medium`、クライアント Codex CLI（`codex exec`、`--ignore-user-config`）。
- 1本ごとに新しいセッション。会話履歴の引き継ぎ、途中の助言、結果のフィードバックはしない。
- 制限時間 60分（Claudeの条件Bと同じ）。時間切れでもそのまま採点する。

## 条件Bに特有の点（Cとの違い）

- 指示文：`runs/gpt/b/syn_XX/prompt.txt` の中身をそのまま渡す（CineRecon L1、Claudeと同じ文）。
- MCP：`runs/gpt/b/syn_XX/codex_mcp.toml` の `blender`（blend-ai）だけ。vreconのMCPはつながない。
  blend-aiは、ユーザーが起動したBlender（ポート9876）に接続する。開始前にポートが開いているか確認し、
  開いていなければ実行せずに報告する。
- 作業フォルダ（`-C`）：`runs/gpt/b/syn_XX/workspace`。サンドボックスは workspace-write（この中だけ書き込み可）。
- シェル：使ってよい（Claudeの条件Bも Bash/Read/Write を許可した）。映像のフレーム抽出（ffmpeg）や画像の閲覧も可。
  Web検索、別エージェント、アプリ連携は無効。
- 正解データ（`evaluation/synthetic/*/gt.json`, `gt_passes/`, `meta.json`）や、他の試行の出力、`runs/` の他のフォルダを
  読ませない。workspace の外を読んだ形跡がログにあれば、その試行は無効として記録する。
- GPUは使ってよい（Blenderのレンダー用）。ただし同時に1本だけ実行する（8GBしかない）。

## 1本ごとの手順

1. 毎回、Blenderの中身を空にしてから始める（前の試行のオブジェクトが残らないように）。
   空にする操作は試行の前にこちら側で行い、エージェントにはさせない。
2. `codex exec` で実行。ログは `runs/gpt/b/syn_XX/logs/agent.jsonl` に保存。
3. 終了後に採点：
   `.venv/Scripts/python.exe baselines/gpt_codex/finish.py b runs/gpt/b/syn_XX --model gpt-6.1-sol --minutes 60`
   （bpyが読み込めない場合は、条件Cのときと同様に代替環境で採点し、その旨を記録する）
4. 成果物（`workspace/output/scene.blend` など）がない場合も、失敗として記録する。成功扱いにしない。

## 最後に出してほしいもの

- 条件Cのときと同じ形式のレポート：映像ごとの IoU、焦点距離誤差、タイミング誤差、実行時間、成果物の有無。
- 監査：blend-ai 以外のMCPを呼んでいないこと、workspace 外のファイルや正解データに触れていないこと。
- 失敗・中断した試行は別に保存し、有効な6試行と混ぜない。
