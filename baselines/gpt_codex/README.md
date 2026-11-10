# GPT（Codexアプリ）で条件BとCを回す手順

Claudeで回したときと同じ映像、同じ指示文、同じ採点で、GPTに条件BとCをやらせるための手順。
BとCは**必ず同じモデル**で回す（モデルの差とツールの差が混ざらないように）。

## 準備済みのもの（`runs/gpt/`、git管理外）

- `runs/gpt/b/syn_01` 〜 `syn_06`：条件B。`workspace/input/reference.mp4` に映像だけが入ってる（正解データは入れてない）。
- `runs/gpt/c/syn_01` 〜 `syn_06`：条件C。Claudeの比較（`runs/llm_c4`）と同じ開始シーン・同じ予算（30回）。
- 各フォルダに：
  - `prompt.txt`：GPTにそのまま貼る指示文
  - `codex_mcp.toml`：つないでいいMCPサーバー（1つだけ）

作り直すとき：
```
.venv/Scripts/python.exe baselines/gpt_codex/prepare.py b evaluation/synthetic/syn_01 runs/gpt/b/syn_01
.venv/Scripts/python.exe baselines/gpt_codex/prepare.py c syn_01 runs/gpt/c/syn_01
```

## 1回分の回し方

1. **MCPを1つだけにする。** Codexの設定（`~/.codex/config.toml` の `[mcp_servers.*]`、またはアプリの設定画面）で、
   他のMCPサーバーを全部オフにして、そのフォルダの `codex_mcp.toml` の中身だけを追加する。
   - B：`blender`（blend-ai）。先にBlenderを起動して、blend-aiのサーバーを動かしておく。
   - C：`vrecon`。
2. **作業フォルダ**を開く。
   - B：`runs/gpt/b/syn_XX/workspace`（ここ以外のファイルを読ませない。Bはシェルとファイル操作も使ってよい）
   - C：どこでもいい。ただし**シェルとファイル操作は使わせない**（vreconのツールだけ）。指示文にも書いてある。
     Codexでシェルを完全に切れない場合は、承認を「毎回確認」にして、コマンドは全部断る。
3. `prompt.txt` の中身を貼って実行。途中で口を出さない（質問されても答えず「指示どおり自分で続けて」とだけ返す）。
4. 制限時間：B は60分、C は30分を目安に止める。
5. 終わったら採点：
   ```
   .venv/Scripts/python.exe baselines/gpt_codex/finish.py b runs/gpt/b/syn_01 --model <モデル名>
   .venv/Scripts/python.exe baselines/gpt_codex/finish.py c runs/gpt/c/syn_01 --model <モデル名>
   ```
   セッションのログ（`~/.codex/sessions/` の該当ファイル）を、Bは `logs/agent.jsonl`、Cは `llm/agent.jsonl` に
   コピーしておくと、正解データを覗いていないかの検査に使える。

## 公平さのルール

- どちらにも正解データ（`gt.json`, `gt_passes/`, `meta.json`）を見せない。Bの作業フォルダには映像しか入れていない。
- Bの指示文は CineRecon L1（Claudeのときと同じ）、Cの指示文は `scripts/run_llm_repair.sh --use-budget` と同じ。
- 床のチェック模様：映像に映っているものを手がかりにするのは、どちらの条件でも許す（Bが模様から大きさを推定するのは
  正当な画像の読み取り）。採点はマスクのIoUと焦点距離の相対誤差なので、シーンの単位のずれは点数に影響しない。
  ただし、Cは道具の範囲でしか直せないので、「Bが映像から読み取れることを、Cの道具は使えていない」という差は残る。
  これは条件の違いそのものとして扱う。
- まずは各映像1回ずつ（B 6回、C 6回）。差がはっきりしなければ3回ずつに増やす。
