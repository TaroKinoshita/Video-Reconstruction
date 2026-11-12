# 条件B（GPT）のばらつき測定：依頼文

（以下をそのままGPTに渡す。リポジトリ：C:\Users\kinos\OneDrive\デスクトップ\MyProject\Video-Reconstruction）

---

前回の条件B 6本と、まったく同じやり方で、追加の4試行を実行してください。
目的は、Bを繰り返したときに結果がどのくらいばらつくかを見ることです。

## 試行（4本、1本ずつ順番に）

| 試行 | フォルダ | 映像 |
|---|---|---|
| 1 | `runs/gpt/b_repeat/syn_01_r2` | syn_01 |
| 2 | `runs/gpt/b_repeat/syn_04_r2` | syn_04 |
| 3 | `runs/gpt/b_repeat/syn_01_r3` | syn_01 |
| 4 | `runs/gpt/b_repeat/syn_04_r3` | syn_04 |

各フォルダには、`prompt.txt`、`codex_mcp.toml`、`clip.txt`、空の `workspace/` と `logs/` が準備済みです。

## 前回と同じにする点（変えないでください）

- モデル `gpt-6.1-sol`、推論 `medium`、Codex CLI（`codex exec`、`--ignore-user-config`）、制限時間 60分。
- 指示文は各フォルダの `prompt.txt` をそのまま渡す。MCPは blend-ai だけ。作業フォルダは各フォルダの `workspace`。
- 前回と同じ起動の仕組み（`condition_b_launcher`）を使う。
- 1本ごとに新しいセッションで始める。前回の試行の結果、ログ、レポートは、エージェントに見せない。
- 毎回、試行の前にBlenderを空にする（こちら側で行う）。
- 正解データ（`gt.json`、`gt_passes/`、`meta.json`）と、`runs/` の他のフォルダは読ませない。

## 採点

```
.venv/Scripts/python.exe baselines/gpt_codex/finish.py b runs/gpt/b_repeat/syn_XX_rN --model gpt-6.1-sol --minutes 60
```

## 失敗の扱い

- 利用上限や起動の不具合で止まった試行は、無効として別に保存する。
- 無効になったら、同じフォルダを空に戻して、やり直す。
- エージェント自身の失敗（成果物がない、時間切れ）は、そのまま有効な結果として記録する。

## 最後に出してほしいもの

- 4試行の表：IoU、焦点距離誤差、タイミング誤差、実行時間、成果物の有無。
- 前回の syn_01（.940）と syn_04（.712）を加えた、映像ごとの3回の平均と標準偏差。
- 監査：blend-ai 以外のMCPを呼んでいないこと。workspace の外や正解データに触れていないこと。
