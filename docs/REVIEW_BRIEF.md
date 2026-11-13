# 設計レビュー依頼（2026-11-14）

このリポジトリ（vrecon）全体の設計を外から見て評価してほしい。作り直すべき所、改善すべき所、やめるべき所、
足りない検証を指摘してほしい。コードを直すのではなく、**判断材料を出すのが目的**。

- 返してほしいもの：下の「レビューの問い」への答えと、それ以外に気づいた重大な点。各指摘に
  (a) 何が問題か、(b) 根拠（ファイル・行・数字）、(c) 直すならどうするか、(d) 重さ（作り直し／改善／好み）を付ける。
- ユーザーへの説明は日本語、結論から、専門外の人にも分かる言葉で、誇張なし（CLAUDE.md）。
- レビュー中にやらないこと：GPUを使う処理（SAM・4RC・GVHMR・描画）を回さない、外部サービス（fal など）に何も送らない、
  トークンやキーのファイルを読まない、`runs/` の大きなデータを書き換えない、コミットしない（提案だけ）。
  テスト（`.venv/Scripts/python.exe -m pytest -q`、約80秒、156件）と、軽い読み取りスクリプトは回してよい。

---

## 1. プロジェクトの要約

- 目的：参考映像を、**後から編集できる Blender の3Dシーン**（カメラ・配置・動き・タイミング）に作り直し、そのシーンから
  書き出した制御映像（奥行き・輪郭など）で動画生成（V2V、MiniMax-H3 + Fun ControlNet Union）を行う。研究・非営利。
- 研究の問い（`docs/PLAN.md` 2章）
  - RQ1：構造化された診断と専用の修正ツールを与えると、素の LLM + Blender MCP（条件B）より再構成が良くなるか（条件C）。
  - RQ2：3Dで加えた編集が、生成映像にどこまで残るか。
- 最初の完成地点 M1（PLAN 3.1）：単一カットで RQ1 を示す（M1-本体）＋ RQ2 を測る（M1-V2V）。
- 利用者：Blender で短いCGを作る個人（ユーザー本人）。ユーザーの方針：**Blender シーンを完成させてから V2V**。

## 2. 読む順番

1. `CLAUDE.md`（環境・規則）
2. `docs/PLAN.md`（計画書、最優先。2章 問い、3章 範囲、6章 修正ループ、11章 評価、12章 段階、16章 見直し）
3. `docs/ARCHITECTURE.md`（責任分担・修正ループ）、`docs/CONVENTIONS.md`（座標・単位・回転・時間）
4. `docs/HANDOFF.md`（経過の記録。長い。4・5章が課題と経過、最後の節が現状）
5. `docs/DIRECTION_OPTIONS.md`（研究の方向の選択肢、ユーザー判断待ち）
6. `docs/VENUE_PLAN.md`（会場の全周3D化）、`docs/PROPS_PLAN.md`（小道具の汎用の仕組み）

## 3. 経緯（時系列の要点）

| 日付 | 何をしたか | 結果の要点 |
|---|---|---|
| 11-01〜06 | 計画、データの型（4層）、Blender ビルダー、読み戻し、制御映像、合成映像6本（正解つき）、採点、修正ループ（提案→制約→作業コピー→評価→採用）、CLI/MCP | 段階0完了。合成6本で数値探しが画角を回復 |
| 11-07〜09 | 観測の道具：SAM 3.1、4RC（最初の3D）、点の追跡（LK / CoTracker3 / TAPNext++）、MegaSaM・MoRe と比較 | 4RCを採用（物のIoUが最良）。CoTracker3は削除、追跡はLKのまま。4RCの焦点距離は12〜29%ずれ、修理で直すと他のずれを打ち消すので焦点距離は固定 |
| 11-09〜12 | 条件C（LLMが仮説を選ぶ）を Claude・GPT で、条件B（素のLLM＋blend-ai MCP）を一部 | C：Claude .827 / GPT .830、決め打ち順 .837（合成6本の平均IoU）。B：syn_01 .945・.940、syn_02 .912（**BがCを大きく上回る**、2〜3本のみ）。道具側の欠陥（カメラ採用・対応付けの抜け道）を修正 |
| 11-12 | 実写（Pexels）で通し、2Dレイヤー、入口の振り分け | 4RCが床を正しく作れない映像は修理で直らない |
| 11-13 | テニス実写（312コマ、静止カメラ・パンわずか）：GVHMR（人）、SAM、4RC、プロキシ、背景、補完、コート校正（白線→ITF寸法）、会場（LLMの配置）、制御映像 | 焦点距離2104 pxで固定、メートル単位。会場を全周に |
| 11-13〜14 | LLM の「見る」道具（view/look）、タイミング合わせ（align_timing / hold_still）、フェーズE：fal（GPT Image 2.5 Sunburst）で画面外の絵→LLMの配置、見た目の確認（restyle）、服の色、観客、ラケット・ボール、小道具の汎用化 | 6本でタイミング誤差 7.8→6.1 コマ。fal の画面外の絵は会場IoUを上げなかった（.70→.68）、判定は上がった（3/7→5〜6/7）。小道具は合成でcm台 |

## 4. 構成（コード）

```
src/vrecon/            約10,400行（.py）。bpy を呼ぶのは scene/ だけ（規則）
  schema/              データの型（Project、SceneObject、Camera、TransformKey、Evidence、status、locked …）、版つきJSON
  scene/               Blender：builder（シーン構築・描画）、character（SMPL-X骨格）、passes / color_passes / controls（制御映像）、
                       readback / ingest / handoff（人の編集の読み戻し）、look、convert、coverage
  run/                 manager（提案・制約・採用・原子的な反映）、policy（決め打ち順＝LLMなしの条件C）、service（CLI/MCP共通）、
                       look / view（LLMに見せる道具）、complete（補完）、refine
  optimize/            シルエット損失（箱の z-buffer ラスタ）、LK 点追跡損失、パターン探索、光線法（既定オフ）
  metrics/             採点（numpyのみ）：IoU、カメラ、再投影、タイミング、新しい視点、会場
  analysis/            観測→シーン：from_4d（4RC）、from_gvhmr / place_person（人）、proxies、background、planes、court /
                       to_court（コート校正）、observations、appearance（服の色）、triage、units、
                       props / prop_fit / prop_table / prop_shapes / rig_fk（小道具）
  venue/               会場の配置データ（layout）、部品生成（generate）、観測への合わせ（fit）、組み立て（build）
  layers2d/            2Dレイヤーの道
  control/core.py      制御映像（numpy/cv2/ffmpeg）。H3 の条件：24fps、32の倍数、17n+5 コマ
  cli.py、mcp_server.py
scripts/               約5,000行。実験・実行用（make_*_clips、eval_*、build_venue、llm_venue、fal_*、fit_props、props_recut、
                       export_scene_controls、camera_moves、clay_views、run_llm_repair.sh、real_pipeline.sh …）
gpu/observe/           別環境で動く観測（run_sam3、run_gvhmr、run_tapnextpp、run_megasam、run_more、gvhmr_body_mesh）
gpu/h3_control/        V2V（H3）を借りたGPUで回すスクリプト（未実行）
baselines/             条件Bの実行キット（Claude＋MCP、GPT＋Codex）
evaluation/            合成映像（synthetic 6本、props 4本、phase1a、venue）、修理ベンチマーク、thresholds.yaml（下書き）
tests/                 156件
```

## 5. テニスでの流れ（今の最終形、V2V直前）

| 段 | 道具 | 出力 |
|---|---|---|
| 1 観測 | SAM 3.1（物のマスク、`gpu/observe/run_sam3.py`）、4RC（点群・カメラ）、GVHMR（人の骨格、WSL）、SMPL-X体メッシュ | `runs/human/tennis/` |
| 2 校正 | 白線→ITFコート（`analysis/court.py`）。焦点距離固定、向きだけ毎コマ。4RCの点を校正カメラの光線へ移す | コート座標・メートル |
| 3 人 | GVHMR をシーンのカメラの光線上に置く（足が床に乗る縮尺）。他の人は立ち姿・座り姿の人形 | `analysis/place_person.py`、`proxies.py` |
| 4 会場 | 観察のまとめ → LLM（claude -p、Readのみ）が会場の配置JSON → 観測に合わせて部品を生成。画面外は fal の絵を参考に LLM が仮説で（status=hypothesis） | `venue/`、`scripts/llm_venue.py`、`fal_outside.py` |
| 5 補完 | 見えていない所を延長・複製。元の視点で見えていた物を隠さない検査 | `run/complete.py` |
| 6 タイミング | 観測した動き出しにキーを合わせる、開始前は静止 | `run/service.py` |
| 7 小道具 | 手の周りの窓でSAM（`props_recut.py`）→ 状態（持たれている／宙／見えない）→ 当てはめ（形のIoU・放物線）→ 見えない区間は物理＋仮説 | `scripts/fit_props.py` |
| 8 書き出し | カメラの動き5種 × 制御映像5種（depth、canny、clay、id、look） | `runs/venue/tennis/fal2c_props/final/`（`runs/` はgit外） |
| 9 見た目の確認 | 書き出しの1コマを fal で写真風に（形が保たれるか、物が別の物に化けないか） | `fal2c_props/restyle_*/sheet.png` |

## 6. 主な数字（詳しくは各文書）

- 条件C（合成6本、平均IoU）：決め打ち順 .837、Claude .827、GPT .830。条件B：syn_01 .945（Claude）/ .940（GPT）、syn_02 .912（GPT）。
  GPTのB残り4本はクレジット上限で中断（無効扱い、やり直し待ち）。
- 修理ベンチマーク（正解にずれを入れて修理）：IoU 平均 .906。4RCスタート（焦点距離固定）：.840〜.890。
- タイミング（6本、looker → +align → +hold）：7.8 → 6.75 → 6.08 コマ。syn_04 は17のまま（カメラの横移動が再現されず物が動く）。
- 会場（合成 ven_01、新しい視点の観客席IoU）：見えた物だけ .08、LLM .62/.65、手作業 .88。fal の画面外の絵ありで平均 .68（なし .70）。
- 小道具（合成）：トス 1.6〜3 cm、投げてキャッチ 1.8 cm、置いた後 0.7 cm、手の中で静止（接触なし）35〜48 cm、
  打った後 0.5〜1.1 m、ラケットの向き 1.2〜2.0°（がたつき 0.2°）。
- テニス実写：コートの白線への乗り中央値 .99、選手の身長 1.84 m（本人 1.85 m）、ラケット 228/312 コマを形で当てはめ。

## 7. 主な決定とその理由

| 決定 | 理由 | 出所 |
|---|---|---|
| 最初の3Dは4RC | MegaSaM・MoRe より合成6本で物のIoUが良い | HANDOFF 5章3 |
| 4RCスタートでは焦点距離を修理しない | 焦点距離を動かすと他のずれを打ち消して悪化（.23） | 同上 |
| 点追跡は LK | TAPNext++ は合成で劣る、CoTracker3 は動きを小さく追う | 同上 |
| カメラ提案は床・壁の証拠だけで採用、全体損失が上がらないこと（許容0.03） | LLM が損失を下げつつ正解から離れた抜け道を塞ぐ | HANDOFF、`run/policy.py` |
| 規格で決まる物（コート・ネット）は規格の位置 | 4RC のネットが +1.9 m ずれた | VENUE_PLAN フェーズB |
| 見えない所は status=hypothesis、根拠を記録 | 観測と推測を区別する（規則） | PLAN、PROPS_PLAN |
| 画像生成はシーンの参考だけ（形の出所はシーン） | 生成の絵は観測ではない | VENUE_PLAN フェーズE |
| 正解データは採点専用、LLM・エージェントに渡さない | 評価の公平さ | CLAUDE.md、11.8 |
| 小道具は状態（持つ／宙／見えない）で扱う | 状態を決めずに当てはめると嘘の答え（持ったままのボールに放物線、50 px ずれ） | PROPS_PLAN |

## 8. 自己評価：気になっている所（レビューで特に見てほしい）

1. **RQ1 の土台が揺らいでいる。** B（LLM が Blender を直接いじり、映像を自分で見る）が C（こちらの道具）を大きく上回った
   （2〜3本）。C の道具はマスクの重なりと点の追跡しか見ておらず、4RC の開始シーンから大きく動けない。
   `docs/DIRECTION_OPTIONS.md` の選択肢（道具を強くする／LLM に「見る係」をさせる／主張を絞る／B を土台にする）のどれが良いか。
2. **テニス以降、研究の比較（RQ1）から離れた作業が多い。** 会場の全周化、fal、観客、服、小道具は「V2V で使えるシーン」には
   必要だが、RQ1 の評価（合成6本と条件B/C）とはつながっていない。M1 に対して優先順位が正しいか。
3. **scripts/ の肥大化。** 約5,000行の実験スクリプトに処理の本体が入っているものがある（例：`proxies_scene.py`、
   `export_scene_controls.py`、`build_venue.py`、`fit_props.py`）。CLI（`vrecon`）や MCP から使えるのは一部だけ。
   パイプラインとして再実行できる形になっているか。テニス1本を最初から作り直す手順は、HANDOFF の記録を辿らないと分からない。
4. **テニス専用・映像専用の値。** 焦点距離固定、コートの規格、会場の配置、`tennis_after.json` の行き先、観客密度 0.08 など。
   汎用の仕組みと、映像ごとの設定・仮説の境目が文書でしか管理されていない。
5. **仮説（hypothesis）の量が多い。** 会場の見えない側、観客、ボールの見えない区間、ラケットの見えないコマ。シーンの中で
   evidence と status は付いているが、V2V に渡すときは区別されない。仮説の部分を制御映像でどう扱うべきか（弱める・外す・そのまま）。
6. **評価の物差し。** 合成映像は市松模様・縞の床と壁で、校正しやすい（B に有利かもしれない）。実写の正解はない。
   会場・小道具は自前の合成で測っており、作った人と採点した人が同じ。
7. **カメラの横移動が再現できない（syn_04）。** 4RC・修理ともに、カメラの平行移動を物の移動で代用してしまう。テニスは
   カメラがほぼ動かないので影響なしとしたが、一般の映像では致命的になりうる。
8. **データの型と描画の制約。** 物は箱かメッシュ、人は SMPL-X（姿勢補正なし）、小道具は簡単な形。V2V の制御として十分か。
9. **8 GB VRAM。** SAM と描画を同時に回すと数十倍遅くなった。観測はローカル、生成は借りたGPU。分担は妥当か。
10. **時間の扱い。** テニス映像は 5 コマに 1 コマが複製（24→30 fps の水増し）。小道具では本当の時刻を付けたが、
    人（GVHMR）・タイミング・制御映像（24 fps に変換）は複製を意識していない。

## 9. レビューの問い

1. 全体の設計（4層のデータ、修正ループ、bpy の隔離、CLI/MCP、観測は別環境）は、目的（RQ1・RQ2・編集できるシーン）に合っているか。
   作り直すなら何をどう変えるか。
2. RQ1：B が C を上回る結果を踏まえて、研究の設計（条件B/Cの定義、比較の公平さ、主張）をどう直すべきか。
   `DIRECTION_OPTIONS.md` の選択肢への意見、または別案。
3. M1 までの残り作業の優先順位は妥当か（V2V＝M1-V2V、条件Bのやり直し、基準値の固定、実写の検証映像12本）。
   やめるべき作業、後回しにすべき作業はあるか。
4. パイプラインの再現性：テニスを最初から作り直せる形か。scripts/ と src/ の分け方、設定（映像ごとの値・仮説）の持ち方を
   どう整理すべきか。
5. 観測と仮説の区別（status / evidence）は十分か。V2V に渡す段で仮説をどう扱うべきか。
6. 評価：合成映像の作り方、物差し、基準値（`evaluation/thresholds.yaml`、`evaluation/THRESHOLDS_PROPOSAL.md`）の問題点。
7. 汎用化：会場（`docs/VENUE_PLAN.md`）と小道具（`docs/PROPS_PLAN.md`）の仕組みは、テニス以外の映像に広げられる形か。
8. コードの質：テストで守れていない重要な所、壊れやすい所、重複。
9. V2V（`gpu/h3_control/`、制御映像の種類・解像度・長さ）に進む前に、確かめておくべきことはあるか。

## 10. 環境と制約（レビューの前提）

- Windows 11、i7-13700H、32 GB RAM、RTX 4060 Laptop 8 GB。Blender 5.2、Python 3.13（`.venv`、bpy 5.2.3）。
- リポジトリのパスに日本語（OneDrive 配下）：OpenCV のファイル関数は使えない（`control/core.py` の読み書きを使う）。
  文字コードは常に utf-8。
- 観測の道具は別環境（`%LOCALAPPDATA%\vrecon\envs\...`、WSL の GVHMR）。V2V（H3）は約80 GB 必要で借りたGPU。
- ライセンス：4RC（S-Lab、非商用）、SMPL-X（体のデータは `runs/` のみ）、MoRe（表記なし、不採用）。研究・非営利。
- 外部サービス：fal（画像生成、`X-Fal-Store-IO: 0`、保存なし）、claude -p（LLM）。正解データは送らない。
- `runs/` と大きな映像は git 外。合成の評価映像は小さいのでコミットしている。

## 11. すぐ見られる成果物

- テニスの最終シーン：`runs/venue/tennis/fal2c_props/final/scene.blend`（git外、ローカルのみ）と `project.json`
- 確認画像：`runs/venue/tennis/fal2c_props/views_all.png`（4コマ×3視点）、`restyle_120/sheet.png`、`restyle_240/sheet.png`
- 合成の採点結果：`evaluation/repair_benchmark/`、`runs/props/*.json`、`runs/venue/ven_01/*/scores.json`
- コミット履歴：`git log --oneline`（112件）。最近の変更の要約はコミットメッセージにある。
