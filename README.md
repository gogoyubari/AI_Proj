# ① faster-whisperの導入編
***```whisper_test.py```*** は faster-whisperの動作確認用として 現在開発中のシステムから基本部分のみを抜き出したものです。
- 設定パラメーターによる挙動の確認
- .srtファイルを出力 -> メディアプレーヤー等で簡易確認
  
## faster-whisperについて
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) は 本家[openai/whisper](https://github.com/openai/whisper) の推論部分（transformer）を [Ctranslate2](https://github.com/OpenNMT/CTranslate2/)（C++で記述）で置き換えたものです。本家版に対してより高速・省メモリとなっています。
- 学習モデルは 本家のモデルをCtranslate2用にコンバートされたものを使います。

## GPU環境構築
1. [CUDA 12.6](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) をインストールします。

2. CUDAのバージョン確認
```ps
PS > nvcc -V
```

3. [cuDNN 9.5.1](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=12) をダウンロード後 解凍した```bin\```フォルダのすべての.dllファイルを```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\```フォルダにコピーします。
> [!IMPORTANT]
> ***CUDA 12.6***に対応するcuDNNのバージョンは***9.5.1***となります。
>
4. システム環境変数にCUDA_PATHが設定されているか確認
```ps
PS> echo $env:CUDA_PATH
```
## faster-whisperのインストール
```ps
(.venv) PS> pip install faster-whisper
```
<details>
<summary>Other installation methods (click to expand)</summary>
  
### initial_prompt 修正パッチ版のインストール
  
```ps
(.venv) PS> pip install --force-reinstall "faster-whisper @ https://github.com/gogoyubari/faster-whisper/archive/refs/heads/master.tar.gz"
```

</details>

## テストプログラムの実行
```ps
(.venv) PS> pip install pysubs2 term-printer
```
```ps
(.venv) PS> python whisper_test.py C:\[メディアファイルのフォルダ]\test.mp4
```
### whisper_test.py の説明
> [!NOTE]
> ★印 -> whisperの出力に大きく影響するパラメータ

```py
import os
os.environ["CT2_VERBOSE"] = "2" # Ctranslate2のデバッグ情報を表示します（"2"でlog_lebel=DEBUG）
os.environ["CT2_CUDA_ALLOCATOR"] = "cub_caching" # デフォルトのCUDAキャッシングにメモリリークの疑い？（調査中）別のGPUキャシュ方法を指定しています
os.environ["CT2_CUDA_CACHING_ALLOCATOR_CONFIG"] = "4,3,15,3221225471"
os.environ["CUDNN_LOGLEVEL_DBG"] = "2" # cuDNNのデバッグ情報を表示します（"2"でlog_lebel=WARNING）
os.environ["CUDNN_LOGDEST_DBG"] = "stderr" # cuDNNのデバッグ情報の出力先

import time
import sys
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from ctranslate2 import set_random_seed
import logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

import pysubs2 # whisper出力を.srtフォーマットで出力するライブラリ
from term_printer import Color, cprint # 色付きのprint()関数 cprint()を使う
import re # 正規表現（文字列置換など）のライブラリ
import dataclasses # whisper出力をdict型で受け取る

def tsToStr(ts): # 秒（int）を"HH:MM:SS.SSS"（str）に変換する関数
    try:
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = ts % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    except:
        return f"00:00:00.000"
    
def whisper(filepath): # whisperメインルーチン
    # model_size_or_path="distil-large-v3"
    model_size_or_path="large-v3-turbo" # それぞれのモデルは一長一短 ★

    model = WhisperModel(
        model_size_or_path=model_size_or_path,
        device="cuda",
        compute_type="float16", # 量子化タイプ "int8_float16"にすると軽くなるが精度が落ちる ★
        download_root="model", # modelのダウンロード場所を指定
    ) # 初回実行時にのみ モデルの実体がデフォルトで"C:\User\[ユーザー]\.cache\huggingface\hub\"にダウンロードされる（ダウンロード場所は別途指定可能）
    set_random_seed(0) # Ctranslate2の乱数シードを固定する

    with open('initial_prompt_J.txt', encoding="utf_8_sig") as f: # "initial_prompt"を準備する
        lines = f.read().strip().split('\n')
        init_prompt = ''.join(['<|0.00|> ' + line.strip() + '<|0.00|>' for line in lines]) # プロンプトにダミーのタイムスタンプtokenを挿入
        print("initial_prompt has been loaded.")

    segments, info = model.transcribe(
        audio = filepath, # 内部でpcm/16bit/16KHz/monoのオーディオデータに変換するので、入力ファイルのフォーマットは何でもOK
        language = "ja", # 言語を明示的に指定 "en"...
        log_progress = False, # Trueでプログレスバー表示
        temperature = [n/10 for n in range(11)], # 再試行時 temperatureを0.1ステップで増加させる
        # beam_size = 5, # 探索の深度 ★
        # condition_on_previous_text = True, # 一つ前の結果をプロンプトに反映するかどうか ★
        prompt_reset_on_temperature = 0.5, # どこまで崩壊したらパラメータをリセットするか ★
        initial_prompt = init_prompt, # 出力のスタイルに影響 ★
        word_timestamps = True, # 単語単位でタイムスタンプ Startとendのタイムが正確になる ★
        vad_filter = True, # 素材によって有利不利が分かれる ★
        vad_parameters = VadOptions( # vad_filter=Trueのときに有効 "#"でコメントアウトした設定はデフォルト値となる ★
            # onset = 0.4,
            # offset = 0.263,
            # min_speech_duration_ms = 0,
            # max_speech_duration_s = 2.0,
            # min_silence_duration_ms = 500,
            speech_pad_ms = 800
        ),
        hallucination_silence_threshold = 0.2 # ハルシネーション検出の閾値 ★
    )

    d = []
    start = time.time() # 実行開始時刻セット

    for segment in segments: # "segments"はPythonのイテレータなので ここで初めて実行される
        if re.match(r'.*[\.\,\?\!]$', segment.text):
            col = Color.WHITE
        else:　# 例えば 末尾が期待する文末でなかったとき黄色でprint()してみる
            col = Color.YELLOW
        cprint(f"[{segment.id}] {tsToStr(segment.start)} {tsToStr(segment.end)} {segment.text}", attrs=[col]) # デバッグ用出力
        d.append(dataclasses.asdict(segment))
    else:
      print(f"time: {time.time()-start}") # 処理時間計測

    subs = pysubs2.load_from_whisper(d)
    base_name = os.path.splitext(os.path.basename(filepath))[0] # .srtのファイル名は入力ファイル名を流用
    subs.save(f"{base_name}.srt", format_='srt', encoding="utf_8_sig") # .srtの文字コードはUTF-8 BOM付（Windowsメディアプレーヤーの都合）

if __name__ == '__main__':
    args = sys.argv

    if len(args) > 1: # コマンド引数に入力ファイル名があれば
        try:
            whisper(args[1])
        except KeyboardInterrupt:
            print("main_thread has been interrupted.")
        except Exception as e:
            print(e)
```
## 観察される問題点
- 文末が句点".,!?"にならない
- 突然大文字だけになる
- 出力が細かく断片化
- ハルシネーションが継続

## 簡易プレビュー
- ```VLCプレーヤー```や```Windowsメディアプレーヤー```などのメニューでsrtファイルを指定すれば簡易プレビューできます。
- オープンソースの代表的な字幕ソフトとしては [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit/releases) がおすすめです。
