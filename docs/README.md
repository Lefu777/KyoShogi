# KyoShogi

dlshogiの互換エンジンで、dlshogiとほぼ同等の探索速度[^search_speed]・棋力[^elo_rating]を持ち、dlshogi用のmodelを使うことが出来ます。

dlshogiと探索経路・評価値ともに完全に一致することを確認しています。[^match_search_dl]

第4回電竜戦において、予選10位, A級10位となりました。

## 特徴
[第4回電竜戦](https://denryu-sen.jp/)のアピール文書にて追って記します(予定)ので、そちらを参照して下さい。

## ビルド環境[^build_env]
* Windows11
* Visual Studio 2022
* CUDA 11.8
* cuDNN 8.7.0[^cuDNN_version]
* TensorRT 8.5

## 注意点等
- 実行は自己責任でお願いします。本ソフトの実行によって発生した如何なる損害の責任も取りません。
- 完全な動作は保証していません。可能な限りデバッグしていますが、棋力に関わる致命的なバグがある可能性があります。
- 開始直後の探索が遅いです。対策は簡単なので、気が向いたら対策します。
- あくまで勉強用に書いたコードです。ソースコードは本当に汚いです。デバッグ用のコードのままの箇所もあります。

## 謝辞
* [dlshogi](https://github.com/TadaoYamaoka/DeepLearningShogi)
  - hcpe系の教師の読み込みとparse, bit入力特徴量の作成 に使用しています。
  - ucb_score の計算(fpu reduction, FastLog, etc...), Parallel PUCT, PvMateSearcher, 推論部 の実装を参考にしています。
* [cshogi](https://github.com/TadaoYamaoka/cshogi)
  - 盤面の管理, 合法手生成, 入力特徴量作成に使用しています。
  - dfpnの実装を参考にしています。
* [Apery](https://github.com/HiraokaTakuya/apery)
  - 盤面の管理, 合法手生成に使用しています。
* [YaneuraOu](https://github.com/yaneurao/YaneuraOu)
  - 盤面の管理, 合法手生成に使用しています。
* [KomoringHeights](https://github.com/komori-n/KomoringHeights)
  - 証明駒の計算の実装を参考にしています。

## ライセンス
ライセンスはGPL3ライセンスとします。

[^search_speed]:平手局面において、jc26 がTODO、dlshogi がTODO です。
[^elo_rating]: たややん互角局面24手目開始で、水匠5 1500万ノード(8スレ, hash:16GB) に対して、jc26 6万ノードでKyoShogi がTODO、dlshogi がTODOでした。但しこれはV0.2.5においてであり、V0.2.7 においては保証しません。
[^match_search_dl]: ある局面において、プレイアウト制限約10万ノードで、任意のプレイアウトの経路が一致し、任意のプレイアウトの任意の局面においてucb_score が完全に一致する事を確認しています。但しこれはV0.2.5においてであり、V0.2.7においては保証しません。
[^build_env]:必要なライブラリはdlshogiと全く同じなので、Windows 11であればdlshogi が動けば動く可能性が高いです。
[^cuDNN_version]:ちょっとあやふやです。8.6.0の可能性があります。