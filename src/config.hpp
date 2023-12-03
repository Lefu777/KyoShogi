#pragma once
#include <string>

// ==================================================
// exclusive mode
// ==================================================
//// 以下のマクロは同時に定義してはならない。排他的である。
// 対局用
//#define USI_ENGINE

// 教師生成用(現状random player のみのサポート)
//#define GEN_SFEN

// test
#define TEST

// ==================================================
// exclusive mode of exclusive mode
// ==================================================

//// この中で排他的
#ifdef USI_ENGINE
// random player
//#define RANDOM_PLAYER

// uct random player
//#define UCT_RANDOM_PLAYER

// puct player
//#define PUCT_PLAYER

// parallel puct player
//#define PARALLEL_PUCT_PLAYER
#endif

//// この中で排他的
#ifdef TEST
// bitboard test1
//#define BB_TEST1

// bitboard test2
//#define BB_TEST2

// 指し手生成祭り
//#define MOVE_GENERATION_FESTIVAL

// vector test
//#define VECTOR_TEST

// read teacher test
//#define READ_TEACHER_TEST

// square test
//#define SQ_TEST

// dataloader test
//#define DATALOADER_TEST

// NNTensorRT test
//#define NN_TENSORRT_TEST

// make input feature test
//#define MIF_TEST

// parse csa test
//#define PARSE_CSA_TEST

// load dlshogi teacher
//#define LOAD_DLSHOGI_TEACHER_TEST

// read yobook test
//#define READ_YOBOOK_TEST

// make_inpu_features() FEATURE_V1 test
//#define MIF_V1_TEST

// make_inpu_features() FEATURE_V2 test
#define MIF_V2_TEST

// dfpn test
//#define DFPN_TEST

// cshogi の__Board::mateMove() のdebug 用。
//#define CSHOGI_BOARD_MATE_MOVE_TEST

// board の仕様を確認
//#define BOARD_TEST

#endif

// ==================================================
// independent mode
// ==================================================

//// debug モード

//#define DEBUG
// デッドロックを検知して警告
//#define DEBUG_LOCK_SAFE
// デッドロックを検知して警告
//#define DEBUG_UNLOCK_SAFE
// 本来の正解手順の手数が不当に低く計算された所為で、他の筋がpv として出てきてしまい、どの局面がバグってるか分からないので、
// 特定の局面では、特定の手のみを採用するように変更。
//#define DEBUG_GETPV_20231026_0
// root でbreak した時に情報を表示する。スレッドセーフ。
//#define DEBUG_PRINT_WHEN_ROOT_BREAK_20231101_0
// root node でpn/dn の計算が終わるごとに、0番目のスレッドで情報を出力する
//#define DEBUG_PRINT_ROOT_INFO_20231103
// 探索終了後にroot entry の情報を出力する。
//#define DEBUG_PRINT_WHEN_SEARCH_DONE_20231101_0

//#define DEBGU_PUCT_20231125_0
//#define DEBGU_PUCT_20231125_0_COND (g_playout_count >= 2024)
//#define DEBGU_PUCT_20231125_1
//#define DEBGU_PUCT_20231125_2
//#define DEBGU_PUCT_20231125_3    // fast log についての統計情報を表示

//// 入力特徴量

#ifdef PUCT_PLAYER
// python-dlshogi2 のモデル(checkpoint.pth)で推論する。(入力特徴量の変更)
//#define PYTHON_DLSHOGI2

// 一番初めに用いた入力特徴量
#define FEATURE_V0

#elif defined PARALLEL_PUCT_PLAYER

// cshogi.h の入力特徴量関連の定数と、入力特徴量作成の関数 で使用	
// 一番初めに用いた入力特徴量
//#define FEATURE_V0

// FEATURE_V0  + 王手
//#define FEATURE_V1

// dlshogi
#define FEATURE_V2

// 入力特徴量をbit で転送する場合。
#if defined(FEATURE_V2)
#define USE_PACKED_FEATURE
#endif

#else
// USI_ENGINE で無いときも、cshogi.h が必要な時はこれを定義していないと。。。
//#define FEATURE_V1

// dlshogi
#define FEATURE_V2

#endif

//// dfpn の種類

// cshogi のdfpn を使用
//#define DFPN_CSHOGI
// dlshogi のdfpn を使用
//#define DFPN_DLSHOGI
// 並列呼出しに対応しようとしたけどしなかった版(自分でも何変えたか分かってない)
//#define DFPN_PARALLEL
// dlshogi由来のバグを修正 && lock(), unlock() を追加しただけで使用はしてない
//#define DFPN_PARALLEL2
// 並列化
//#define DFPN_PARALLEL3
// 並列化 + 置換表set 出来るように
//#define DFPN_PARALLEL4
// dlshogi のdfpn が潜在的に抱えているバグを修正。
#define DFPN_PARALLEL5

//// dfnp mod

// expand and compute pn(n) and dn(n) の時、中途半端に代入せずに、
// 最後にまとめて代入する。途中で何度もlock, unlock するのを防止。
#define DFPN_MOD_V0

// 3手詰 + 2手詰めを簡素化
// -> 遅くなったので削除
//#define DFPN_MOD_V1        // 1手詰め
//#define DFPN_MOD_V1_0    // 1手詰め + 2手詰め

// TODO: pn=0 を手数違いで最大手数引き分けでdn=0 と上書きするのはダメだけど、
//       それ以外の箇所は別にこれせんでええ気がしてる。(一旦は念の為全部。)
// pn に上書きしてよいのは、pn != 0 の時のみ
#define DFPN_MOD_V2
// dn に上書きしてよいのは、dn != 0 の時のみ
#define DFPN_MOD_V3

// 古い世代を見つけた時も残りも走査する。
// -> 未検証 & 未実装
// -> cluster は前から使っていくんだからこれ意味ないよね
//#define DFPN_MOD_V4
// AndNode におけるpp (証明駒) の計算を変更
// -> 私が間違っていた。
//#define DFPN_MOD_V5

// dfpn_parallel5 にて独自の処理を無効化する。
// このマクロを定義することで、一部の未検証のコードをdlshogi の実装にする。(切り分けに使用)
//#define DFPN_MOD_V6

// 反証駒の実装を修正
// -> "後手の駒を先手の駒で表す" ではなく、
//    "先手が持っていても後手を詰ませられない駒の集合" とする。
//    極大を目指すことは変わらない。
//    更新をOrで行うことが変わる。
//    (dlshogi の実装では、And で反証駒を更新しており、
//    先手の持ち駒が増えないタイミングで、反証駒=="(実質的な)先手の駒" が増えていた。
//    これは優越関係を利用する上で致命的なはずである。
//    ex> dlshogi の実装で起こり得ること。
//    以下の筋が不詰みとする。
//        Or : 銀打ち, 先手持ち駒=銀1枚, 反証駒=銀1枚
//        And: 銀取り, 先手持ち駒=銀0枚, 反証駒=銀1枚    // :=Dpos
//        Or : hoge  , 先手持ち駒=銀0枚, 反証駒=銀0枚
//    以下の筋が詰みとする。
//       Or : 銀打ち, 先手持ち駒=銀2枚, 証明駒=銀2枚
//       And: 銀取り, 先手持ち駒=銀1枚, 証明駒=銀1枚    // :=Ppos
//       Or : hoge  , 先手持ち駒=銀1枚, 証明駒=銀1枚
//    この時、
//    Dpos, Ppos のentry.hand は衝突しており、
//    優越関係を満たす場合にそれを活用するため、
//    どちらか先に現れた局面の結果が採用されてしまう。
//    (先手の持ち駒の更新と、反証駒の更新 に手数のラグがあることで、
//    そのラグの間(つまりDpos) が不適切な値(本当はdp=0枚(@20231129))となっている。)
// : 追記(忘れないように補足)(@20231129)
//   上記の例は、And の局面で攻め側の銀が0枚なら耐えていて詰まないが、銀が1枚あると詰んでしまうという状態。
//   不詰みの場合、反証駒の銀=0枚になるまでに手数的にラグがあり、先手が銀を打ってすぐにdpでの銀の枚数が減らない。
//   つまり、dp が実際の集合よりも大きくなってしまい、pn に侵食してしまう。
#define DFPN_MOD_V7
// num_searched に合わせてth_child を変える
#define DFPN_MOD_V8
// num_searched に合わせてeps を変える
//#define DFPN_MOD_V9
// 置換表の再利用。
#define DFPN_MOD_V10

//// その他

// select_max_ucb_child にて、子ノードがis_draw ならreward_mean=0.5に。
// -> あ、いや、これ意味ねぇーわ。
//#define CONSIDER_IS_DRAW

// 色々表示する
//#define VERBOSE

// NOTE
//     : コンパイル時定数であるNO_EXTEND_TIME_PLY_LIMIT は、
//       コンパイル時にoffset を足す。
//     : 変数で且つUSI_Option である、g_div_init, g_book_moves は、
//       毎度試合前にsetoption してくれるので、isready でoffset を加える。
//       (但しこれは、GUI 側が対局前に毎度setoption してからisready を送ってくれて、
//        且つ内部で値を弄る前の値を、記憶しておいてくれる前提の実装である。)
//     : g_draw_ply は何もしなくて良い。
//       (∵512手引き分けは最初の4手も入れてなので、g_draw_ply=512 とした時のg_draw_ply とboard.ply() の基準点(原点)は等しい。)
//       (恐らく、これを含まないとすると、実装によっては面倒なことになるソフトが出てくるのでこうしたんだろうね。)
// 第4回電竜戦では最初に4手、玉の往復をする。
// それに対応して、オプションにもその分だけoffset を足す。
//#define IS_DENRYUSEN_PLY_OFFSET

// cshogi をcython でbuild する場合、cython では不要な関数を定義しないようにする。
// (過去の負の遺産で、cshogi のbuild が別のプロジェクトとなっている。
// 向こうのプロジェクトにコピーするファイルは最小限にしたい。
// "python で呼び出す必要のない関数"の為に無駄にコピーするヘッダを増やしたくない。)
//#define FOR_BUILD_CSHOGI


// ==================================================
// const
// ==================================================
// 勝率と評価値の変換時の定数
// この名前は適切じゃない気もするけど、私が分かるのでok
constexpr int PONANZA = 756;

// IS_DENRYUSEN_PLY_OFFSET 参照
constexpr int DENRYUSEN_PLY_OFFSET = 4;

// ==================================================
// utils
// ==================================================
inline std::string get_dfpn_mode() {
#if defined DFPN_CSHOGI
	return "DFPN_CSHOGI";
#elif defined DFPN_DLSHOGI
	return "DFPN_DLSHOGI";
#elif defined DFPN_PARALLEL
	return "DFPN_PARALLEL";
#elif defined DFPN_PARALLEL2
	return "DFPN_PARALLEL2";
#elif defined DFPN_PARALLEL3
	return "DFPN_PARALLEL3";
#elif defined DFPN_PARALLEL4
	return "DFPN_PARALLEL4";
#elif defined DFPN_PARALLEL5
	return "DFPN_PARALLEL5";
#else
	return "NOT_DEFINED";
#endif
}

// feature をint に
inline int feature_ver() {
#ifdef FEATURE_V0
	return 0;
#elif defined FEATURE_V1
	return 1;
#elif defined FEATURE_V2
	return 2;
#else
	return -1;
#endif
}

// feature をstring に
inline std::string get_feature_mode() {
#if defined FEATURE_V0
	return "FEATURE_V0";
#elif defined FEATURE_V1
	return "FEATURE_V1";
#elif defined FEATURE_V2
	return "FEATURE_V2";
#else
	return "NOT_DEFINED";
#endif
}

// ==================================================
// path
// ==================================================
/*
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include;C:\MyLib\TensorRT\TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6\TensorRT-8.5.1.7\include;


C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64;C:\MyLib\TensorRT\TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6\TensorRT-8.5.1.7\lib
*/